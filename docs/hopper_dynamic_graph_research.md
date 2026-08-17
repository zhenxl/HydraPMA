# 从 H20 Profiling 到 Hopper 动态图架构

本文从 HydraPMA 在 NVIDIA H20 上的第一轮实测出发，分析当前 GPU 动态图数据结构的主要瓶颈，随后整理可从 CPU 动态图系统迁移到 Hopper 的优化思想，并给出一个可验证的研究架构和实验路线。

原始数据和更细的计数器说明见 [H20 初始结果](h20_initial_results.md)。本地 `results/` 目录保留完整 CSV、JSON、Nsight Systems trace 和 Nsight Compute report。

## 1. 第一轮 H20 Profiling 结果

### 1.1 实验环境与基线状态

- GPU：NVIDIA H20，Compute Capability 9.0，78 SM。
- CUDA：12.9。
- Driver：575.57.08。
- 使用单张空闲 H20，避免与在线推理任务共享 GPU。
- segment benchmark 覆盖 scalar、`cp.async` 和 Hopper bulk-async/TMA 三条路径。
- SlabHash 固定在 commit `eec7135a...`，经过只修改构建系统的 SM90 patch 后原生编译运行。

原版 GPMA/LPMA 使用旧 CUDA Dynamic Parallelism CDP1，并在 device code 中调用 `cudaDeviceSynchronize()`。CUDA 12.9 在 SM90 上不再支持该调用，加载兼容 PTX 时返回：

```text
cudaErrorUnsupportedDevSideSync:
the provided PTX contains unsupported call to cudaDeviceSynchronize
```

因此原版 GPMA/LPMA 被记录为 H20 兼容性失败，而不是通过删除同步、改变算法语义后报告性能。

### 1.2 PMA segment sweep

完整 sweep 包含：

- 10 个 segment size：256 B 到 96 KiB。
- 3 个 density。
- 5 次进程级重复。
- 3 种 movement backend。
- 共 450 条原始 mode 记录，correctness 全部通过。

稳态结果的主要现象：

- `cp.async` 在大多数点上最快。
- TMA 只在 16 KiB 附近出现约 2.3%-2.4% 的窄幅优势。
- 32 KiB 及以上 TMA 再次落后。
- 当前 bulk transaction 最大为 16 KiB，32 KiB segment 至少需要两次 load transaction 和两次 store transaction。

这表明 segment size 不能直接等同于 TMA tile size。较大的 PMA segment 应被拆成固定大小 tile 并流水执行，而不是为整个 segment 分配双份 shared memory。

### 1.3 SlabHash 吞吐基线

Mode 0，4,194,304 keys 和 queries，expected chain 为 0.6，5 次迭代：

| 指标 | 结果 |
|---|---:|
| Load factor | 0.55 |
| Build rate | 27.81 M elements/s |
| Singleton search | 10,588.08 M queries/s |
| Bulk search | 12,355.96 M queries/s |

Mode 3 使用 256 Ki-operation batch、3 个初始化插入 batch 和 1 个 mixed batch。不同 expected-chain 样本下，驱动报告的 mixed rate 为 5,027.82-7,668.02 M operations/s。

这些数字用于建立可运行对照组；系统研究仍需将数据生成、初始化、结构更新和查询分别计时。

### 1.4 Nsight Systems：端到端首先受 orchestration 限制

SlabHash mode-3 trace：

| 类别 | 次数 | GPU/API 时间 |
|---|---:|---:|
| Mixed update kernel | 40 | 1.866 ms |
| Bucket-count kernel | 20 | 0.798 ms |
| CUDA memset | 730 | 4.303 ms |
| D2H copy | 80 | 5.499 ms |
| H2D copy | 40 | 1.031 ms |
| `cudaMalloc` API | 110 | 46.834 ms |
| `cudaFree` API | 110 | 20.280 ms |

被测进程还在 host `poll` 中累计约 20.4 s，主要关联 benchmark 的随机 batch 生成、逐样本 allocation/initialization 和 profiler 协调。

第一个系统结论是：

> 在优化某个 update kernel 之前，必须先复用 allocation、将 metadata 和 batch 生成迁到 device，并减少每批次的 memset、copy 和 synchronization。

### 1.5 Nsight Compute：SlabHash 是随机访存延迟受限

单个 262,144-operation mixed-update kernel：

| 指标 | 结果 |
|---|---:|
| Duration | 42.94 us |
| Compute throughput | 42.96% |
| DRAM throughput | 18.61%，748.23 GB/s |
| L1 hit rate | 0.11% |
| L2 hit rate | 15.52% |
| Achieved occupancy | 78.76% |
| Branch efficiency | 95.15% |
| No eligible warp | 53.05% |
| Long-scoreboard stall | 17.65 cycles/issue |

该 kernel 既没有打满 HBM，也不缺理论 occupancy，branch divergence 也不是主要问题。核心问题是低 cache hit 下的 dependent irregular access：warp 在等待前一次随机 load 返回，无法产生足够的 memory-level parallelism。

这说明 CPU 上的 pointer-based delta chain、B-tree、skip list 和逐节点版本链不能直接照搬到 GPU；GPU delta 必须尽量使用连续 block 和 integer offset，并由 warp cooperative scan。

### 1.6 Nsight Compute：当前 TMA 路径没有形成流水

Density 0.75，working set 16 MiB。NCU duration 只用于 counter 对比，不替代稳态 sweep。

| Segment | 路径 | NCU duration | DRAM peak | Active warps | Long scoreboard |
|---:|---|---:|---:|---:|---:|
| 8 KiB | `cp.async` | 11.14 us | 38.52% | 87.56% | 7.63 |
| 8 KiB | TMA | 13.15 us | 33.01% | 72.64% | 15.68 |
| 16 KiB | `cp.async` | 12.67 us | 33.98% | 64.06% | 5.37 |
| 16 KiB | TMA | 13.79 us | 31.42% | 49.61% | 14.91 |
| 32 KiB | `cp.async` | 13.89 us | 30.59% | 33.51% | 4.18 |
| 32 KiB | TMA | 17.79 us | 24.20% | 24.14% | 11.90 |

其他关键计数器：

- TMA pipe utilization 只有 0.18%-0.55%。
- 双缓冲 shared memory 随 segment 从 8/16/32 KiB 增长为 16/32/64 KiB。
- shared-memory block limit 相应从 9 降到 6，再降到 3。
- 当前 kernel 是串行 phase：bulk load，等待，重排，bulk store，再等待。

第二个 kernel 结论是：

> TMA 不是更快的 memcpy 指令。只有当 producer 发起传输后，consumer 有独立工作可做，并通过固定 tile 多 stage 重叠 load/compute/store 时，TMA 才可能产生收益。

### 1.7 Profiling-guided optimization：fixed tile 与双 stage pipeline

根据 1.6 的 shared-memory residency 和 barrier 结论，第一轮优化保留所有旧路径，并新增两个独立 ablation：

- `tma_tiled`：固定 16 KiB output tile，shared memory 固定为 32 KiB；按 output tile 计算所需的连续 live-prefix input range。
- `tma_pipeline`：两个 16 KiB stage；thread 0 专门发起 TMA load/store，其余 7 个 warps 负责 clear、gap placement 和完成通知，使 tile `n+1` 的 load、tile `n` 的 compute 与 tile `n-1` 的 store 重叠。

完整严格 sweep 使用原配置：64 MiB working set、5 次进程级重复、每点 5 次 warmup 和 20 次 measured iteration。加入两个新 mode 后共有 750 条记录，correctness 全部通过。

下面给出三个 density 的中位数范围：

| Segment | `cp.async` | Single-stage tiled | Two-stage pipeline | Pipeline vs `cp.async` |
|---:|---:|---:|---:|---:|
| 16 KiB | 45.17-45.96 us | 41.99-45.23 us | 55.07-65.26 us | 0.70-0.82x |
| 32 KiB | 44.11-47.00 us | 42.58-46.35 us | 42.09-49.59 us | 0.95-1.05x |
| 64 KiB | 67.36-79.22 us | 45.41-50.74 us | 41.94-48.77 us | 1.61-1.62x |
| 96 KiB | 66.60-78.53 us | 49.04-53.35 us | 40.49-46.65 us | 1.65-1.68x |

结论不是“pipeline 对所有 segment 都更快”。双 stage 使用 64 KiB shared memory，并牺牲一个 producer warp；在 tile 数不足的小 segment 上无法摊薄启动、barrier 和 specialization 成本。测得的初步 adaptive policy 是：

```text
< 16 KiB:        cp.async
16-32 KiB:      single-stage tma_tiled
>= 64 KiB:      two-stage tma_pipeline
```

64/96 KiB、density 0.75 的 NCU counter 进一步验证了优化机制：

| Segment | 路径 | Duration | Dynamic SMEM | SMEM block limit | Active warps | Long scoreboard | Barrier stall | DRAM peak |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 64 KiB | 原整段 TMA | 27.62 us | 131 KiB | 1 | 8.80% | 5.24 | 0.09 | 15.72% |
| 64 KiB | tiled | 17.18 us | 32 KiB | 6 | 39.58% | 7.35 | 4.95 | 18.86% |
| 64 KiB | pipeline | 16.03 us | 64 KiB | 3 | 31.31% | 3.52 | 1.08 | 20.06% |
| 96 KiB | 原整段 TMA | 28.61 us | 197 KiB | 1 | 8.75% | 4.97 | 0.11 | 15.06% |
| 96 KiB | tiled | 20.54 us | 32 KiB | 6 | 27.01% | 6.09 | 3.79 | 15.49% |
| 96 KiB | pipeline | 14.05 us | 64 KiB | 3 | 25.58% | 2.72 | 0.97 | 22.87% |

第一步 fixed tile 主要解决 residency；第二步 pipeline 虽将 SMEM 从 32 KiB 提回 64 KiB，却通过 overlap 将 96 KiB 的 long-scoreboard 从 6.09 降到 2.72、barrier stall 从 3.79 降到 0.97。说明 active warps 并非唯一目标，足够的异步工作重叠可以在较低 occupancy 下更快。

流量口径需要明确：原 scalar/`cp.async`/整段 TMA 会读取并写回完整 segment；新 tiled/pipeline 利用本 microbenchmark 已知的 compact live prefix，只读取 `live_count` 个 input entry，但仍写回完整 output segment。因此 CSV 的 `effective_gbps` 表示完成一个逻辑 segment redistribution 的吞吐，不是实际 HBM byte throughput。延迟比较对这个特化 primitive 有效，但集成真实 PMA 时必须加入任意 gap scan、live-entry compaction 和 update planner，不能把该结果直接宣称为端到端 GPMA speedup。

优化后的下一瓶颈与行动项：

1. 根据 density 缩小每个 stage 的 input buffer，避免固定保留完整 16 KiB input tile。
2. 用 persistent CTA work queue 摊薄每 segment block 的启动和尾部 wave。
3. 将 batch planner、gap scan 和 live compaction 接入同一流水，验证非 compact input。
4. 对混合 segment size 实现 device-side adaptive dispatcher。
5. 集成真实 insert/delete batch 后重新运行 Nsight Systems，确认 allocation/memset/copy 不再主导。

## 2. 从 Profiling 推导出的研究问题

第一轮实测把问题收敛到四个层次：

1. 如何将随机 edge update 转换成合并后的连续 segment 工作。
2. 如何避免 hash/delta 路径中的 dependent pointer chasing。
3. 如何让 TMA 与重排计算真正重叠，而不是增加 barrier 成本。
4. 如何让 query 在 update/compaction 期间看到一致快照，而不全局停机。

CPU 动态图研究在这四个问题上已有大量可迁移思想。需要迁移的是 batch、layout、versioning 和 affected-region 机制，而不是其指针丰富的数据结构实现。

## 3. 可从 CPU 迁移的优化方式

### 3.1 Batch sort、dedup 与 conflict elimination

PPCSR 和 batch-parallel dynamic graph algorithm 的共同模式是先对 batch 做规范化，再并行处理独立区域。PPCSR 的重要观察是，多数更新只修改小区域，而偶尔的大范围 rewrite 高度并行且 cache-efficient。

Hopper 映射：

```text
edge update batch
        ↓
radix sort by (src, dst, timestamp)
        ↓
dedup + cancel insert/delete pairs
        ↓
map update to vertex / segment / PMA level
        ↓
merge overlapping rebalance intervals
        ↓
color or partition into conflict-free waves
        ↓
warp / CTA / cluster execution
```

预期收益：

- 用一次 planner 消除大量 segment lock 和失败 CAS。
- 同一 segment 的更新被合并，只做一次 density check 和 rebalance。
- 提前计算 write set，可以一次 prefix-sum 分配目标空间。
- update batch 越大，规划成本越容易摊薄。

### 3.2 Degree-aware hybrid layout

Terrace 根据真实图的 degree skew，对低度和高度顶点使用不同表示。该思想比把所有 adjacency list 塞进统一 PMA 更适合 power-law graph。

建议的 Hopper 路径：

| 顶点/segment 类型 | 表示 | 执行单元 |
|---|---|---|
| 很低度 | vertex header 内 inline edges | 单 warp |
| 低度、频繁点查 | 小型连续 delta block | 单 warp |
| 中度 | PMA segment | warp 或 CTA + `cp.async` |
| 高度 hub | tiled PMA/base+delta | persistent CTA + TMA pipeline |
| 超大 rebalance | 多 tile region | 可选 thread-block cluster |

路由阈值不应只依赖 degree，而应由以下运行时特征共同决定：

```text
route = f(segment_bytes, density, batch_updates,
          hotspot_score, query_frequency, delta_size)
```

### 3.3 Base + delta + background compaction

GraphOne 使用 edge list 与 adjacency list 的互补表示以及 dual versioning；LiveGraph 使用连续 Transactional Edge Log 保持 adjacency scan 的顺序性。

建议的 GPU 结构：

```text
Base: 只读 CSR 或较稳定 PMA
Delta: 每个 vertex/partition 的连续 append block
Index: integer offset，不使用裸指针
Compactor: 后台固定 tile TMA merge
Publication: segment-level epoch/version CAS
```

查询以 warp 为单位 merge base 与小 delta；delta 超阈值后进入后台 compaction。TMA 只处理足够大的连续 compaction，不参与单条随机 insert。

这直接回应两个 profiling 现象：

- 避免 SlabHash 式随机依赖链。
- 给 TMA 创造足够大的、可流水的连续工作。

### 3.4 Hotspot-adaptive delta partition

GTX 使用 delta-chain-level concurrency，并对 temporal locality 和热点邻接表提高并发度。GPU 不能直接移植 pointer-based delta chain，但可以迁移热点自适应分片。

```text
cold vertex: 1 delta partition
warm vertex: 4 delta partitions
hot hub:    8/16 delta partitions

partition = hash(dst) mod P
```

同一 warp 先 ballot 更新数量，再用一次 warp-aggregated `atomicAdd` 预留连续空间。这样将每 edge CAS 转换成每 warp 或每 partition reservation。

compaction 时再将多个 partition merge 成一个有序或半有序 adjacency segment。

### 3.5 Safe/unsafe update 与 affected frontier

RisGraph 通过 localized access 和 safe/unsafe update 分类获得跨更新并行。这个思想适合迁移到动态 reachability、SSSP 和 k-core。

执行模型：

```text
safe update
    → 只更新存储或局部 certificate
    → fast queue

unsafe update
    → 可能改变算法状态或引发 cascade
    → affected frontier
    → GPU frontier expansion
```

例子：

- SSSP 插入边 `(u,v,w)`，若 `dist[u] + w >= dist[v]`，通常无需传播。
- k-core 更新若不会跨越当前 core boundary，可直接提交。
- reachability 更新若不触及当前 dependency edge，可避免全图遍历。

动态算法部分的潜在突破不是更快的 BFS，而是减少进入 BFS/frontier 的更新数量。

### 3.6 Segment-level MVCC 与 epoch publication

LiveGraph、Teseo 和 GTX 都用 snapshot/version 机制解耦更新和读取。GPU 上应采用 segment/block 粒度，而不是每 edge 放置复杂事务元数据。

```cpp
struct SegmentHeader {
    uint64_t visible_epoch;
    uint64_t replacement_offset;
    uint32_t state;
    uint32_t delta_count;
};
```

更新与发布：

```text
privately build replacement segment
        ↓
system/device fence
        ↓
CAS publish offset + epoch
        ↓
retire old segment
        ↓
epoch-based batch reclamation
```

查询固定一个 read epoch，因此 compaction 不需要暂停全图查询。

### 3.7 Batch dynamic connectivity：迁移算法，重写结构

CPU batch-parallel Euler Tour Tree 和 HDT dynamic connectivity 证明 join/split、replacement-edge search 和 connectivity query 可以批量并行。

不应直接迁移：

- pointer-based balanced tree。
- skip-list node traversal。
- 单更新递归 level propagation。

可迁移为：

- Euler-tour flat chunk/PMA sequence。
- pointer 改成 integer offset。
- join/split 改成 sorted splice descriptor batch。
- HDT level update 改成 level frontier。
- replacement-edge search 按 component/level 分桶。

这是高风险、高研究价值方向，适合作为 HydraPMA 数据结构稳定后的算法验证，而不是第一阶段实现。

## 4. 建议的 HydraPMA 架构

```text
                       ┌──────────────────────┐
edge stream ─────────→│ Batch Normalizer     │
                       │ sort/dedup/cancel    │
                       └──────────┬───────────┘
                                  ↓
                       ┌──────────────────────┐
                       │ Conflict Planner     │
                       │ segment/level waves  │
                       └──────────┬───────────┘
                                  ↓
                  ┌───────────────┼────────────────┐
                  ↓               ↓                ↓
             inline path      delta/PMA path   large-region path
             one warp         cp.async CTA     TMA pipeline
                  └───────────────┼────────────────┘
                                  ↓
                       ┌──────────────────────┐
                       │ Epoch Publication    │
                       └──────────┬───────────┘
                                  ↓
                      query snapshot + affected frontier
```

整体可概括为：

```text
HydraPMA = stable CSR/PMA base
         + degree-aware contiguous delta overlay
         + batch conflict planner
         + warp-specialized tile compactor
         + segment epoch snapshot
         + algorithm-specific affected frontier
```

## 5. Hopper warp specialization 设计

### 5.1 Persistent CTA

一个 CTA 持续从 device work queue 中获取 rebalance descriptor，避免为每个 segment 单独 launch。

建议角色：

- Producer warp：读取 descriptor，发起下一 tile 的 TMA load。
- Consumer warps：对当前 tile 做 live-entry filter、rank、gap placement 和 merge。
- Publisher：发起 TMA store，更新 barrier 和 segment metadata；可由 producer warp 中的 elected thread 承担。

### 5.2 固定 tile，而不是 segment-sized shared memory

候选 tile size：8 KiB 或 16 KiB。无论 PMA segment 是 16 KiB、64 KiB 还是 1 MiB，shared-memory footprint 都保持为 2-3 个 tile。

```text
stage 0: load tile n+1
stage 1: compact tile n
stage 2: store tile n-1
```

这应避免当前 32 KiB segment 使用 64 KiB shared memory、active warps 降至 24.14% 的问题。

### 5.3 `mbarrier` 状态机

每个 stage 至少维护：

- empty/ready barrier。
- expected transaction byte count。
- tile epoch/sequence number。
- consumer completion count。

关键不是减少一次 `__syncthreads()`，而是确保 producer 在 consumer 工作期间继续为下一个 stage 发起数据搬运。

### 5.4 Thread-block cluster 的使用边界

Cluster/DSM 只用于一个 rebalance region 明显大于单 CTA 可高效处理的范围。它不应成为默认路径，因为 cluster residency 会限制调度和 occupancy。

适合场景：

- 高度 hub 的大 adjacency compaction。
- 多 segment 合并成一个新 region。
- 同一 GPC 内多个 CTA 共享 planner metadata 或 tile boundary。

## 6. 研究假设

### H1：Batch planner 能替代 segment locking

按 segment/level 预规划 conflict-free wave，可以减少失败 CAS、重复 density check 和重叠 rebalance，总 update throughput 随 batch size 增长。

### H2：Degree-aware hybrid 优于统一 PMA

inline + delta + PMA/TMA 的混合布局在 power-law 和 hotspot workload 下，同时改善低度顶点更新延迟与高度顶点 scan bandwidth。

### H3：固定 tile 的 warp-specialized TMA 才能跨过 `cp.async`

当前 phase-serial TMA 不构成有效 baseline。只有 load/compute/store overlap 后，TMA 才可能在 32 KiB 以上 region 获得稳定收益。

### H4：Base+delta 能减少 foreground rebalance

小更新先 append delta，把 PMA rewrite 推迟并合并，可降低每条 update 的写放大；代价是查询需要 merge，存在可测的 delta-size crossover。

### H5：Affected frontier 比全量重算更重要

对 k-core、SSSP、reachability，safe/unsafe 分类和 dependency tracking 降低实际传播规模，带来的收益可能大于单次 frontier kernel 优化。

## 7. 初步实验矩阵

### 7.1 数据与更新流

- RMAT/Kronecker，控制 degree skew。
- Uniform 与 Zipf source-vertex update。
- Timestamp-ordered hotspot stream。
- Sliding-window insert/expire。
- Insert/delete ratio：100/0、90/10、50/50。
- Batch size：1K、10K、100K、1M。

### 7.2 Baseline

- SlabHash native SM90。
- 当前 scalar/`cp.async`/TMA segment primitive。
- Rebuild-to-CSR：CUB radix sort + scan。
- 原版 GPMA/LPMA：只记录 H20 compatibility result。
- 后续 host-orchestrated GPMA port：明确标注为 port。

### 7.3 Ablation

1. 无 planner vs sort/dedup vs conflict-free wave。
2. 统一 PMA vs degree-aware hybrid。
3. foreground rebalance vs base+delta。
4. segment-sized shared memory vs fixed tile。
5. phase-serial TMA vs 2-stage vs 3-stage pipeline。
6. per-update atomic vs warp-aggregated reservation。
7. stop-the-world publication vs segment epoch。
8. full recompute vs affected frontier。

### 7.4 指标

- Update throughput 与 P50/P95/P99 batch latency。
- Query throughput、scan bandwidth 和 point lookup latency。
- Rebalance/compaction bytes per update。
- Delta amplification 与 query merge cost。
- DRAM/L2 throughput、cache hit rate。
- Long/short scoreboard、barrier stall、eligible warps。
- TMA pipe utilization 和 overlap ratio。
- Achieved occupancy、shared-memory residency。
- Allocation、memset、copy 和 launch API 时间。
- Correctness：snapshot consistency、edge visibility、算法结果。

## 8. 实施顺序

### Milestone A：去掉系统噪声

- 复用 GPU allocation。
- device-resident batch metadata。
- 减少 memset 和 D2H validation。
- 为 update、planner、rebalance、publication 加 NVTX range。

### Milestone B：Batch planner

- radix sort/dedup。
- segment mapping。
- interval merge。
- conflict-free wave。

### Milestone C：Base+delta 与 degree-aware routing

- contiguous delta block。
- warp-aggregated append。
- low/medium/high-degree route。
- query-side base/delta merge。

### Milestone D：真正的 warp-specialized TMA compactor

- 固定 tile。
- persistent CTA。
- 2/3-stage `mbarrier` pipeline。
- `cp.async`/TMA adaptive dispatcher。

### Milestone E：并发查询与动态图算法

- segment epoch publication。
- k-core 或 reachability affected frontier。
- sliding-window expiration。
- 最后评估 batch dynamic connectivity。

## 9. 可能的论文主线

最有说服力的主线不是单独提出一个更快的 PMA copy kernel，而是：

> 将 CPU 动态图中的 batch conflict elimination、degree-skew adaptive layout、delta/versioning 和 dependency-aware incremental processing，统一映射到 Hopper 的 persistent execution、warp specialization、TMA pipeline 与 segment-level snapshot publication。

对应贡献可以组织为：

1. Hopper-native dynamic graph storage architecture。
2. Conflict-free batch planner 与 degree-aware update routing。
3. Fixed-tile warp-specialized PMA compactor。
4. Query/update concurrent epoch protocol。
5. 从 storage maintenance 到 incremental analytics 的端到端验证。

## 参考资料

- [PPCSR: A Parallel Packed Memory Array to Store Dynamic Graphs](https://epubs.siam.org/doi/10.1137/1.9781611976472.3)
- [Terrace: A Hierarchical Graph Container for Skewed Dynamic Graphs](https://people.eecs.berkeley.edu/~aydin/terrace.pdf)
- [GraphOne: A Data Store for Real-time Analytics on Evolving Graphs](https://www.usenix.org/conference/fast19/presentation/kumar)
- [LiveGraph: Transactional Graph Storage with Sequential Adjacency Scans](https://www.vldb.org/pvldb/vol13/p1020-zhu.pdf)
- [Teseo and the Analysis of Structural Dynamic Graphs](https://vldb.org/pvldb/vol14/p1053-leo.pdf)
- [GTX: A Write-Optimized Latch-free Graph Data System](https://arxiv.org/abs/2405.01418)
- [RisGraph: Real-Time Streaming Analysis on Evolving Graphs](https://arxiv.org/abs/2004.00803)
- [Batch-Parallel Euler Tour Trees](https://arxiv.org/abs/1810.10738)
- [Parallel Batch-Dynamic Graph Connectivity](https://arxiv.org/abs/1903.08794)
- [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/archive/12.9.0/hopper-tuning-guide/contents.html)
