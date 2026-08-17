# HydraPMA：已有想法、技术细节与当前进展

> 更新时间：2026-08-17  
> 当前代码基线：`f2767bb Add arbitrary-gap Hopper PMA compaction`

## 1. 我们到底在解决什么问题

HydraPMA 当前瞄准的不是某一个孤立的图算法，而是一个更底层的问题：

> 在 Hopper GPU 常驻内存中，如何高吞吐地处理批量 edge insert/delete，并把发生变化的 adjacency segment 维护成适合 GPU 连续扫描的有序、带 gap 布局。

输入是一批动态图更新：

```text
(src, dst, op, timestamp)
```

底层存储是 PMA 风格 segment：

```text
| e | e | _ | e | _ | e | _ | _ |
```

维护路径大致是：

```text
update batch
    -> sort / dedup / cancel
    -> locate vertex and PMA segment
    -> identify density violation
    -> merge overlapping rebalance ranges
    -> scan live entries
    -> stable compact and redistribute gaps
    -> publish the replacement segment
```

当前已经真正实现并在 H20 上验证的是其中的“scan live entries + stable compact + redistribute gaps”部分。batch planner、真实 insert/delete merge、PMA level propagation 和 snapshot publication 仍是下一阶段。

因此，当前可以准确声称：

- 已有 Hopper-native PMA segment redistribution microbenchmark。
- 已覆盖 compact live prefix 和任意 gap 两类输入。
- 已验证 fixed tile、TMA producer/consumer pipeline、stable scan 和 buffered scatter。
- 已形成基于 segment size、density 的初步 dispatcher 证据。

当前还不能声称：

- 已经完成端到端 GPMA/LPMA 替代。
- 已经实现完整动态图数据库或 MVCC。
- 已经证明 PageRank、k-core、connectivity 等算法获得端到端加速。

## 2. 为什么传统 CPU 动态图结构不能直接搬到 Hopper

CPU 动态图常依赖：

- linked adjacency list；
- tree/skip-list node；
- pointer-based delta chain；
- fine-grained lock；
- 单更新递归传播；
- cache-coherent random access。

这些机制在 GPU 上会放大为：

- pointer chasing 和 dependent memory latency；
- warp divergence；
- 随机 global memory transaction；
- 每 edge atomic/CAS；
- 低 eligible-warps；
- 无法形成足够长的 bulk-synchronous 工作。

Hopper 的优势则集中在另一组能力：

- 高带宽 HBM；
- 大量并行 thread blocks；
- TMA bulk asynchronous copy；
- `mbarrier`；
- warp specialization；
- shared-memory staging；
- device-resident work queue；
- 对大 batch 的排序、scan 和 frontier 扩展。

所以这里的迁移原则不是“复刻 CPU 数据结构”，而是：

> 保留 CPU 算法中的 batch、locality、versioning、affected-region 思想，同时把 pointer-rich structure 重写成 flat array、integer offset、descriptor batch 和 GPU frontier。

## 3. 已有 idea 地图

| Idea | CPU 侧来源/动机 | Hopper 映射 | 当前状态 |
|---|---|---|---|
| Batch sort/dedup/cancel | PPCSR、batch dynamic graph | radix sort + segmented scan | 待实现 |
| Conflict-free update wave | 避免锁竞争和重复 rebalance | interval merge + descriptor wave | 待实现 |
| Degree-aware hybrid layout | Terrace 等 skew-aware container | inline / delta / PMA 分流 | 设计阶段 |
| Base + delta | GraphOne、LiveGraph 一类思路 | stable base + contiguous delta block | 设计阶段 |
| Fixed-tile compaction | 避免 segment-sized working set | 16 KiB tile + fixed SMEM | 已实现 |
| Warp-specialized TMA | overlap memory and compute | producer warp + 7 consumer warps | 已实现 |
| Arbitrary-gap stable scan | 真实 PMA gap discovery | ballot/scan + rank placement | 已实现 |
| Buffered rank-order scatter | 同时降低 barrier 和保持写合并 | shared compact tile | 已实现 |
| Adaptive dispatcher | 不同 degree/segment crossover | size+density runtime route | 初步证据 |
| Segment-level MVCC | LiveGraph/GTX/Teseo 类 versioning | epoch + replacement offset + CAS | 待实现 |
| Affected frontier | RisGraph、incremental algorithms | safe/unsafe queue + GPU frontier | 算法阶段 |

### 3.1 Baseline 不是一个数字，而是四种设计路线

我们的 baseline 应覆盖不同的数据结构哲学。否则只和某一个旧 GPMA artifact 比较，无法判断收益究竟来自 Hopper 指令、batch 方式，还是换了数据结构。

#### GPMA：标准 density-driven PMA 路线

固定 artifact：`third_party/LPMA` 中的 `gpma.cuh`。

GPMA 的核心不是“用一个大数组代替 adjacency list”这么简单，而是一棵隐式 PMA 层次：

```text
leaf segment
    -> level 1 interval
    -> level 2 interval
    -> ...
    -> root
```

每一层根据 interval width 计算 lower/upper density threshold。batch update 先定位 leaf：

```text
locate_leaf_batch
    -> compact affected leaf/interval
    -> merge updates
    -> test density
    -> evenly redispatch
    -> propagate violating nodes upward
```

artifact 源码中可以看到：

- root/leaf density threshold 插值；
- CUB block scan compaction；
- block radix sort；
- 每个 tree node 由一个 CUDA block 处理；
- `rebalance_width = segment_length << level`；
- `up_level_batch`；
- root resize。

它代表的 baseline 问题是：

> 在不改变经典 GPMA update/rebalance 语义的前提下，Hopper compactor 能给 level rebalance 带来多少收益？

GPMA 也是后续 integration baseline：应先保留它的 leaf mapping、density policy 和 level propagation，只替换 compact/redispatch primitive，才能分离 kernel 优化收益。

当前 H20 状态不是性能失败，而是兼容性失败。旧 artifact 使用 legacy CDP1 device-side `cudaDeviceSynchronize()`，在 SM90/CUDA 12.9 上无法加载。不能通过删除同步来声称“原版 baseline 已运行”，因为这会改变递归/层级执行的正确性。

合理的后续处理有两个：

1. 保留原版 compatibility result；
2. 另做 host-orchestrated 或 device-queue port，并明确标为 port，不冒充 unmodified GPMA。

#### LPMA artifact：level-aware PMA 组织路线

固定 artifact commit：`eb22cd4e1515e83ace93868a2f1e9f2b3b6a53ea`。

该仓库名为 LPMA，主要实现还包括 `rpma.cuh`。从 artifact 代码可以确认，它仍保留：

- density hierarchy；
- batch leaf location；
- per-level rebalance；
- compact/merge/redispatch；
- level key/value pointer arrays；
- mixed update/query 路径。

它的 baseline 价值不是提供另一条 memcpy，而是回答：

> 通过 level-aware storage organization 减少或重组 PMA 数据移动，和单纯加速 monolithic rebalance 相比，哪一种收益更大？

HydraPMA 应吸收它的 level-aware 思路，但避免直接延续旧式 device-side orchestration。后续比较应拆成：

- 原 artifact compatibility；
- 现代化 orchestration port；
- port + Hopper compactor。

这样可以分别测出“执行模型升级”和“TMA/warp specialization”的贡献。

#### SlabHash：warp-oriented 随机更新路线

固定 artifact commit：`eec7135a3e5d40ba7fb9a7e88f7bd6d49abc4dee`。

SlabHash 是 warp-oriented dynamic hash table：

- key hash 到 bucket；
- bucket 对应一个或多个固定尺寸 slab；
- 一个 warp 协作搜索/插入/删除 slab；
- SlabAlloc 提供 warp-centric fixed-size allocation；
- mixed benchmark 支持 insert/delete/existing search/missing search。

它代表和 PMA 几乎相反的取舍：

| 维度 | SlabHash | PMA/HydraPMA |
|---|---|---|
| 单点 update | 强 | 需要 batch 摊薄 |
| point lookup | 强 | 依赖 segment search |
| 有序 adjacency scan | 弱 | 强 |
| pointer/dependent access | 多 | 目标是连续 offset |
| 空间组织 | bucket + slabs | ordered gapped array |
| 大范围 analytics | 需要收集/遍历 slabs | 可连续扫描 |

所以 SlabHash 不是“另一个 PMA baseline”，而是随机更新/lookup 路线的对照组。当前 H20 NCU 已显示其 mixed update 不是 HBM 带宽饱和，而是 dependent random load latency：

- DRAM peak 18.61%；
- L2 hit 15.52%；
- no eligible warp 53.05%；
- long scoreboard 17.65。

这组结果支持一个 hybrid idea：低度或高频 point-update vertex 使用小 delta/hash partition，高度或 scan-heavy vertex 使用 PMA base。

#### Rebuild-to-CSR：bulk synchronous 路线

这条 baseline 目前尚未接入代码，但必须加入。

它不维护复杂动态图结构，而是每个 batch：

```text
old edges + updates
    -> radix sort
    -> dedup/cancel delete
    -> prefix sum row offsets
    -> rebuild CSR
```

优势：

- 完全连续；
- GPU sort/scan 成熟；
- rebuild 后查询和 analytics 最友好；
- batch 足够大时可能胜过细粒度 dynamic maintenance。

劣势：

- 小 batch 写放大严重；
- 每批可能移动全图；
- publication 需要双 buffer 或 epoch；
- latency 对 graph size 敏感。

它回答的是最重要的 crossover 问题：

> batch 多大、affected region 多大时，局部 PMA maintenance 仍然优于直接 rebuild？

如果 HydraPMA 在大 batch 上输给 rebuild-to-CSR，这不代表研究失败，而是 dispatcher 应把该区域路由到 bulk rebuild。

#### Segment movement ablation：机制 baseline

仓库内的 `scalar`、`cp_async`、`tma_bulk`、`tma_tiled`、`tma_pipeline` 不是竞争系统，而是硬件机制 ablation：

- `scalar`：普通 cooperative copy 的下界。
- `cp_async`：Ampere/Hopper 通用的 per-thread async staging。
- `tma_bulk`：phase-serial TMA，验证“只换指令”是否有效。
- `tma_tiled`：隔离 fixed footprint/residency 收益。
- `tma_pipeline`：隔离 producer/consumer overlap 收益。

arbitrary-gap 四条路径继续拆分：

- `gap_scan`：完整 scan 基线；
- raw TMA：加载优化；
- chunked：barrier 优化但写合并退化；
- buffered：barrier 与 coalescing 同时优化。

这些 ablation 的作用是建立因果链，不能替代端到端 GPMA、SlabHash、rebuild-to-CSR comparison。

#### 公平 baseline contract

最终比较至少同时报告两种口径：

1. steady-state GPU data-structure time；
2. 包含 allocation、copy、planner、publication 的 end-to-end batch latency。

所有系统必须统一：

- 相同 initial graph；
- 相同 insert/delete/query stream；
- 相同 batch size 和 update distribution；
- 相同 correctness semantics；
- 相同预分配/扩容约束；
- 相同 GPU 和可用 memory budget；
- 同时报告 update throughput、P50/P95/P99 latency、memory footprint；
- 查询侧同时测 point lookup 和 adjacency scan。

还应单独报告 preprocessing。不能把一个 baseline 的 host random generation、allocation 和 D2H validation 算进核心时间，却把另一个实现只报 kernel time。

## 4. 当前代码中的数据模型

每个 entry 固定为 16 bytes：

```cpp
struct alignas(16) Entry {
    uint64_t key;
    uint64_t value;
};
```

空槽用 `UINT64_MAX` 作为 key sentinel。固定 16-byte entry 有两个目的：

1. 与 `cp.async` 的 16-byte copy 粒度自然匹配。
2. TMA bulk load/store 的 tile size 和 shared-memory footprint 容易控制。

假设 segment capacity 为 `C`，live entry 数为 `L`，第 `i` 个有序 live entry 的目标位置是：

```text
dst(i) = floor(i * C / L)
```

这保证：

- key 顺序不变；
- gap 近似均匀分布；
- output density 与 PMA segment threshold 相匹配。

benchmark 目前支持两种输入 layout：

- `prefix`：前 `L` 个位置为 live entries，其余为空。
- `spread`：live entries 已均匀散布在整个 input segment 中。

`prefix` 用于分析已知 compact input 的搬运/重排开销；`spread` 用于把真实 gap scan 成本纳入测量。

## 5. Compact-prefix 路径的演化

### 5.1 原始三条路径

- `scalar`：cooperative scalar global-to-shared copy。
- `cp_async`：每线程发起 16-byte `cp.async`。
- `tma_bulk`：整段 TMA load，等待，重排，再整段 TMA store。

原始 TMA 没有形成流水，其执行顺序本质上是：

```text
load -> wait -> compute -> store -> wait
```

而且 shared memory 随 segment size 增长为 `2 * segment_bytes`，直接压低 residency。

### 5.2 Fixed tile

`tma_tiled` 把 working set 固定成 16 KiB input tile + 16 KiB output tile，总计 32 KiB dynamic shared memory。

这一步解决的是 residency：

- segment 可以继续增大；
- CTA 的 shared-memory footprint 不再随 segment 增长；
- 一个 SM 可以驻留更多 blocks；
- output tile 仍然可以通过 TMA bulk store。

### 5.3 两阶段 producer/consumer pipeline

`tma_pipeline` 使用两个 stage，每个 stage 包含 16 KiB input 和 16 KiB output，总 dynamic shared memory 为 64 KiB。

角色划分：

```text
thread 0 / producer warp:
    issue TMA load for tile n+1
    issue TMA store for tile n-1
    manage stage reuse

7 consumer warps:
    clear output tile n
    place live entries
    publish compute completion
```

目标是形成：

```text
load(n+1) || compute(n) || store(n-1)
```

stage 复用依赖两个 load `mbarrier`、phase parity 和 `compute_ready[stage]`。producer 只有在 consumer 发布完成后才能覆盖对应 stage。

在 compact-prefix sweep 中，64/96 KiB segment 上 pipeline 相对 `cp.async` 达到约 1.6x，但这条路径利用了“live entries 已知在连续前缀”这一条件，因此不能代表真实任意-gap PMA 的完整成本。

## 6. 任意-gap 路径的技术细节

### 6.1 `gap_scan`：正确且简单的 GPU 基线

一个 CTA 负责一个 segment：

1. cooperative clear 整个 output segment；
2. 每轮读取 256 个 input entries；
3. 每个 warp 用 ballot 得到 valid mask；
4. 8 个 warp count 做 CTA prefix；
5. 每个 live entry 得到稳定的全局 live rank；
6. 按 `dst(rank)` scatter。

这条路径会检查全部 input slot，不依赖 gap 位置。它没有 dynamic shared-memory staging，适合小 segment，也是任意-gap dispatcher 的基础路径。

### 6.2 `gap_tma_pipeline`：TMA load + ballot scan

一个 producer warp 管理两个 16 KiB input stage，7 个 consumer warps 扫描 ready tile。

consumer 每次处理 224 entries：

```text
ballot valid lanes
    -> 7 warp counts
    -> prefix across warps
    -> stable live rank
    -> rank-order global scatter
```

每个 224-entry batch 大致需要三次 named consumer barrier：

1. warp counts ready；
2. warp offsets ready；
3. scatter finished。

一个完整 1024-entry tile 需要约 5 个 batch，也就是约 15 次 consumer barrier。它保持了较好的 rank-order store，但同步频率偏高。

### 6.3 `gap_tma_chunked`：一次重要的负结果

为了减少 barrier，我们让 224 个 consumer threads 各自负责一个连续 input chunk。完整 tile 下每个线程最多扫描约 5 个 entries。

每线程先统计本 chunk 的 live count，然后：

```text
thread local count
    -> warp shfl prefix
    -> 7 warp totals
    -> CTA prefix
    -> thread-owned stable rank range
```

这样每 tile 只需要约三次 barrier。

小工作集上它明显更快，但 64 MiB 严格 sweep 中反而变慢。原因是同一 warp 的相邻线程拥有相隔约 4-5 个 entry 的 chunk；它们同时 scatter 时，目标 rank 不连续，破坏了 global-store coalescing。

NCU 证据：

- barrier stall：5.04 降到 1.83；
- excessive L2 sectors：150,784 增长到 357,632；
- 小工作集被 L2 掩盖，大工作集暴露真实写放大。

这个结果说明：

> GPU scan 优化不能只看同步次数；rank 分配方式还必须显式维护 warp-level output coalescing。

### 6.4 `gap_tma_buffered`：低同步 + 合并写

buffered 路径保留 chunk scan，但不直接向 global output scatter。

流程变为：

```text
TMA load input tile
    -> contiguous per-thread chunk scan
    -> stable prefix
    -> write live entries to shared compact[local_rank]
    -> consumer barrier
    -> consecutive lanes read compact[]
    -> rank-order global scatter
```

shared-memory 布局：

```text
stage 0 input: 16 KiB
stage 1 input: 16 KiB
compact tile: 16 KiB
total:        48 KiB
```

代价：

- 多一次 shared write/read；
- 多一个 consumer barrier；
- shared-memory block limit 低于 32 KiB 版本。

收益：

- barrier 仍远少于 raw ballot pipeline；
- global scatter 恢复为连续 rank；
- excessive sectors 接近 direct scan/raw pipeline；
- HBM 工作集下获得稳定收益。

## 7. H20 实测更新

### 7.1 Baseline 兼容性

- 原始 GPMA/LPMA 使用 legacy CDP1 device-side synchronization，在 H20/CUDA 12.9 上触发不支持错误，不能通过简单删除同步来伪造可运行 baseline。
- SlabHash 只做 SM90 build compatibility 修改后可以运行。
- SlabHash mixed-update NCU 表明主要瓶颈是 dependent random memory latency，而不是 DRAM peak bandwidth。
- 初始 PMA phase-serial TMA pipe utilization 很低，同时 segment-sized shared memory 限制 residency。

### 7.2 任意-gap 严格 sweep

最终配置：

- 64 MiB working set；
- 5 次独立进程重复；
- 每点 5 次 warmup；
- 20 次 measured iteration；
- 6 个 segment size；
- 3 个 density；
- `prefix` / `spread` 两种 layout；
- 4 个 arbitrary-gap mode。

共 720 条记录，全部 correctness 通过。

`spread`、density 0.7 的中位数：

| Segment | Direct scan | Raw TMA | Chunked | Buffered | Buffered vs scan | Buffered vs raw TMA |
|---:|---:|---:|---:|---:|---:|---:|
| 4 KiB | 70.54 us | 103.19 us | 123.83 us | 138.34 us | 0.51x | 0.75x |
| 8 KiB | 66.27 us | 75.22 us | 80.98 us | 91.50 us | 0.72x | 0.82x |
| 16 KiB | 66.39 us | 63.00 us | 68.89 us | 71.17 us | 0.93x | 0.89x |
| 32 KiB | 71.40 us | 63.02 us | 70.07 us | 65.16 us | 1.10x | 0.97x |
| 64 KiB | 93.93 us | 84.58 us | 96.81 us | 69.59 us | 1.35x | 1.22x |
| 96 KiB | 101.40 us | 94.45 us | 127.73 us | 86.29 us | 1.18x | 1.09x |

这里最可信的结论是：

- 4/8 KiB：TMA specialization 固定成本无法摊薄，direct scan 最好。
- 16/32 KiB：raw TMA pipeline 是更稳妥的选择。
- 64 KiB：buffered 路径稳定胜出。
- 96 KiB：buffered 在低/中 density 胜出；90% density 下 raw pipeline 略好。

这还是候选 policy，需要补测 48/80 KiB、更多 density 和混合 segment wave。

### 7.3 64 KiB NCU 对照

条件：density 0.7、`spread`、16 MiB profiling working set。

| Path | Duration | SM peak | DRAM peak | Long scoreboard | Barrier | Excessive sectors | Dynamic SMEM |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct scan | 28.06 us | 25.36% | 15.91% | 9.17 | 5.76 | 150,784 | 0 KiB |
| Raw TMA | 26.56 us | 31.85% | 17.32% | 3.50 | 5.04 | 150,784 | 32 KiB |
| Chunked | 25.41 us | 27.34% | 18.05% | 4.42 | 1.83 | 357,632 | 32 KiB |
| Buffered | 23.10 us | 30.82% | 19.96% | 4.17 | 2.23 | 163,328 | 48 KiB |

可以把优化链条概括成：

```text
direct scan
    -> TMA lowers global-load scoreboard
    -> chunk scan lowers barriers but breaks stores
    -> shared compact buffer restores coalescing
```

## 8. 当前最重要的技术认识

### 8.1 TMA 不是“更快的 memcpy”

如果 load、compute、store 仍按 phase 串行，TMA 只会增加 barrier 和控制成本。收益来自 producer 与 consumer 的独立工作以及 stage overlap。

### 8.2 Fixed footprint 比盲目追求 occupancy 更重要

segment-sized shared memory 会让大 rebalance 只能驻留一个 CTA。固定 tile 首先解决可扩展性，之后再权衡 stage 数、register 和 occupancy。

### 8.3 低 barrier 不等于高性能

`gap_tma_chunked` 是目前最有价值的负 ablation：同步显著改善，但写合并恶化导致大工作集回退。以后每个 prefix/scan 改动都必须同时检查：

- barrier stall；
- long/short scoreboard；
- eligible warps；
- L2 theoretical excessive sectors；
- 实际 DRAM read/write bytes；
- 小工作集与大工作集的差异。

### 8.4 Microbenchmark 必须写清流量口径

CSV 的 `effective_gbps` 是“完成一次逻辑 segment redistribution”的吞吐，不等于实际 HBM byte throughput。

- compact-prefix tiled/pipeline 只读取 `live_count` 个连续 input entries；
- arbitrary-gap modes 扫描整个 input；
- arbitrary-gap modes 清空整个 output，再 scatter live entries；
- buffered 还包含 shared compact traffic。

所以跨路径比较应优先使用 latency，并结合 NCU 实际 traffic 解释。

## 9. 下一层系统 idea 的技术展开

### 9.1 Batch planner：把锁竞争变成预规划

建议的 update record：

```cpp
enum class EdgeOp : uint8_t { Insert, Delete };

struct EdgeUpdate {
    uint32_t src;
    uint32_t dst;
    uint32_t timestamp;
    EdgeOp op;
};
```

planner pipeline：

```text
radix sort by (src, dst, timestamp)
    -> collapse duplicate inserts/deletes
    -> cancel insert/delete pairs
    -> map src to adjacency segment
    -> calculate post-update live count
    -> find violated PMA level
    -> emit rebalance interval
    -> merge overlapping intervals
    -> bucket descriptors by execution path
```

建议的 work descriptor：

```cpp
struct RebalanceTask {
    uint64_t input_offset;
    uint64_t output_offset;
    uint32_t capacity;
    uint32_t old_live_count;
    uint32_t insert_begin;
    uint32_t insert_count;
    uint32_t delete_begin;
    uint32_t delete_count;
    uint16_t pma_level;
    uint16_t flags;
};
```

planner 的目标不是做复杂预测，而是一次性消除：

- 多个 update 对同一 segment 的重复 density check；
- overlapping rebalance；
- segment lock retry；
- 每 edge atomic reservation；
- host 参与的逐 segment launch。

第一版可以由 CUB radix sort + scan 组成，不需要一开始就实现 persistent kernel。

### 9.2 Device-side dispatcher

当前 profiling 支持如下初步路由：

```text
route = f(segment_bytes, density, live_count, update_count)
```

候选规则：

```text
arbitrary gap:
    < 16 KiB                 -> gap_scan
    16-32 KiB               -> gap_tma_pipeline
    64 KiB                  -> gap_tma_buffered
    96 KiB, density <= 0.7  -> gap_tma_buffered
    96 KiB, high density    -> gap_tma_pipeline
```

真正系统中不应让 host 逐任务选择 kernel。更合理的第一版是 planner 把 task 分桶：

```text
small_queue
medium_queue
buffered_tma_queue
raw_tma_queue
```

每个 queue 一次 launch，之后再评估 persistent CTA 是否值得。

### 9.3 Degree-aware hybrid layout

统一 PMA 对 power-law graph 并不理想：

- 大量低度 vertex 为 density metadata 和 segment 留洞付出过高空间成本；
- 高度 hub 的大 segment 才能真正摊薄 TMA 和 pipeline 固定成本；
- 热点 vertex 需要更高 update concurrency。

建议分层：

```text
very low degree:
    vertex header inline edges

low degree / frequent updates:
    small contiguous delta block

medium degree:
    PMA segment, one warp or one CTA

high-degree hub:
    tiled base + partitioned delta

large rebalance region:
    TMA buffered compactor
```

路由不能只看 degree，还应考虑：

- update frequency；
- query frequency；
- delta size；
- hotspot score；
- segment density；
- batch update count。

### 9.4 Base + delta

直接对 foreground PMA 做每次 rebalance 会产生 write amplification。更实际的结构是：

```text
Base:
    read-mostly CSR/PMA segment

Delta:
    per vertex or partition contiguous append block

Compactor:
    background merge base + delta

Publication:
    segment-level epoch switch
```

写路径先用 warp-aggregated reservation 向 delta append；delta 超过阈值后生成 compaction task。查询端用一个 warp merge base 与小 delta。

需要测量的 crossover：

- delta append latency；
- query-side merge cost；
- foreground PMA rebalance cost；
- compaction bytes per update；
- 不同 update/query ratio 下的最优 delta threshold。

### 9.5 Hotspot-adaptive delta partition

对高频 hub，可以把一个 delta 分成多个连续 partition：

```text
cold: 1 partition
warm: 4 partitions
hot:  8 or 16 partitions
```

```text
partition = hash(dst) mod P
```

同一 warp 先 ballot 本 partition 的 update 数，再用一次 warp-aggregated `atomicAdd` 预留连续空间，避免每 edge atomic。compaction 时再把多个 partition merge 为有序 adjacency。

### 9.6 Segment-level epoch publication

查询与更新并发时，不应在每个 edge 上放复杂 MVCC metadata。更适合 GPU 的粒度是 segment：

```cpp
struct SegmentHeader {
    uint64_t visible_epoch;
    uint64_t active_offset;
    uint64_t replacement_offset;
    uint32_t state;
    uint32_t delta_count;
};
```

发布协议：

```text
build replacement privately
    -> device/system fence
    -> CAS active_offset + visible_epoch
    -> retire old segment
    -> batch epoch reclamation
```

query 固定 read epoch。第一版不需要完整事务，只需验证：

- query 不看到半写 segment；
- insert/delete batch 在 epoch 边界原子可见；
- old segment 在所有旧 epoch reader 离开后回收。

## 10. 上层动态图算法如何接入

数据结构稳定后，优先验证 sliding-window stream：

```text
new edge batch
    -> insert queue

expired timestamp range
    -> delete queue

planner
    -> affected segments
    -> compaction/publication
```

算法侧优先考虑 affected frontier，而不是每批全量重算。

### Dynamic k-core

```text
edge updates
    -> degree/core boundary test
    -> safe updates commit directly
    -> unsafe vertices enter frontier
    -> GPU cascade expansion
```

### Reachability / SSSP

维护 dependency/certificate。只有可能改变结果的 unsafe updates 进入 propagation frontier。

### Dynamic connectivity

这是更高风险方向。CPU Euler Tour Tree/HDT 的思想可以迁移，但 pointer-based tree 必须重写为 flat chunk/PMA sequence、integer offset 和 batched join/split descriptor。它不适合作为当前第一阶段的验收算法。

## 11. 当前更新记录

### `fffd83e`：baseline 与初始 profiling

- 建立 HydraPMA 仓库；
- 拉取 GPMA/LPMA 与 SlabHash baseline；
- 记录 H20 compatibility；
- 建立 scalar/`cp.async`/TMA microbenchmark；
- 完成第一轮 Nsys/NCU。

### `6b45165`：fixed tile 与 compact-prefix pipeline

- 新增 `tma_tiled`；
- 新增双 stage `tma_pipeline`；
- 将 shared-memory footprint 与 segment size 解耦；
- 完成 750-row sweep；
- 更新 profiling-first research 文档。

### `f2767bb`：任意-gap compaction

- 新增 `prefix` / `spread` layout；
- 新增 `gap_scan`；
- 新增 raw TMA、chunked 和 buffered 三个 ablation；
- 完成 720-row strict sweep；
- 完成 32/64 KiB NCU；
- buffered 非整 tile 通过 compute-sanitizer memcheck，0 errors；
- 加入可复现 gap sweep 与 profiling 脚本。

## 12. 接下来最具体的一步

下一阶段不应继续孤立优化 copy kernel，而应完成最小可用的真实更新闭环：

```text
synthetic insert/delete batch
    -> GPU sort/dedup/cancel
    -> segment mapping
    -> one-level rebalance planner
    -> merge old live entries with updates
    -> adaptive gap compactor
    -> validation against CPU reference
```

建议先限制问题：

- 单 GPU；
- graph 常驻 HBM；
- 每个 vertex 暂时只有一个 PMA segment；
- 不做跨 level upward propagation；
- batch 内 update 已有 timestamp；
- 暂不做并发 query；
- 先支持 insert/delete 正确性和吞吐。

第一组验收指标：

1. 1K/10K/100K/1M batch update throughput。
2. Uniform 与 Zipf source distribution。
3. Insert/delete ratio：100/0、90/10、50/50。
4. Segment density：0.5/0.7/0.9。
5. Planner、merge、compaction 各自的 GPU time。
6. bytes moved per committed update。
7. 与 rebuild-to-CSR、SlabHash 以及可运行 baseline 的对照。

完成这一闭环后，当前 segment primitive 才真正成为动态图 storage maintenance，而不是单独的 kernel benchmark。

## 13. 可复现实验入口

构建：

```bash
cmake -S . -B build -DHYDRAPMA_CUDA_ARCH=sm_90a
cmake --build build -j
```

单点任意-gap 对照：

```bash
./build/segment_bench \
  --segment-bytes 65536 \
  --density 0.7 \
  --layout spread \
  --mode gap_all \
  --working-set-mb 64
```

完整 sweep：

```bash
python3 scripts/run_segment_sweep.py \
  --binary ./build/segment_bench \
  --config configs/h20_gap_sweep.json \
  --output results/h20_gap_sweep.csv
```

关键文件：

- `src/segment_bench.cu`：所有 kernel ablation。
- `configs/h20_gap_sweep.json`：任意-gap strict sweep。
- `scripts/run_segment_sweep.py`：原始 CSV 驱动。
- `scripts/profile_h20.sh`：Nsys/NCU profiling。
- `docs/h20_initial_results.md`：最初 profiling 原始记录。
- `docs/hopper_dynamic_graph_research.md`：profiling-first 完整研究路线。

## 14. 一句话总结

当前最有价值的进展不是“我们写了一个更快的 TMA copy”，而是确认了一个可复用的 Hopper 动态图设计规律：

> 先用 fixed tile 和 warp specialization 隐藏全量 gap scan 的访存延迟，再通过 shared rank-order buffer 同时控制同步成本与 scatter 合并；随后用 batch planner 把随机 edge updates 转换为这种 GPU 擅长的连续 segment 工作。
