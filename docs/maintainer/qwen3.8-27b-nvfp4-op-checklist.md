# Qwen3.8-27B NVFP4 Op 支持清单

## 1. 目标与边界

本文是固定 artifact `qwen3_8_27b_nvfp4.ninfer` 的 Text 权重接入仓库内部语义 Op
之前必须完成的 Op 级清单。Artifact 合同由
[`qwen3.8-27b-artifact.md`](qwen3.8-27b-artifact.md) 固定；本文根据其中全部物理 parent、
当前 Qwen3.6 family schedule，以及 `qwen3_6_27b` package 的三个 execution leaf，推导精确的
Op 需求。

本文跟踪的基本单位是一个精确的 format/problem registration：

```text
语义 Op × public form × 权重 format/layout × 精确 geometry × 支持的 extent domain
```

它不是一个 artifact tensor、一个模型层、一个 CUDA kernel 或一次 launch。每个清单项必须说明：

- 它覆盖哪些变化后的 Text parent，以及 parent 的当前语义消费者；
- 完整持久化权重合同和精确 `[N,K]`；
- public 输入、输出、状态和 selector tensor 的 dtype 与逻辑维度；
- fused row order、可观察副作用、alias 规则、extent domain 和 workspace query；
- 独立 oracle、回归测试和 public-Op benchmark 的完成证据；
- 同一 geometry 的工作如何从 generic FP8 Linear 实现、调优和模板化，连续演进到实际
  fused Op/epilogue，以及最终采用什么 route。

本文不重复已经存在的 FP8 container/layout codec 或 converter 工作，也不涵盖 target artifact
binder、mixed weights profile、model view 填充、target leaf dispatch、family workspace 组合、
registry 选择、Engine 加载、CLI、serving 或端到端性能。后续集成任务不能替代本文任一 Op 项的
完成证据。

MTP-private 和 Vision 权重的 format 与 geometry 没有变化，因此不需要新增 Op admission。MTP
仍然 alias Text token embedding 和 full output head，所以这两个变化后的物理 parent 分别由同一
`embedding` 和 `linear` registration 覆盖，不新增 MTP 专用 Op。

本清单从 14 个未完成的完整 registration 开始：6 个 generic FP8 Linear、1 个 embedding，以及
7 个复用 Linear contraction 实现的 fused/专用 projection registration。当前 `[14336,5120]`
的 L2/A1、`[16384,5120]` 的 L3/G1、`[34816,5120]` 的 L4/M1、`[5120,6144]` 的 L5/R1，
以及 `[5120,17408]` 的 L6/R2 纵向工作流均已完成；它们都接纳全部正 `T`、A16/A8 arithmetic
profile 和 caller workspace。2026-08-17 对完整 `T=1..48` 曲线重新逐点审计后，L2 在 `T<=11`
使用 A16、从 `T=12` 使用 A8，A1 则在 `T<=10` 使用 A16、从 `T=11` 使用 A8；L3 在 `T<=10`
使用 A16、从 `T=11` 使用 A8，G1 在 `T<=7` 使用 A16、从 `T=8` 使用 A8；L4 在 `T=1` 使用
A8、`T=2..4` 使用 A16、从 `T=5` 回到 A8，M1 独立采用 `T=1` A8、`T=2` A16、`T>=3` A8。
L5/L6 在 `T<25` 使用 A16、`T>=25` 使用 A8；R1 在 `T<22` 使用 A16、`T>=22` 使用 A8；R2
在 `T<25` 使用 A16、`T>=25` 使用 A8。每个实际 Op 拥有自己的 resolver，并依据自身完整语义
调用确定 route。A1 直接写入
Q/K/gate/V，G1 直接写入 QKV/Z，M1 直接写入最终
SwiGLU 结果，R1/R2 直接完成原地 residual 更新；G2/G3 仍待完成。Artifact 中另外 112 个
NVFP4 parent 已有精确支持；第 12 节单独记录其核对依据，不能只因 format 名称相同便假定已覆盖。

## 2. 固定执行事实

### 2.1 维度记法和执行 extent

矩阵权重使用逻辑维度 `[N,K] = [输出行数,输入列数]`。除显式保留 batch 维度的 GDN 形式外，
public activation 和结果都使用 dimension zero fastest 的连续 `[rows,T]` 存储。本文所有 projection
输入、projection 结果、非线性结果和 residual 都是 `BF16`。`T` 为正数；decode、verification 和
prefill 是实现与调用场景，不是不同的数学 Op。

当前 target 的重要生产 extent 为：

- 普通 compact decode：聚合列数 `T=B`，其中 `1 <= B <= 8`；
- MTP target verification：每个 sequence 的宽度 `2 <= W <= 6`，batch `1 <= B <= 8`，
  rank-two Op 使用展平后的 `T=B*W`；
- Text prefill：正的 chunk extent，`T=1024` 是主要 full-chunk 性能点；
- GDN snapshot/record：保留显式 `[rows,W,B]`，不能把状态语义简化成展平后的 `T`。

GDN snapshot 的 public domain 保持现有合同：dense `B=1` 接受任意正 `W`；`B=2..8` 接受
`W=1..16`。Replay-record 的 public domain 是 `B=1..8`、`W=2..16`。Qwen3.8 MTP 只使用
其中 `W<=6` 的子集，但新增 format admission 不能缩窄已有语义 overload 的 domain。

### 2.2 Row-scaled FP8 `Weight` 合同

下文所有 FP8 项都消费一个完整、不可变的 `Weight`：

```text
qtype                 = QType::FP8_E4M3FN_ROW_BF16S
layout                = QuantLayout::RowScale
ndim                  = 2
n / k                 = registration 固定的精确 N / K
shape                 = [N,K]
padded_shape          = [N,K]
qdata                 = E4M3FN code plane，N*K bytes
qhigh                 = null
high_plane_bytes      = 0
scales                = BF16 row-multiplier plane，N words
scale_dtype           = DType::BF16
scale_ne              = [N,1,1,1]
scale_nb[0]           = 2
scale_nb[1..3]        = 2*N
group_size / group    = K
```

Scale plane 从 code plane 末尾按 256 bytes 对齐后开始。该格式没有 weight divisor、input divisor、
zero point、K-group scale 或独立命名的 scale tensor；任何 divisor 字段都不参与数值解释。

对于 E4M3FN code word `c[n,k]` 和 BF16 row multiplier `s[n]`，持久化数据所表示的权重为：

```text
c32        = exact_e4m3fn_to_binary32(c[n,k])
s32        = exact_bfloat16_to_binary32(s[n])
w_hat[n,k] = binary32(c32 * s32)
```

独立 Op fixture 必须先精确解码这条公式，再计算完整 Op；不能复制 source quantization、activation
quantization、MMA operand、staging cast、累加顺序或 production kernel。Runtime wrapper 需要验证
format identity、layout、rank、完整逻辑/物理 geometry、plane、scale metadata、payload bounds 和所需
alignment；对于受信 artifact，不在加载或调用时扫描每个 code/scale word。

### 2.3 Public activation-compute policy

除 `embedding` 外，每个新增 FP8 registration 都有 policy-bearing form。Artifact 所需的完整合同是
`LinearPolicy::AllowA8`，并覆盖该 registration 的整个 extent domain。Public activation 和输出仍是
BF16；private resolver 可以选择已验证的 A16 route，也可以先把 activation 私有量化为 FP8。该 policy
不承诺所有 extent 都走 A8，也不规定 activation encoder、scale granularity、workspace 表示或 MMA
instruction。

实现可以额外接纳有价值的 `A16Only` form。它必须声明精确 extent domain 并有直接 oracle 覆盖，
但不能代替完整正 extent 上的 `AllowA8` registration。`AllowA4` 不属于 row-scaled FP8 profile。

所有 policy-bearing capacity query 必须覆盖与 execution 相同的 format、geometry、policy 和 inclusive
extent interval。合法的零 workspace route 返回零；需要 activation quantization 或 staging 的 route
只能使用 caller-owned transient storage。

### 2.4 以实际 Op 为终点的 Linear-first 纵向工作流

Generic Linear 是同一 geometry 下实现实际 Op 的起步形式，不是一个独立交付阶段。每条 projection
工作流都按下面的连续顺序推进：

```text
精确 generic Linear registration
        -> 数值正确
        -> 完成当前 geometry 的完整 Linear extent 与 route 调优
        -> 固化可复用的 geometry / mainloop / kernel 模板
        -> 替换输出路径或增加 epilogue/post，形成实际语义 Op
        -> 直接验证和调优完整 public Op
```

具体工作流为：

| Geometry `[N,K]` | 连续实现路径 | 实际终点 |
|---:|---|---|
| `[248320,5120]` | L1 `linear` | full output head 本身就是 Linear；E1 是同 geometry 的独立 gather |
| `[14336,5120]` | L2 `linear` → 固化 kernel 模板 → A1 split-output epilogue | `attn_input_proj` |
| `[16384,5120]` | L3 `linear` → 固化 kernel 模板 → G1 split-output → G2/G3 convolution/state post | 三个 GDN form |
| `[34816,5120]` | L4 `linear` → 固化 kernel 模板 → M1 SwiGLU epilogue/post | `linear_swiglu` |
| `[5120,6144]` | L5 `linear` → 固化 kernel 模板 → R1 residual epilogue | `linear_add` |
| `[5120,17408]` | L6 `linear` → 固化 kernel 模板 → R2 residual epilogue | `linear_add` |

其中有以下约束：

- [ ] 不设置“先完成全部 L1..L6，再统一开始 fused Op”的全局门槛。L2 调优后立即做 A1；L3
  调优后立即做 G1/G2/G3；L4/L5/L6 同理，不等待其他 geometry。
- [ ] L2..L6 的 generic registration 提供最直接的数值验证和 kernel 调优入口，但 Linear 自身不是
  对应工作流的完成点。实际 consumer 的 epilogue、输出布局、非线性或状态效果完成之前，该
  geometry 的 Op 工作仍未完成。
- [ ] Linear 调优的产物必须是可直接改造成实际 Op 的 kernel primitive、mainloop、schedule、
  output-policy 接口和 workspace recipe，而不是只能写 dense BF16 输出的封闭实现。
- [ ] 当前 geometry 的 Linear 必须先完成完整正 `T` domain、全部可达 arithmetic profile、
  latency-sensitive interval、throughput anchor、workspace 和 production route；只有 `T=1` 或
  provisional mainloop 不能触发实际 Op 开发。
- [ ] 当前 Linear 完成后立即接入实际 Op；在这两者之间不要转去批量实现另一个 Linear geometry，
  也不要继续开展与实际 consumer 无关的候选扩张或性能工作。
- [ ] 实际 Op 先采用已经完成的 Linear route 候选和分界，再用完整 public call 测量 epilogue/post
  的影响。只有该测量可以调整最终 route；private kernel-only 结果不能单独决定实际 Op route。
- [ ] 复用 Linear kernel 不等于在 public 层调用 `linear`，也不要求先物化 dense BF16 projection。
  实际 Op 直接负责最终输出与状态副作用；私有 BF16 中间量只是某条 route 的 arithmetic profile，
  不是 oracle 必须复制的语义边界。
- [ ] 不为实际 Op 复制 FP8 decode、activation quantization 或 contraction mainloop。这些共性实现
  归 `src/ops/linear` 所有；实际 Op 只增加自己的 output policy、epilogue/finalizer、post 和语义
  dispatch。

这也是现有 NVFP4 的开发顺序：对一个具体 geometry 先把 generic Linear 的完整 domain 做通并
调优，再把同一 kernel/mainloop 改造成 attention、GDN、SwiGLU 或 residual 所需的输出形式；它
不是用 provisional Linear route 提前进入 fused，也不是“先做完所有 Linear，再另起一个 fused
阶段”。

### 2.5 本轮 kernel 建立与 route 调优目标

L1..L6 以及由 L2..L6 连续改造得到的 rank-two projection Op 使用以下统一尺度。它既规定第一个
代表性 geometry 的打通过程，也规定后续 geometry 如何复用模板；E1、G2 和 G3 的不同执行域在本节
末尾单独说明。

本清单把计算机制与调优区域作为两个正交维度：

- SIMT route 不使用 Tensor Core，直接消费 represented BF16 activation，不进行 activation
  quantization；当前 `T=1` kernel 属于该机制，同一机制可以按实测需要为其他 T 实例化；
- MMA route 使用 FP8 Tensor Core，因而需要把 public BF16 activation 私有量化为相应 MMA
  operand；MMA 可以出现在任意实测胜出的 T，并不天然只属于 prefill；
- `T=1`、`T=1..48` latency-sensitive interval 和 `T=1024` throughput anchor 描述的是调优与
  验证区域，不是三种 kernel 类别。

- [x] 开始第一个 FP8 contraction kernel 前，从 L2..L6 选择并记录一个代表性 geometry。选择必须
  同时给出其紧邻的实际 consumer、预期复用的 contraction mainloop 和需要参数化的 output
  boundary，不能选择一个没有实际 Op 落点的合成 problem。
- [x] 该 geometry 先打通完整 public Linear 链路：row-FP8 Weight admission、`AllowA8`、workspace
  capacity、独立 fixture/oracle、conformance test 和 public Linear benchmark。Kernel 候选只能在
  这些入口能够验证真实 registration 后开始计时。
- [x] 首先实现和调优 `T=1` kernel。每个候选以编译期 schedule 参数形成明确实例；临时 benchmark
  可以直接调用内部 launcher 强制候选，但必须使用相同输入、cache 状态、设备和 timing 方法，并在
  候选进入 production selector 后通过 public Linear 重新验证。
- [x] 在 `T=1024` 建立和调优 MMA mechanism。该阶段只完成 throughput anchor，没有提前 sweep
  `T=1..48`、确定 SIMT/MMA crossover 或进入相邻实际 Op。
- [x] MMA activation quantization 的 execution decomposition 按证据决定，不预先强制实现两条
  route。若完整 production route 尚未接近相应硬件 roofline，再以完整调用比较片上量化与独立
  materialization；比较时保持 quantization 公式、scale 粒度和表示一致，并计入全部 launch、
  workspace traffic、MMA 与输出。L2 `T=1024` public Linear 已达到 `364.672 us`、`412.22
  TFLOP/s`，即 FP32 accumulation 的 `419 TFLOP/s` 稠密峰值的 `98.38%`，理论余量不足以改变
  decomposition 决策，因此停止追加片上量化候选。
- [x] `T=1024` 是 Text projection prefill 的主性能锚点。Production selector 必须实际采用经同一
  oracle 验证的 A8 Tensor Core route；public Linear 测量覆盖完整调用，同时以所选 contraction 的
  FP8 tensor-core roofline 作为 kernel 调优退出目标。`T>48` 只测回答当前吞吐或后续 crossover
  问题所需的少量点，不做逐 T 密集 candidate sweep。L2 当前 A8 route 已通过独立 oracle 数值
  criterion，并达到上述 roofline 退出目标。
- [x] `T=1024` MMA 收敛后，测 surviving MMA 与多 token SIMT 实例在较低 T 的表现。2026-08-17
  的完整 candidate×T 复审推翻了早先“从 T=2 起 A8 胜出”的结论：L2 在 `T=2..11` 仍由 A16
  SIMT 胜出，只有 `T>=12` 才切到 A8。Production selector 已编码 `T<=11 -> A16`、
  `T>=12 -> A8`，同时保留 `T=1` decode 与 compile-time multi-token SIMT 实例。
- [x] 对每个整数 `T=1..48` sweep 所有仍合法且有竞争力的 SIMT/MMA 候选。该 interval 覆盖
  `B<=8` 与 `W<=6` 的并发和 speculative-decode 上界；即使某个整数不能直接写成当前的 `B*W`，
  也保留连续 sweep，以确定真实 crossover 和相邻 extent 的延迟。只有 crossover 附近的证据需要时
  才增加面向较低 T 的 MMA 实例，不预先实例化参数笛卡尔积。
- [x] Latency-sensitive interval 内按逐 T 测量选择最低且可重复的 route。最终 public benchmark
  必须只调用 public Op，并报告完整 `T=1..48` latency 曲线及相邻 median 变化；每个 production
  boundary 在 `b-1/b/b+1` 重新验证。不得保留可由另一合法候选避免的明显 latency cliff；候选差异
  落在测量不确定度内时，采用边界更稳定且更便于同一 kernel 模板复用的选择。RTX 5090、CUDA
  13.1、cold-cache、10 次 public-call 的 L2 最终曲线为 `50.432..63.840 us`；最大相邻增幅正是
  route seam `T=11 -> 12` 的 `58.656 -> 62.464 us`（`+6.49%`），未再出现未报告的额外 cliff。
- [x] 只有当前 geometry 的完整正 `T` Linear semantics、A16/A8 profile、workspace、`T=1..48`
  route 和 `T=1024` 吞吐目标全部完成后，才修改输出路径形成相邻 A/G/M/R Op。随后在实际 Op 的
  完整 public call 上重复相应曲线、boundary 和 throughput anchor；epilogue/post 改变最终 winner
  时，以实际 Op 的 public 结果为准。

L1 output head 同样覆盖 `T=1..48` hot interval，但不是 Text prefill 的 `T=1024` Tensor Core
锚点。E1 对 `T=1..48` 和 `T=1024` 测量完整 gather，但不套用 contraction roofline。G2/G3 保留
显式 `[rows,W,B]` 语义：hot-path sweep 覆盖当前产品可达的 `B=1..8`、`W<=6` 组合及其 public
boundary，不能为了复用 rank-two 曲线而把状态 Op 的 public domain 展平为任意 `T`。

### 2.6 当前 L2→A1 里程碑

第一个代表性 geometry 已固定为 `[N,K]=[14336,5120]`，相邻实际 consumer 是 A1
`attn_input_proj`。当前完成事实如下：

- generic `linear` 与 fused `attn_input_proj` 都已注册该 geometry 的全部正 `T`；`A16Only` 与
  `AllowA8` 均合法。L2 在 `T<=11` 采用 A16、`T>=12` 采用 A8；A1 依据完整 split-output 调用
  独立采用 `T<=10` A16、`T>=11` A8；
- A16 GEMV、A8 activation quantization/workspace、MMA mainloop 和 production schedule 由 Linear
  所有；A1 只替换 output policy，直接写 Q/K/gate/V 四个最终 allocation，没有物化 packed parent
  输出或复制 contraction 实现；
- 独立 fixture 精确解码 E4M3FN code 与 BF16 row scale；public Linear 的 convenience/`AllowA8`
  form，以及 public A1 的 A16/A8、tail、T=1024 和四个语义 row range，均已直接通过同一
  naive-FP64 oracle；
- RTX 5090、CUDA 13.1、cold-cache、80 次 public-call 测量中，Linear A16 median/min/p95 为
  `50.464/49.632/51.200 us`，one-read effective bandwidth 为 `1455.8 GB/s`，即 sustained-read
  probe 的 `86.94%`；A1 的 A16/AllowA8 median 分别是 `50.432/50.464 us`，split-output
  epilogue 与 dense output 的差异落在测量波动内；
- 原生 FP8 MMA 候选的最佳 public median 为 `69.632 us`，未达到 direct route；BF16 packed
  accumulation 不满足 A16 数值 criterion，因此 T=1 保留 direct route；
- T=1024 A8 public Linear median 为 `364.672 us`、`412.22 TFLOP/s`，达到 FP32 accumulation
  稠密峰值 `419 TFLOP/s` 的 `98.38%`；public A1 为 `363.776 us`、`413.23 TFLOP/s`，达到
  `98.62%`。2026-08-17 的 10 次 cold-cache public sweep 中，L2 `T=1..48` 为
  `50.432..63.840 us`，最大相邻增幅是 `T=11 -> 12` 的 `+6.49%`；A1 为
  `50.464..64.320 us`，最大相邻增幅为 `+4.64%`，其 `T=10 -> 11` route seam 反而从
  `62.752` 降至 `62.080 us`。Split-output 因而拥有与 L2 不同但平滑的最终 boundary。

T=1 没有达到早期提出的 `<=48.75 us` / sustained-read `>=90%` 参考目标；在 direct A16、原生
FP8 MMA 和 BF16 packed accumulation 的实测与数值筛选后，约 `50.3 us` direct A16 仍是最终
production 选择。该未达到的参考值记录为测量结果，不再作为 L2/A1 的退出阻塞。完整
`T=1..48` 曲线、A16/A8 boundary、workspace、正 `T` domain、四输出 oracle 和 `T=1024`
roofline 目标现已共同完成，因此 L2→A1 纵向工作流已退出。

### 2.7 当前 L3→G1 里程碑

`[N,K]=[16384,5120]` 的 generic `linear` 与 G1 `gdn_input_proj` 已完成以下资格验证：

- 两个 public Op 都注册全部正 `T`、`A16Only`、`AllowA8` 和真实 caller-workspace capacity；
  L3 的 `AllowA8` 在 `T<=10` 解析为 A16、从 `T=11` 解析为 A8；G1 独立采用 `T<=7` A16、
  `T>=8` A8；
- L3 显式注册该 geometry 的 A16/A8 production schedule，同时复用既有 GEMV、activation
  quantization 和 MMA mainloop；G1 直接写入 `qkv [10240,T]` 与 `z [6144,T]`；
- public Linear 的 A16/A8 与 public G1 的 Q/K/V/Z row range、非整 tile tail 和 `T=1024` 均通过
  独立 E4M3FN/BF16-row-scale decode 加 naive-FP64 oracle；workspace query 与 execution high-water
  一致，已有 Q4/Q5、W8 和 NVFP4 G1 form 保持通过；
- RTX 5090、CUDA 13.1、cold-cache、10 次 public-call 复审中，L3 `T=1..48` 为
  `58.592..68.768 us`，最大相邻增幅是 `T=6 -> 7` 的 `+5.47%`，`T=10 -> 11` route seam
  从 `68.768` 降至 `65.536 us`；
- L3 的 `T=1024` public median 为 `423.232 us`、`405.92 TFLOP/s`，达到 FP32-accumulate FP8
  峰值 `419 TFLOP/s` 的 `96.88%`；完整 G1 为 `423.200 us`、`405.95 TFLOP/s`，同为
  `96.88%`；
- G1 的 `T=1..48` 为 `58.592..67.584 us`，最大相邻增幅是 A16 区间内 `T=6 -> 7` 的
  `61.408 -> 66.848 us`（`+8.86%`）；`T=7 -> 8` route seam 略降至 `66.528 us`。完整 G1
  benchmark 因而支持与 L3 不同的 boundary。

L3→G1 的完整正 `T` domain、数值 profile、workspace、hot interval 和吞吐锚点已经完成。G2/G3
仍需在各自的显式 `B/W/state` public domain 上独立资格验证，不能由本里程碑代替。

### 2.8 当前 L4→M1 里程碑

`[N,K]=[34816,5120]` 的 generic `linear` 与 M1 `linear_swiglu` 已完成以下资格验证：

- 两个 public Op 都注册全部正 `T`、`A16Only`、`AllowA8` 和真实 caller-workspace capacity；
  L4 的 `AllowA8` 在 `T=1` 采用 A8、`T=2..4` 采用 A16、`T>=5` 回到 A8；M1 独立采用
  `T=1` A8、`T=2` A16、`T>=3` A8；
- public Linear 的 A16/A8，以及 public M1 的 gate/up row order、非整 tile tail、最终 SwiGLU
  输出和 `T=1024` 均通过独立 E4M3FN/BF16-row-scale decode 加 complete naive-FP64 oracle；
  workspace query 与 execution high-water 一致，已有 Q4、W8、NVFP4 LinearSwiGLU form，以及
  已完成的 FP8 attention/GDN projection 保持通过；
- RTX 5090、CUDA 13.1、cold-cache、30 次 public-call 候选比较中，L4 的 `T=1` A8/A16
  median 分别为 `121.984/126.208 us`，完整 M1 分别为 `122.080/124.160 us`，因此该 geometry
  没有沿用较窄 N 的 T=1 A16 boundary；
- 2026-08-17 的 10 次 cold-cache public sweep 中，L4 `T=1..48` 为 `118.784..125.536 us`，
  最大相邻增幅为 `T=2 -> 3` 的 `+2.80%`；M1 为 `118.048..125.920 us`，最大相邻增幅正是
  `T=2 -> 3` route seam 的 `+3.25%`。非单调 arithmetic route 没有形成明显 latency cliff；
- L4 的 `T=1024` public median 为 `827.680 us`、`441.08 TFLOP/s`；完整 M1 为
  `828.704 us`、`440.53 TFLOP/s`。两者都越过 benchmark harness 固定的 FP8/FP32-accumulate
  `419 TFLOP/s` 参考线；针对完整 M1 public `T=1024` 调用中 fused MMA kernel 的 NCU 2025.4.1
  捕获报告 `SM: Pipe Tensor Cycles Active = 90.80%`、`Compute (SM) Throughput = 90.80%`。
  因此固定参考线比值只作为吞吐调优退出证据，不解释为 NCU 利用率，也不再追加 kernel 候选。

L4→M1 的完整正 `T` domain、A16/A8 数值 profile、workspace、hot interval、SwiGLU 语义和吞吐
锚点已经完成。

### 2.9 当前 L5→R1 与 L6→R2 里程碑

两个 residual geometry 的 generic `linear` 与实际 `linear_add` 已连续完成：

- L5/R1 `[5120,6144]` 与 L6/R2 `[5120,17408]` 都接纳全部正 `T`、`A16Only`、`AllowA8` 和
  caller-owned activation workspace；capacity query 与 execution high-water 一致；
- L5/L6 分别注册自己的 decode 与 MMA schedule，同时复用既有 row-FP8 decoder、activation
  quantizer 和 contraction mainloop。R1/R2 在相同 mainloop 的输出 epilogue 中读取 original
  residual、以 FP32 完成相加并直接写最终 BF16 residual，没有物化 public Linear 输出；
- public Linear 的 A16/A8 与 public LinearAdd 的完整 `original_residual + W*x` 公式、两个真实
  geometry、route 两侧、非整 MMA tile `T=65` 和 `T=1024` 均通过独立 E4M3FN/BF16 row-scale
  decode 加 naive-FP64 oracle；activation 与 weight 保持不变；
- L5/L6 的共享权重多 token SIMT route 消除了逐 token launch，两个 generic resolver 均采用
  `T<25 -> A16`、`T>=25 -> A8`。10 次 cold-cache public sweep 中，L5 的 `T=1/2` 为
  `24.576/23.840 us`，已消除早先漏报的近翻倍变化；完整区间为 `23.840..56.320 us`。最大相邻
  增幅是 `T=19 -> 20` 的 `38.176 -> 47.104 us`（`+23.39%`），route seam `T=24 -> 25` 为
  `48.448 -> 54.528 us`（`+12.55%`）；
- L6 的完整区间为 `62.464..141.312 us`，最大相邻增幅是 SIMT 内部 `T=17 -> 18` 的
  `92.192 -> 107.808 us`（`+16.94%`），route seam `T=24 -> 25` 为
  `121.696 -> 140.288 us`（`+15.28%`）。这些仍存在的寄存器/occupancy seam 明确保留在证据中，
  不以区间最小/最大值代替逐点报告；
- R1 的 fused epilogue 改变了 crossover，独立确定 `T<22 -> A16`、`T>=22 -> A8`。其
  `T=1/2` 为 `25.408/25.696 us`，完整区间为 `25.408..57.792 us`；最大相邻增幅是
  `T=12 -> 13` 的 `34.080 -> 40.960 us`（`+20.19%`），`T=19 -> 20` 还存在
  `47.104 -> 56.480 us`（`+19.91%`），而 `T=21 -> 22` route seam 略降；
- R2 独立保持 `T<25 -> A16`、`T>=25 -> A8`。新增 fused 专属 streaming/v8/低 row schedule
  后，完整区间为 `60.672..143.808 us`，最大相邻增幅降为 `T=19 -> 20` 的
  `109.856 -> 121.920 us`（`+10.98%`），`T=24 -> 25` route seam 为 `+6.11%`；
- R1/R2 的最终 schedule 来自一次包含两个 geometry、`T=2..24` 和 18 套 warp/row/vector/cache/
  unroll/token-tile 候选的完整开发矩阵。R1 上述两个约 20% seam 在该矩阵中没有更快且更平滑的
  SIMT 或 A8 替代，因而作为当前不可避免的已知结果接受；开发期 private-launcher benchmark 与
  losing instances 已删除；
- L5 与 R1 的 `T=1024` public median 分别为 `175.104/177.344 us`、
  `367.92/363.27 TFLOP/s`；L6 与 R2 分别为 `476.928/480.480 us`、
  `382.73/379.90 TFLOP/s`。对应 Linear contraction-only MMA 分别达到 `400.17` 与
  `413.35 TFLOP/s`，完整 route 的剩余时间来自必须执行的 activation quantization，因此不再扩张
  kernel 候选。

L5→R1 与 L6→R2 的完整正 `T` domain、独立 route、A16/A8 数值 profile、workspace、原地 residual
语义和 throughput anchor 均已完成。

## 3. Artifact 到 consumer 的完整账本

### 3.1 Row-scaled FP8 parent

| Artifact 角色 | 完整 parent `[N,K]` | 层/位置 | 当前直接语义 consumer | Generic Linear 起步项 |
|---|---:|---:|---|---|
| token embedding | `[248320,5120]` | global，1 | `embedding` | 与 L1 同 geometry，但不是 contraction |
| full-attention Q/K/gate/V input | `[14336,5120]` | 16 个 full-attention layer | `attn_input_proj` | L2 |
| full-attention output | `[5120,6144]` | 16 个 full-attention layer | `linear_add` | L5 |
| GDN Q/K/V/Z input | `[16384,5120]` | 48 个 GDN layer | `gdn_input_proj`、snapshot、record | L3 |
| GDN output | `[5120,6144]` | 48 个 GDN layer | 与 attention output 相同的 `linear_add` registration | L5 |
| MLP gate/up | `[34816,5120]` | layer `56..63`，8 | `linear_swiglu` | L4 |
| MLP down | `[5120,17408]` | layer `56..63`，8 | `linear_add` | L6 |
| full output head | `[248320,5120]` | global，1 | `linear` | L1 |
| **FP8 物理 parent 总数** |  | **146** |  |  |

48 个 GDN input parent 对应三个语义 consumer，是因为不同 schedule 对同一个不可变权重施加不同的
完整状态效果；它们仍是 48 个物理 artifact object，不是 144 个。反过来，16 个 attention output
和 48 个 GDN output 合并成同一个 `linear_add [5120,6144]` registration，因为 Op 不携带模型角色
或 layer discriminator。

Generic Linear registration 的数量也不等于直接 artifact consumer 的数量。L2..L6 是各自实际
consumer 工作流的实现与调优起点，不会在 target schedule 中额外插入一次 Linear 调用，也不构成
独立于实际 consumer 的交付阶段。

### 3.2 已有 NVFP4 parent

| Artifact 角色 | 完整 parent `[N,K]` | 层/位置 | 已有精确 consumer | 状态 |
|---|---:|---:|---|---|
| MLP gate/up | `[34816,5120]` | layer `0..55`，56 | `linear_swiglu`，NVFP4 BlockScaleK16M128x4 | existing-exact |
| MLP down | `[5120,17408]` | layer `0..55`，56 | `linear_add`，NVFP4 BlockScaleK16M128x4 | existing-exact |
| **NVFP4 物理 parent 总数** |  | **112** |  |  |

这两个 shape 的完整 parent row order、E2M1/E4M3FN/divisor 解码、A16/A4 policy、workspace query、
独立 oracle suite 和 public benchmark 都已经注册。Qwen3.8 的 NVFP4 parent 不会消费已有的 NVFP4
attention-input、GDN-input 或 `[5120,6144]` residual registration；这些额外 registration 的存在
不会增加本文工作量。

## 4. 必需 registration 总表

共需 14 个精确 row-scaled FP8 registration：

| ID | 类别 | 语义 Op | 精确 FP8 parent `[N,K]` | Public 结果/效果 | Artifact 覆盖 | 状态 |
|---|---|---|---:|---|---|---|
| L1 | generic | `linear` | `[248320,5120]` | `BF16 [248320,T]` | full head，1 个 parent | [ ] |
| E1 | gather | `embedding` | `[248320,5120]` | gather `BF16 [5120,T]` | embedding，1 个 parent | [ ] |
| L2 | generic | `linear` | `[14336,5120]` | `BF16 [14336,T]` | A1 工作流的 Linear 起步项 | [x]（T<=11/A16、T>=12/A8；完整正 T 与性能目标已完成） |
| A1 | fused projection | `attn_input_proj` | `[14336,5120]` | 独立 Q/gate/K/V 输出 | 16 个 parent | [x]（完整正 T；T<=10/A16、T>=11/A8） |
| L3 | generic | `linear` | `[16384,5120]` | `BF16 [16384,T]` | G1/G2/G3 工作流的 Linear 起步项 | [x]（完整正 T；T<=10/A16、T>=11/A8） |
| G1 | fused projection | `gdn_input_proj` | `[16384,5120]` | 独立 QKV 与 Z 输出 | 48 个 parent | [x]（完整正 T；T<=7/A16、T>=8/A8） |
| G2 | fused projection/state | `gdn_input_proj_conv_snapshot` | `[16384,5120]` | Q/K/V/Z 加 convolution-state snapshot | 同 48 个 parent | [ ] |
| G3 | fused projection/record | `gdn_input_proj_conv_record` | `[16384,5120]` | Q/K/V/Z 加 replay projection record | 同 48 个 parent | [ ] |
| L4 | generic | `linear` | `[34816,5120]` | `BF16 [34816,T]` | M1 工作流的 Linear 起步项 | [x]（完整正 T；T=1/A8、T=2..4/A16、T>=5/A8） |
| M1 | fused projection | `linear_swiglu` | `[34816,5120]` | `BF16 [17408,T]` SwiGLU | layer `56..63`，8 个 parent | [x]（完整正 T；T=1/A8、T=2/A16、T>=3/A8） |
| L5 | generic | `linear` | `[5120,6144]` | `BF16 [5120,T]` | R1 工作流的 Linear 起步项 | [x]（完整正 T；T<25/A16、T>=25/A8） |
| R1 | fused projection | `linear_add` | `[5120,6144]` | 原地 residual `BF16 [5120,T]` | 64 个 parent | [x]（完整正 T；独立实测 T<22/A16、T>=22/A8） |
| L6 | generic | `linear` | `[5120,17408]` | `BF16 [5120,T]` | R2 工作流的 Linear 起步项 | [x]（完整正 T；T<25/A16、T>=25/A8） |
| R2 | fused projection | `linear_add` | `[5120,17408]` | 原地 residual `BF16 [5120,T]` | layer `56..63`，8 个 parent | [x]（完整正 T；独立实测 T<25/A16、T>=25/A8） |

位置数量只用于证明 artifact 覆盖，不是 runtime dispatch 输入，不能出现在 Op wrapper 或 kernel
selector 中。总表按 geometry 把 Linear 起步项与实际 Op 相邻排列：L2→A1、L3→G1/G2/G3、
L4→M1、L5→R1、L6→R2。L2..L6 虽然各自是有效的 Linear registration，但必须与相邻的
A/G/M/R 项作为一条连续工作流推进。

## 5. 各纵向工作流的通用 FP8 Linear 起步项：L1..L6

### 5.1 六个 registration 的共同合同

对每个 L 项分别完成以下工作。L1 的实际终点就是 output-head Linear；L2..L6 完成这些步骤后，
应立即进入同一 geometry 的 A/G/M/R 项，而不是转去批量实现下一个 Linear：

- [ ] 只接纳表中精确的完整 row-scaled FP8 `Weight [N,K]`；不得把完整 parent 拆成持久化 row
  child，也不得把同 shape 的其他 qtype/layout 一并放宽。
- [ ] 接受连续、非空、16-byte-aligned 的 `x BF16 [K,T]`，写入与之分离且连续、非空、
  16-byte-aligned 的 `out BF16 [N,T]`，覆盖每个正 `T`。
- [ ] 要求 x、out、两个 weight plane 和 live workspace 满足完整 non-overlap 合同；weight 不可变，
  Op 不持有 persistent state。
- [ ] 注册 `LinearPolicy::AllowA8`，并让 `linear_workspace_capacity_bytes` 对同一精确 problem、
  policy 和所有合法正 inclusive `T` interval 返回 execution 的真实上界。
- [ ] 从精确解码的 `w_hat` 与 represented BF16 input 计算每个完整 dot product，并直接与公共
  naive-FP64 Linear oracle 比较。Oracle 不插入 private activation quantization 或 production
  accumulation tree。
- [ ] 对每个可达 A16/A8 arithmetic profile 使用明确的数值 criterion，覆盖 route 起点、终点、
  相邻 interior 点和真实 geometry；不能只与另一个 kernel 做 pairwise parity。
- [ ] 在 public Linear benchmark 中加入该 exact point，用它调优 contraction kernel、形成可复用
  模板并完成当前 Linear 的 production route。计时必须包含 activation quantization、workspace
  traffic 和完整 Linear call，不能用 private mainloop timing 代替。
- [ ] 在 RTX 5090/CUDA 13.1 上为该 geometry 的实际生产 extent 和所有 route 分界建立足够的
  选路证据：hot interval 按第 2.5 节穷举 `T=1..48`，更大 extent 只测决定 prefill route 所需的
  稀疏点并以 `T=1024` 为主锚点。
- [ ] 把调优得到的 geometry、schedule、kernel primitive、output-policy 接口、activation
  workspace recipe 和完整 route facts 放在 Linear-owned 实现中；当前 Linear 完成后立即用它实现
  对应 A/G/M/R consumer。保留仍需在实际 Op 中比较的少量候选，待完整 fused public route 确定后
  再删除落选候选和 benchmark-only private entry point。
- [ ] 保持所有现有 Q4/Q5/Q6/W8/BF16/NVFP4 Linear registration 与回归。

### 5.2 各 geometry 的特定职责

| ID | 输入 `x` | 输出 `out` | 关键实测 extent | 直接用途及后续复用 |
|---|---|---|---|---|
| L1 | `BF16 [5120,T]` | `BF16 [248320,T]` | hot interval 每个 `T=1..48`、route boundary | full output head 的真实 public consumer；不对应 fused 项 |
| L2 | `BF16 [5120,T]` | `BF16 [14336,T]` | hot interval 每个 `T=1..48`、prefill `T=1024`、route boundary | 完成该 Linear 后立即改造成 A1 |
| L3 | `BF16 [5120,T]` | `BF16 [16384,T]` | hot interval 每个 `T=1..48`、prefill `T=1024`、route boundary | 完成该 Linear 后立即改造成 G1，并供 G2/G3 复用 |
| L4 | `BF16 [5120,T]` | `BF16 [34816,T]` | hot interval 每个 `T=1..48`、prefill `T=1024`、route boundary | 完成该 Linear 后立即改造成 M1 |
| L5 | `BF16 [6144,T]` | `BF16 [5120,T]` | hot interval 每个 `T=1..48`、prefill `T=1024`、route boundary | 完成该 Linear 后立即改造成 R1 |
| L6 | `BF16 [17408,T]` | `BF16 [5120,T]` | hot interval 每个 `T=1..48`、prefill `T=1024`、route boundary | 完成该 Linear 后立即改造成 R2 |

L1 的 `linear` 不带 vocabulary-domain 语义。下游 sampling 和 `argmax` 继续接收现有 valid-row
limit `248077`；它们不因 physical output 有 248320 行而新增 format admission。

L2..L6 的 dense BF16 输出是 public Linear 自身的真实结果，也是开发与调优 contraction 最直接的
切入面；它们不是 fused Op 的强制中间表示，也不是独立阶段的终点。每个 L 项一旦形成可复用模板，
并完成当前 Linear 的整个 extent domain 和 production route，就继续修改同一实现路径以交付
A1/G1/G2/G3/M1/R1/R2。实际 Op 必须直接满足自己的最终输出合同，
不能仅以“与 L2..L6 的 BF16 输出再做后处理一致”作为数值正确性证明。

## 6. Embedding

### E1 — row-scaled FP8 `embedding [248320,5120]`

- [ ] 接纳一个完整的 row-scaled FP8 table `[vocab,D] = [248320,5120]`。
- [ ] 接受连续 `ids I32 [T]`，写入连续 `out BF16 [5120,T]`，覆盖每个正 `T`；每个 id 位于
  `[0,248320)`。
- [ ] 计算 `ideal[d,t] = w_hat[ids[t],d]`。重复 id 是相互独立的 gather；即使产品 sampling
  只产生 token id `0..248076`，padded artifact row 仍是合法的 Op 输入。
- [ ] 要求 ids、out 与两个 weight plane 互不重叠。Table 不可变，Op 没有 workspace 或
  persistent state。
- [ ] 扩展独立 embedding fixture：独立精确解码 E4M3FN/BF16 row，覆盖 signed zero、zero-scale
  row、重复/边界 id、全部输出元素、未修改输入，以及 malformed row-scale metadata。
- [ ] 保持 dense BF16、Q6-D5120、W8-D5120 和 W8-D2048 embedding admission 与回归。
- [ ] 在 public embedding benchmark 中加入该 exact profile，覆盖每个 `T=1..48`，并以 `T=1024`
  作为 prefill gather 的主要吞吐点。

E1 没有 activation-compute policy：它只 gather 并重建选中 row，不执行 activation matrix
contraction。它是独立工作流，不参与 Linear 到实际 projection Op 的连续改造过程。

## 7. 全注意力输入投影

### A1 — 单 parent row-scaled FP8 `attn_input_proj`

- [x] 在 `[14336,5120]` 纵向工作流中，L2 完成数值验证与调优、形成 output-policy 可扩展的
  kernel 模板后，立即把其 dense-output 路径改造成 A1 的四输出形式；不等待其他 L 项完成。
- [x] 复用 L2 的 FP8 decode、geometry、A16/A8 contraction mainloop、schedule 候选和 activation
  workspace recipe，不新增独立 FP8 contraction 实现。
- [x] 接纳一个完整 `query_key_gate_value` parent `[14336,5120]`；不得接纳持久化逻辑 row child
  或多个 FP8 weight argument。
- [x] 接受连续 `x BF16 [5120,T]`，覆盖每个正 `T`。
- [x] 写入四个彼此独立的连续 BF16 输出：

  ```text
  q    [6144,T]
  gate [6144,T]
  k    [1024,T]
  v    [1024,T]
  ```

- [x] 严格按下列物理 parent row order 解释权重：

  ```text
  query       [0,6144)
  key         [6144,7168)
  output_gate [7168,13312)
  value       [13312,14336)
  ```

  Public argument order 为 `q, gate, k, v`，有意不同于物理 parent 顺序。
- [x] 从同一 represented `x` 计算四个完整逻辑 projection；不存在可观察的 packed
  `BF16 [14336,T]` 输出。
- [x] 要求 input、完整 parent、live workspace 和四个输出 allocation 互不重叠。
- [x] 注册 `AllowA8`，并扩展 `attn_input_proj_workspace_capacity_bytes`，覆盖 exact FP8
  problem 和每个正 `T` interval。
- [x] 默认采用 L2 的 route 分界；如果 split-output epilogue 改变最优分界，只能依据 A1 的完整
  public benchmark 调整，并在 route test 中固定新分界。
- [x] 扩展独立 full-Op oracle，检查所有四段 row、语义输出位置和全部输出，而不是只检查 aggregate
  projection error 或与 L2 做 parity。
- [x] 保持 two-parent 27B Q4/Q5、single-parent 27B BF16/NVFP4 和 35B W8 admission。
- [x] 扩展 public Attention-input benchmark，覆盖每个 `T=1..48`、route 分界和主要 `T=1024`
  prefill 点。

## 8. GDN 输入投影族

完整 FP8 GDN parent 的物理 row order 为：

```text
query [0,2048)
key   [2048,4096)
value [4096,10240)
z     [10240,16384)
```

Q、K、V 组成 10240 个 causal-convolution channel；Z 是独立 output gate，不参与 convolution
history、snapshot 或 replay record。`[16384,5120]` 工作流先用 L3 做通和调优 contraction，模板
形成后立即把 dense-output 改成 G1 的 QKV/Z 输出，再在同一实现序列中把该 projection 模板接入
G2/G3。G1、G2、G3 复用 L3 的 geometry、decoder、mainloop、schedule 和 activation workspace，
只分别拥有自己的输出映射、convolution/SiLU 和状态效果；它们不是 L3 之后另开的独立阶段。

### G1 — 单 parent row-scaled FP8 `gdn_input_proj`

- [x] 紧接 L3 的 kernel 模板和 route 调优结果实现 G1，不等待其他 geometry；先把 output policy
  改成独立 qkv/z allocation，再直接测量完整 G1 call。
- [x] 接纳一个完整 FP8 `query_key_value_z` parent `[16384,5120]`。
- [x] 接受连续 `x BF16 [5120,T]`，覆盖每个正 `T`。
- [x] 写入两个独立的连续输出：

  ```text
  qkv BF16 [10240,T]  # query 2048，key 2048，value 6144
  z   BF16 [6144,T]
  ```

- [x] 从同一 represented input 计算四个 projection。Z 不是后续 public `linear` call，也没有
  可观察的 packed `[16384,T]` 结果。
- [x] 要求 input、parent、workspace、qkv 和 z 互不重叠。
- [x] 注册 `AllowA8`，扩展 `gdn_input_proj_workspace_capacity_bytes`，覆盖 exact FP8 problem
  与每个正 `T` interval。
- [x] 默认沿用 L3 route；只有 G1 完整 split-output benchmark 能支持不同 route 分界。
- [x] 扩展独立 oracle，直接检查 Q、K、V、Z 逻辑 row range 与两个最终 allocation。
- [x] 保持 27B two-parent Q4/Q5、single-parent NVFP4 和 35B W8 form。
- [x] 扩展 public GDN-input benchmark，覆盖每个 `T=1..48`、route 分界和 `T=1024`。

### G2 — row-scaled FP8 `gdn_input_proj_conv_snapshot`

- [ ] 接纳与 L3/G1 相同的完整 FP8 parent `[16384,5120]`，以及 artifact 存储
  `[4,10240]` convolution 的 runtime transpose view `conv_weight BF16 [10240,4]`。
- [ ] 接受：

  ```text
  x                   BF16 [5120,W,B]
  conv_states         BF16 [10240,3,Slots]
  valid_columns       empty 或 I32 [B]
  initial_state_slots I32 [B]
  snapshot_base_slots I32 [B]
  ```

  `B=1` 接受任意正 `W`；`B=2..8` 接受 `W=1..16`。每个 mixed-width valid extent 位于
  `[1,W]`；每个 `initial_state_slots[b]` 位于 `[0,Slots)`，caller 预留的完整
  `[snapshot_base_slots[b],snapshot_base_slots[b]+W)` 也必须位于 `[0,Slots)`。
- [ ] 写入：

  ```text
  query BF16 [2048,W,B]
  key   BF16 [2048,W,B]
  value BF16 [6144,W,B]
  z     BF16 [6144,W,B]
  ```

- [ ] 从每个选中的 width-three history 开始，仅对 projection 后的 Q/K/V 执行 width-four
  depthwise convolution 和 SiLU；Z bypass convolution 与 state。
- [ ] 把每个有效 destination history 写到
  `[snapshot_base_slots[b], snapshot_base_slots[b]+valid_columns[b])`，其余 state word 保持不变。
  Query/key/value 的 invalid tail 精确写零；Z 对每个安全的物理 `B*W` input column 都执行
  projection，保持现有语义合同。
- [ ] 初始 history 完成加载后，保持现有合法的 same-row initial/destination overlap；保留 caller
  对每行 non-destructive state interval 的前置条件。不能为了 host-side alias validation 同步读取
  device selector。
- [ ] 注册 `AllowA8`，扩展 policy-bearing snapshot capacity query，覆盖 exact FP8 profile、
  exact `B` 和请求的每个 `W` inclusive interval。
- [ ] G1 output policy 做通后，立即把同一 L3 mainloop 接入 snapshot post；默认以刚得到的 L3/G1
  route 为起点，再用 G2 完整 public call 决定 projection+post route。不得用只测 projection 的
  结果替代。
- [ ] 扩展独立 complete-Op oracle，覆盖 projection、convolution、SiLU、四个输出、所有写入的
  snapshot、invalid-tail zeroing 和未修改 state region。覆盖 dense/ragged batch、相关 initial-slot
  关系、`B=1`、`B=8`、`W=1`、`W=16` 与全部 private route boundary，但不构造无意义的笛卡尔积。
- [ ] 保持 Q4/Q5、W8 和 NVFP4 snapshot form。
- [ ] 在 public snapshot/record benchmark 中加入 FP8，覆盖 eager/captured、cold/warm weight、
  target width `1..6` 和 public boundary width。

### G3 — row-scaled FP8 `gdn_input_proj_conv_record`

- [ ] 接纳相同完整 FP8 parent `[16384,5120]` 和 `conv_weight BF16 [10240,4]`。
- [ ] 接受：

  ```text
  x                   BF16 [5120,W,B]
  conv_states         BF16 [10240,3,Slots]  # read-only
  valid_columns       empty 或 I32 [B]
  initial_state_slots I32 [B]
  conv_record         BF16 [10240,W,B]
  ```

  执行 domain 为 `B=1..8`、`W=2..16`；每个 supplied valid extent 位于 `[1,W]`，每个
  `initial_state_slots[b]` 位于 `[0,Slots)`。
- [ ] 写入 query/key `BF16 [2048,W,B]`、value/z `BF16 [6144,W,B]`，并把 convolution 所消费的
  represented projected Q/K/V column 写入 `conv_record`。
- [ ] 应用与 G2 相同的 projection、width-four convolution、SiLU 和 Z-bypass 语义，但不得修改
  `conv_states`。只有每行 `conv_record` 的有效 prefix 有语义定义；query/key/value invalid tail
  精确为零，Z 覆盖每个物理 column。
- [ ] 按现有 record 合同要求 source state、parent、input、record、outputs 与 live workspace
  相互分离。
- [ ] 注册 `AllowA8`，扩展 record capacity query，覆盖 exact FP8 profile、exact `B` 与请求的
  每个 `W` inclusive interval；caller-owned `conv_record` 不是 workspace。
- [ ] 与 G2 共享刚形成的 L3/G1 projection 模板，并立即接入 record post；默认从 L3/G1 route
  开始，只依据 G3 完整 public benchmark 调整自己的分界。
- [ ] 扩展独立 complete-Op oracle，覆盖 projection、convolution、SiLU、Z、record publication、
  ragged-tail 效果，以及 source state 的精确保持。
- [ ] 保持 Q4/Q5、W8 和 NVFP4 replay-record form。
- [ ] 通过同一 public snapshot/record benchmark 测量 G3 的 target 与 public-domain boundary；
  private projection-only timing 不能证明 G3 性能。

## 9. Text MLP gate/up 投影

### M1 — row-scaled FP8 `linear_swiglu [34816,5120]`

- [x] 在 `[34816,5120]` 纵向工作流中，L4 数值验证与调优后立即把 kernel 模板改造成 M1 的
  SwiGLU epilogue/post，不等待其他 L 项完成。
- [x] 复用 L4 的 FP8 decode、geometry、activation quantization、contraction mainloop、schedule
  和 route 候选。
- [x] 接纳精确的完整 FP8 parent `[34816,5120]`，row order 为
  `[gate 17408, up 17408]`。
- [x] 接受连续 `x BF16 [5120,T]`，写入连续 `out BF16 [17408,T]`，覆盖每个正 `T`。
- [x] 要求 x、完整 parent、live workspace 和 out 互不重叠；weight 不可变，Op 不持有
  persistent state。
- [x] 实现完整逻辑公式：

  ```text
  gate[:,t] = Wgate * x[:,t]
  up[:,t]   = Wup * x[:,t]
  out[:,t]  = SiLU(gate[:,t]) * up[:,t]
  ```

- [x] Gate/up projection 保持 private。物化 `BF16 [34816,T]` 既不是必需实现，也不是可观察的
  语义边界。
- [x] 注册 `AllowA8`，扩展 `linear_swiglu_workspace_capacity_bytes`，覆盖 exact FP8 problem
  和每个正 `T` interval。
- [x] 以 L4 route 为起点；允许使用复用 L4 mainloop 但带 fused SwiGLU epilogue 的 kernel，或在
  合适 extent 使用 Linear-owned kernel 加 private post。最终分界只由 M1 的完整 public benchmark
  决定。
- [x] 从精确解码的 FP8 row 和 represented BF16 input 扩展独立 complete-formula oracle，直接检查
  最终输出；A16/A8 profile 仅在 arithmetic profile 实质不同的情况下使用不同的具名 criterion。
- [x] 保持现有 Q4、W8 和 NVFP4 LinearSwiGLU registration。
- [x] 在 public LinearSwiGLU benchmark 中加入 FP8 profile，覆盖每个 `T=1..48`、route boundary
  和 `T=1024`。计时把 activation quantization、workspace traffic、contraction 与 SwiGLU 视为
  一次完整 public call。

## 10. 残差投影

两个 registration 都保持一个完整语义更新：

```text
ideal[:,t] = original_residual[:,t] + W * x[:,t]
```

Final BF16 residual 是 caller 唯一可见的 projection 结果。Public `linear` 再调用 `residual_add`
会暴露一个舍入后的 BF16 projection，不是本文规定的语义边界；但 fused kernel 的 contraction
mainloop 必须分别复用 L5/L6。

### R1 — row-scaled FP8 `linear_add [5120,6144]`

- [x] 在 `[5120,6144]` 纵向工作流中，L5 数值验证与调优后立即用其 output-policy 接口加入
  residual epilogue，不等待其他 L 项完成。
- [x] 复用 L5 的 decode、geometry、activation workspace、contraction kernel/mainloop 与 route
  候选。
- [x] 接纳精确 FP8 parent `[5120,6144]`。
- [x] 接受连续 `x BF16 [6144,T]`，原地更新连续 `residual BF16 [5120,T]`，覆盖每个正 `T`。
- [x] 一个 registration 同时覆盖 16 个 full-attention output 和 48 个 GDN output parent；不得按
  layer type、object name 或 site count dispatch。
- [x] 要求 x、parent plane、live workspace 和 residual 遵守完整 non-overlap 合同。
- [x] 注册 `AllowA8`，扩展 `linear_add_workspace_capacity_bytes`，覆盖 exact problem 与每个正
  `T` interval。
- [x] 以 L5 kernel 候选为起点，由 R1 的完整语义 benchmark 独立确定自身 resolver 分界。
- [x] 从 represented original residual 增加直接的 complete-formula oracle 覆盖。

### R2 — row-scaled FP8 `linear_add [5120,17408]`

- [x] 在 `[5120,17408]` 纵向工作流中，L6 数值验证与调优后立即用其 output-policy 接口加入
  residual epilogue，不等待其他 L 项完成。
- [x] 复用 L6 的 decode、geometry、activation workspace、contraction kernel/mainloop 与 route
  候选。
- [x] 接纳精确 FP8 parent `[5120,17408]`。
- [x] 接受连续 `x BF16 [17408,T]`，原地更新连续 `residual BF16 [5120,T]`，覆盖每个正 `T`。
- [x] 注册 `AllowA8`，扩展同一 capacity query，覆盖 exact problem 与每个正 `T` interval。
- [x] 以 L6 kernel 候选为起点，由 R2 的完整语义 benchmark 独立确定自身 resolver 分界。
- [x] 从 represented original residual 增加直接的 complete-formula oracle 覆盖。

R1 与 R2 共同还需：

- [x] 保持 Q5、W8、BF16 和 NVFP4 LinearAdd registration 及其 policy domain。
- [x] 在 public LinearAdd benchmark 中加入两个 FP8 geometry，覆盖每个 `T=1..48`、route
  boundary 和 `T=1024`；计时必须包含 activation quantization、residual traffic 和所选的全部
  semantic kernel。

## 11. Linear 到实际 Op 的连续改造验收

除各 Op 自身的正确性与性能外，每条 L→实际 Op 工作流还必须满足：

- [ ] 映射固定为 A1←L2、G1/G2/G3←L3、M1←L4、R1←L5、R2←L6；实现、调优记录和
  plan/config 使用同一 geometry 标识，不把两端当成无关项目。
- [ ] 不存在“全部 generic Linear 完成”的前置里程碑，但当前 L 项必须先完成自己的完整正 extent
  domain、arithmetic profile、workspace 和 production route。随后在同一开发序列中立即增加对应
  output policy、epilogue 或 post，并转向实际 Op 的 correctness 与 benchmark。
- [ ] Generic Linear exact registration、oracle case 和 benchmark point 可以保留，作为最小
  contraction 验证入口；它们的保留不构成独立实施阶段，也不能替代实际 consumer 的完成证据。
- [ ] Weight validation、row-FP8 decode 和 activation quantization 使用 Linear-owned 共性实现；
  实际 Op 目录中不出现语义相同的复制版本。
- [ ] Latency-sensitive/throughput 各区域的实际 Op 候选由刚完成调优的同 geometry Linear kernel
  primitive/mainloop/schedule 改造而来。它可以使用不同 output visitor、epilogue 或 post kernel，
  但不能另写等价 contraction mainloop。
- [ ] 实际 Op workspace query 复用相同 activation workspace recipe，并追加本 Op 所需 transient
  allocation；query 与 execution 使用同一 plan facts。
- [ ] Linear benchmark 给出完整候选和 route 分界后，立即测量相邻的实际 Op extent。若最终 route
  相同，route test 固定该关系；若不同，只记录完整 public-call 中 epilogue/post 导致的变化。
- [ ] Linear oracle 证明 contraction；实际 Op complete-formula oracle 独立证明最终公式和副作用。
  两项在同一纵向工作流内连续完成，但不能互相替代。
- [ ] 最终代码中没有为衔接 Linear 与实际 Op 而保留的 duplicate decoder、duplicate quantizer、
  duplicate route table、只能 dense-output 的封闭 kernel 分叉或落选实现。

## 12. 已有支持与排除项证明

### 12.1 NVFP4 MLP parent

Qwen3.8 唯一的 NVFP4 consumer 已经精确覆盖：

- generic `linear` 已注册 NVFP4 BlockScaleK16M128x4 的五个完整 geometry：
  `[14336,5120]`、`[16384,5120]`、`[34816,5120]`、`[5120,6144]` 和
  `[5120,17408]`；
- `linear_swiglu` 已注册 `[34816,5120]`、BF16 `[5120,T] -> [17408,T]`、完整 fused
  公式、policy-bearing workspace、独立 A16/A4 conformance 和 public benchmark；
- `linear_add` 已注册 `[5120,17408]`、BF16 input 与原地 residual、policy-bearing workspace、
  独立 A16/A4 conformance 和 public benchmark；
- 现有 fused NVFP4 路径是在对应 geometry 的 Linear 做通和调优后，立即从 Linear-owned
  geometry、mainloop、schedule、activation quantization 和 kernel template 改造出来；不存在
  独立的“全部 Linear”阶段。这是第 2.4 节 FP8 纵向工作流的直接先例。

Qwen3.8 parent 使用相同完整 shape、row order、scale/divisor 语义和 public tensor geometry。
Layer range `0..55` 只改变 artifact multiplicity。因此本 artifact 不需要新增 NVFP4 Op admission、
kernel、test profile 或 benchmark profile。后续 target binding 仍必须把每个 parent 关联到其各自存储的
input divisor；这不是 Op checklist 项。

### 12.2 MTP-private 与 Vision 权重

MTP 保持五个已有精确 geometry 的 W8 matrix、一个 Q4 optimized draft head、BF16 norm 和现有
semantic call。它 alias 的 token embedding 与 full head 分别由 E1 和 L1 覆盖，不新增 MTP Op。

Vision 保持 Q4/Q5/Q6/W8 allocation，以及 Qwen3.8 groupwise-int target 已经执行的全部精确
geometry。其 Linear、normalization、bias、GELU、RoPE、attention、merger 和 scatter consumer
没有收到变化后的 represented input、output、state 或 weight format。

### 12.3 非 projection Text Op

Text norm、GDN A/B control projection、convolution weight、A-log、dt-bias、recurrent GDN
transition、gated RMSNorm、RoPE、softmax attention、sigmoid gate、sampling、argmax、state
selection 和 scalar/index transform 保持现有 represented input 与效果。Row-scaled FP8 在所属
projection 边界被消费并产生 BF16 语义结果，不会产生新的 attention、recurrent-state、pointwise、
cache 或 sampling Op。

Artifact reader、`row-scale-v1` materialization、`QType` mapping 和 FP8 embedding encoder 已在
representation/producer 边界受到保护；这些检查不能计作上述 14 个 Op registration 中的任何一个。

## 13. 公共完成标准

单个 registration 只有满足所有适用要求后才能勾选；勾选 L2..L6 只表示 Linear 起步项本身成立，
不表示对应纵向工作流完成。工作流必须继续推进到相邻的实际 Op：

- [ ] 权威 contract comment 写明 row-scaled FP8 format、完整精确 parent geometry、公式、逻辑输入
  输出、extent domain、副作用、alias、policy 与 workspace，不把 kernel 组织方式写成语义。
- [ ] Wrapper 只接纳预期 FP8 format/problem，验证完整 runtime `Weight` metadata 与 public tensor，
  拒绝错误的 format、layout、rank、shape、plane、scale、alignment、extent、policy 和 alias 组合。
- [ ] 每个 policy-bearing workspace query 与 execution 在完整 inclusive interval 上使用同一 route
  facts 与 allocation recipe；execution high-water 不超过报告 capacity，Op 不做 hidden device
  allocation。
- [ ] Test-owned fixture 独立构造并精确解码 E4M3FN code 与 BF16 row multiplier，覆盖 finite
  extrema、subnormal、signed zero 和 zero-scale row；production decoder 不能充当 oracle。
- [ ] 每个可达 A16/A8 production arithmetic profile 都直接对同一个完整数学 oracle 验证，使用
  real geometry 与 route boundary。只有 arithmetic profile 实质不同时，A16/A8 才使用不同的
  具名 criterion。
- [ ] Matrix/fused reduction 使用 normwise criterion 和有限的 gross pointwise cap；embedding 使用
  pointwise criterion；每个 criterion 明确 non-finite policy。
- [ ] 测试验证全部输出、记录的 mutation、语义 row placement、invalid-tail/untouched-region、
  input/weight preservation 和 workspace guard。与 Q4/Q5/W8/NVFP4 的 pairwise parity 只能作为
  补充证据。
- [ ] 扩展 wrapper 所影响的 Qwen3.6/Qwen3.8 groupwise、Qwen3.6 NVFP4、35B W8、MTP 与 Vision
  Op regression 保持通过。
- [ ] 相应长期 public Op benchmark 接纳 FP8 exact geometry 并报告完整 call。Production dispatch
  依据 RTX 5090/CUDA 13.1 上第 2.5 节规定的 hot-interval latency 曲线、route boundary 和适用的
  `T=1024` roofline 证据选择；private kernel-only 胜出或 unpack-to-dense fallback 都不足以完成
  该项。
- [ ] 临时候选控制、重复 decoder、落选实现和 benchmark-only private entry point 在 production
  选择后删除。

## 14. 执行方式与清单退出条件

执行单位是一个 geometry 的纵向工作流，不是“Linear 阶段”和“fused 阶段”。具体方式如下：

1. 一次性建立共用 row-FP8 Weight validation，以及 test-owned encoder/decoder fixture 和通用
   malformed-weight case。
2. 从下面任意一条实际 consumer 工作流开始；工作流之间没有“全部 Linear 完成”的依赖：

   ```text
   L2 -> 调优/模板化 -> A1
   L3 -> 调优/模板化 -> G1 -> G2/G3
   L4 -> 调优/模板化 -> M1
   L5 -> 调优/模板化 -> R1
   L6 -> 调优/模板化 -> R2
   ```

3. 在选中的工作流内按第 2.5 节连续完成：public registration → `T=1` SIMT → `T=1024` MMA
   mechanism 与必要的量化 decomposition 决策 → `T=1..48` 的 SIMT/MMA 逐点选路 → 当前
   geometry 的完整 Linear semantics、workspace、oracle 与 public benchmark → 实际 output
   policy/epilogue/post → complete-Op oracle 与完整 public benchmark。当前 Linear 完成前不进入
   实际 consumer；完成后不得转去批量实现其他 Linear，必须继续同一 geometry 的实际 Op。
4. 后续 geometry 可以实例化和扩展已经形成的公共模板；每次仍先用其 generic Linear exact point
   完成该 geometry 的 correctness、完整 extent 和 route，再立即实现该 geometry 的实际 Op。公共
   模板变化时，只回归受影响的已完成工作流。
5. L1 是 full output head 的实际 Op，完成自身 Linear 工作流即可；E1 是独立 gather 工作流。二者
   不需要等待或阻塞上述 fused 工作流。
6. 所有纵向工作流完成后，运行受影响 Op regression，并核对第 3 节 146 个 FP8 parent 和 112 个
   NVFP4 parent 都有且仅有对应的语义 consumer。
7. 然后进入 target binder、leaf dispatch 和 Engine 集成工作。

本清单全部勾选只表示 Op 层已经支持所需 format/problem；它不表示
`qwen3.8-27b/nvfp4` 已能经 Engine 选择、绑定、加载、capture 或执行。14 个 registration 的稳定
contract、test 和 benchmark authority 都已更新，且后续 target 集成不再被 Op 缺口阻塞时，应删除
这份临时清单，而不是保留一份已经完成的迁移日志。
