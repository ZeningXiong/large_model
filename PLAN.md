# 改造计划：kimi-k3-in-c 的流式思路 → Qwen3-235B-A22B（CPU / 32 GB / Windows）

## 已确认的事实（2026-11-06 实测 / 实读）

- 基线：llama.cpp b12144，`--repack 0 -lm mmap`，tg32 = 0.42 tok/s（~2.4 s/token），pp128 = 2.12 tok/s。
  运行中磁盘读 ~1.4 GB/s（盘峰值 4.5–5.2 GB/s），CPU ~33%，可用内存 ~0.2 GB。→ 瓶颈是缺页同步等待，不是带宽或算力。
- 权重来源：已下载的 GGUF（Q4_0 为主），sha256 已校验。磁盘剩余 E: ~27 GB，D: ~75 GB → 放不下第二份 124 GB 重排副本，
  **方案是原地读取 GGUF**。
- GGUF 结构（tools/gguf_inspect.py 读出）：
  - 94 层，hidden 4096，64 Q 头 / 4 KV 头，head_dim 128，128 专家选 8，专家中间维 1536，rope base **5,000,000**，eps 1e-6。
  - 专家按层堆叠为 3 个张量：`ffn_gate_exps` / `ffn_up_exps` [4096,1536,128]，`ffn_down_exps` [1536,4096,128]；
    单个专家是每个张量里的一段连续切片 → **每个专家 3 次读**（gate/up 各 3,538,944 B；down 3,538,944 B 或 Q4_1 时 3,932,160 B，均为 4096 的整数倍）。
  - 量化类型混合：注意力/gate/up 为 Q4_0，部分层 `ffn_down_exps` 为 **Q4_1**，`output.weight` 为 **Q6_K**，router/norm 为 F32，embedding Q4_0。
  - 张量起始偏移**不是 4096 对齐**（mod 4096 = 640/1152/1664/…）→ 无缓冲读需要"对齐窗口读 + 指针偏移"。
- 工具链：本机**没有 C 编译器**；Python 有 numpy 1.26 / torch 2.5；transformers 4.46 不支持 Qwen3-MoE（不打算依赖它）。
  llama.cpp 自带 `llama-quantize` / `llama-tokenize` / `llama-server`，可用作对照。

## 阶段与验收门槛

| 阶段 | 内容 | 验收 |
|---|---|---|
| 0 环境 | MSYS2 + MinGW-w64 GCC（kimi-k3-in-c 已验证可用）；Python venv + numpy + gguf；项目骨架、测试脚本 | `make test` 空跑通过 |
| 1 参考实现 | numpy 版 GGUF 读取 + Q4_0/Q4_1/Q6_K 反量化 + Qwen3-MoE 前向（GQA、NEOX RoPE、QK-Norm、softmax top-8 归一化、SwiGLU）；用 gguf 包生成**随机小模型**（如 4 层/hidden 256/16 专家选 4），llama-quantize 成 Q4_0 | 小模型上 numpy 参考 vs llama.cpp：贪心 token 一致、top-k logprob 接近；真实模型第 0 层单层输出与参考一致 |
| 2 C 引擎核心 | C 版 GGUF 读取；内核 Q4_0/Q4_1 × Q8 激活点积（AVX2/AVX-512）、Q6_K、F32；RMSNorm、RoPE、带 KV cache 的注意力、MoE；全部常驻内存 | 小模型上与 numpy 参考在容差内一致，三种执行路径（整段重算 / 贪心 / 增量）token 一致 |
| 3 先跑 30B | 同架构的 Qwen3-30B-A3B（16 GB，能装下）作为中间目标 | 与 llama.cpp 贪心 token 一致（多 prompt × 32 token），速度量级相当（参考 22 tok/s） |
| 4 235B 朴素流式 | 主干（~3.8 GB）+ lm_head 常驻；专家按需同步无缓冲读（每专家 3 段对齐窗口）；启动前内存预算检查（照搬 k3_run 的做法） | 235B 贪心 token 与 llama.cpp 一致；得到第一个速度数 |
| 5 I/O 引擎 | IOCP 异步 + `FILE_FLAG_NO_BUFFERING`，每层 8 专家 × 3 段一次性提交、按偏移排序、队列深度 32–64；专家槽位缓存（沿用 k3_cache 的 EMPTY/INFLIGHT/key 三态 + 三阶段读取） | 磁盘读速接近 4–5 GB/s；tg ≥ 基线 0.42 tok/s |
| 6 缓存策略 | 记录专家访问轨迹（沿用 `--trace` + `sim_cache.py`），离线比较 LRU / 热门专家静态固定 + LRU / Belady；实现最优者 | 命中率 ≥ 当前页缓存水平（估计约一半） |
| 7 预取 | 用下一层 router（F32 4096×128，很小）对当前隐藏状态打分，提前读下一层专家；读盘与计算重叠 | 预测命中率、浪费字节数有数据；tg 目标 ≥ 1 tok/s（推算上限，非承诺） |
| 8 完整功能 | 预填充按专家去重（chunk-union）；分词器（GGUF 内置 gpt2 BPE + Qwen 预分词规则，先用 llama-tokenize 对拍）；ChatML 模板；采样器；状态保存/续接 | 与 llama-tokenize 输出 token 完全一致；对话可用 |
| 9 测量 | 内存阶梯（Windows Job Object 硬限 8/16/24/32 GB）对比 llama.cpp；同一组问题对比回答 | 出一份与 BASELINE.md 同格式的对比报告 |

## 设计原则（从 kimi-k3-in-c 继承）

- 先加总全部内存需求再分配，超 95% 可用内存拒绝启动；报告实测峰值 RSS。
- 专家读取失败计数，任何丢专家的运行以非零退出。
- 每一步优化都先确认输出 token 不变，再看速度；单次测速差异小于噪声不算结论（重复 3 次）。
- 计算布局的转换只能离线做（llama.cpp 的 repack 在模型大于内存时直接失败就是反例）。

## 不做 / 推后

- 不降 top-k、不跳专家（有损）。
- 不另存 124 GB 重排副本（磁盘不够）；若以后加盘，可把"专家连续 + 4096 对齐"重排作为可选优化。
- 不依赖 transformers / PyTorch 作参考。

## KV cache 增长的处理（补充）

Qwen3-235B：4 KV 头 × 128 × (K+V) × 94 层 × 2 B = 192,512 B ≈ **188 KB/位置（f16）**；int8 94 KB；4k=0.75 GB，32k=6 GB，128k=24 GB（f16）。

1. 统一内存池：固定部分（主干、缓冲）+ KV（增长）+ 专家缓存（弹性，KV 增长时让出槽位）。启动时按最大上下文做预算，保证"主干 + 最大 KV + 最小专家缓存"放得下。
2. KV 默认 int8（与 f16 对比 logits 误差后定）；4-bit 作为可选。
3. 超出内存份额时按层把 KV 溢出到 SSD：注意力按固定层序访问，与主干同样用"固定前 N 层 + 环形槽 + 预取下一层"。
4. 多轮之间保存/恢复 KV（沿用 --save-state）。
5. 有损选项（滑动窗口 + attention sink、按重要性丢 token、对话摘要压缩）只作为显式开关，默认关闭。

## 进度（2026-11-07 全部完成）

| 步骤 | 状态 | 关键结果 |
|---|---|---|
| 0 环境 | ✅ | MSYS2 UCRT64 gcc 16.2 + make（D:\msys64），.venv（numpy、gguf） |
| 1 参考实现 | ✅ | tools/ref_qwen3moe.py；F32 tiny 与 llama.cpp 差 ~0.004；235B 4/4 贪心一致 |
| 2 C 引擎核心 | ✅ | 反量化与 gguf-py 逐位一致；精确路径与参考差 ~2e-5 |
| 3 30B | ✅ | int8×Q4 VNNI 内核 + 批量 MoE；同时段约为 llama.cpp 的 90% |
| 4 235B 朴素流式 | ✅ | 常驻 4.83 GB，峰值 5.5 GB；与 llama.cpp 前 6 token 一致，第 7 个分歧经精确路径裁决 qms 正确 |
| 5 读盘引擎 | ✅ | 无缓冲（--direct）+ 批量异步读 + 专家槽缓存；4.0 GB/s（盘上限 ~6 GB/s，tests/devbw.c） |
| 6 缓存策略 | ✅ | LRU 在 <16 GB 时命中 0%；衰减 LFU（半衰期 8 token）2 GB 10.8%、8 GB 43.2%，与模拟器完全一致；8 GB 缓存 1.25 s/token |
| 7 预取 | ✅ | --prefetch N：预测后续层专家并提前异步读；修复在途请求被移动（OVERLAPPED 地址）的 bug 与文件尾越界读的 bug。盘速提升到 5.2–6.2 GB/s，但错猜增加 9–22% 读量，净效果在噪声内 → 默认关闭。结论：瓶颈是带宽，减少读量（缓存）才是主杠杆 |
| 8 完整功能 | ✅ | 自写分词器（GGUF 词表+合并规则、Qwen2 预分词手写匹配、自生成 Unicode 表）：360 段文本与 llama-tokenize 完全一致、往返逐字节相同；qms chat：ChatML 多轮（KV 跨轮复用，与整段重算一致）、温度/top-k/top-p 采样、UTF-8 流式输出。235B 两轮中文对话正确记住上一轮信息，1.5–2.2 s/token（8 GB 缓存，总 13 GB） |
| 8GB 测试 | ✅ | 可用内存约 8 GB（锁定 17.6 GB）：qms 2.15/3.00 s/token、峰值 6.85 GB；llama.cpp 5.97/6.39 s/token、峰值 11.1 GB → qms 约快 2.4 倍。细节与不严谨之处见 Process.md |

运行时参考：`bin/qms.exe run MODEL --ids ... --gen N --stream-experts --direct --cache-gb G`；回归：`make test`（只用小模型）。
| 10 主干流式 + 4GB | ✅ | `--mem-gb M`：固定前缀 + 双槽环形读取（读主干等待 0.0 s）、embedding 按行读。锁定内存实测：qms 8 GB 2.10 s/token（峰值 7.35 GB）、4 GB 2.61 s/token（峰值 3.60 GB，慢约 24%）；llama.cpp 4 GB 5.95/9.04 s/token。详见 Process.md 第 10 步 |
