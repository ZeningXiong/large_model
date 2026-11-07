# qms：在 4–8 GB 内存的笔记本上运行 Qwen3-235B-A22B

**qms** 是一个用 C 写的 Qwen3-MoE 推理引擎（Windows、纯 CPU）。它把 2350 亿参数的 MoE 模型
Qwen3-235B-A22B（Q4_0 GGUF，124 GiB）的大部分权重留在 NVMe SSD 上，只在需要时读取，
**在可用内存约 4 GB 或 8 GB 的条件下也能运行和对话**，并且读取方式不会改变计算结果。

思路来自 [kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c)（在 8 GB 内存上运行 2.78T 参数的 Kimi K3），
本项目针对 Qwen3-MoE 的结构和 Windows 平台重新实现，并做了逐步验证与测量。

*A C inference engine that runs Qwen3-235B-A22B (MoE, Q4_0 GGUF) on a Windows laptop with 4–8 GB of available RAM by
streaming routed experts (and, at 4 GB, part of the dense trunk) from NVMe. Output is bit-identical regardless of how
much memory is used.*

---

## 实测结果

测试机：Ryzen 9 7845HX（12 核，AVX-512），32 GB DDR5，PCIe 4.0 NVMe（实测随机读上限约 6 GB/s），Windows 11。
"可用内存"通过锁定其余内存来限制（`tests/memhog.c`）。同一段中文 prompt，生成 16–32 个 token。

| 可用内存 | qms | llama.cpp b12144（mmap） | qms 峰值内存 |
|---|---:|---:|---:|
| 8 GB | **约 1.9 s/token**（开启"边读边算"，见 Optimize.md） | 约 6.2 s/token | 7.35 GB |
| 4 GB | **约 2.6 s/token** | 约 7.5 s/token | 3.60 GB |

- 速度主要取决于 SSD：每生成一个 token 要读 6–9 GB 的专家权重，约 80% 的时间在等磁盘。
  在 SSD 较慢、CPU 较弱的电脑上会更慢（估计 3–5 s/token）。
- 这台笔记本的测量波动较大（同一配置前后可差 20–40%），每个数字只重复了 2–3 次；
  详细数据、测试方法与不严谨之处见 [Process.md](Process.md) 和 [Optimize.md](Optimize.md)。

## 工作原理（简述）

1. **MoE 稀疏性**：每层 128 个专家只选 8 个，每个 token 只用到约 22B 参数。专家权重（约 120 GB）**从不全部载入**，
   用到时从 GGUF 文件直接读取（绕过系统文件缓存、批量异步读取）。
2. **主干常驻或流式**：每个 token 都要用的权重（注意力、router、输出层，共 4.83 GB）在 8 GB 档全部常驻；
   在 4 GB 档只固定前若干层，其余层每个 token 按固定顺序预读（"固定前缀 + 环形槽"，来自 kimi-k3-in-c）。
3. **专家缓存**：剩余内存用来缓存读过的专家，淘汰策略为**衰减 LFU**（使用频率每 8 个 token 减半）。
   用真实访问轨迹回放比较过：LRU 在小缓存下命中率为 0%。
4. **边读边算**：一层的专家哪个先读完就先算哪个，不等全部读完。
5. **整数计算内核**：激活量化为 int8，与 4/6 bit 权重做整数点积（AVX-512 VNNI / AVX2，有标量回退）。
6. **自写分词器与对话**：从 GGUF 读取词表实现 Qwen2 分词（与 llama-tokenize 在 360 段测试文本上完全一致），
   ChatML 多轮对话、温度 / top-k / top-p 采样。

**正确性**：所有"读取方式"类的优化（流式读取、缓存、预取、主干流式、边读边算）都要求与"全部权重常驻内存"
的输出**逐位一致**；计算本身与 numpy 参考实现、llama.cpp 交叉验证。

## 环境要求

| 项目 | 要求 |
|---|---|
| 系统 | Windows 10/11 x64（POSIX 代码路径写了但未测试） |
| CPU | 支持 AVX2（AVX-512 VNNI 更快）；编译用 `-march=native`，生成的程序只适用于本机 CPU |
| 内存 | 可用内存 ≥ 4 GB |
| 磁盘 | 约 130 GB 放模型，越快越好（建议 PCIe 4.0 NVMe） |
| 编译 | [MSYS2](https://www.msys2.org) UCRT64：`gcc`、`make`、`libgomp` |
| 测试（可选） | Python 3.12 + `numpy`、`gguf`；[llama.cpp](https://github.com/ggml-org/llama.cpp) 发布版（用于生成测试模型和对照） |

## 快速开始

### 1. 编译

在 MSYS2 UCRT64 终端（或把 `D:\msys64\ucrt64\bin` 加入 PATH 的终端）中：

```bash
pacman -S --needed mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-libgomp make
make            # 生成 bin/qms.exe（静态链接，不依赖 DLL）
```

### 2. 下载模型

[unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF](https://huggingface.co/unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF)
的 `Q4_0` 目录（3 个分片，共 133 GB）。国内可用镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF --include "Q4_0/*" --local-dir E:/models/qwen3-235b
```

ModelScope 上也有同名仓库，文件 sha256 相同，可用于断点续传。

### 3. 对话

```bat
set QMS_MODEL=E:\models\qwen3-235b\Q4_0\Qwen3-235B-A22B-Instruct-2507-Q4_0-00001-of-00003.gguf
chat_8gb.bat          :: 约 7.7 GB 内存
chat_4gb.bat          :: 约 3.75 GB 内存
```

输入问题按回车；空行或 `/exit` 退出。可附加参数，例如 `chat_8gb.bat --system "回答不超过三句话" --seed 42`。

### 4. 命令行

```text
qms chat MODEL [选项]                     交互式多轮对话
qms run  MODEL --ids a,b,c --gen N [选项]  按 token id 生成（测试与测速用）
qms tok  MODEL 文本文件                    分词，输出 token id
qms detok MODEL a,b,c 输出文件             把 token id 解码为文本
qms info MODEL                            显示 GGUF 元数据
```

| 常用选项 | 作用 |
|---|---|
| `--mem-gb M` | 总内存预算：先固定主干，余下给专家缓存（自动开启流式读取） |
| `--stream-experts` `--cache-gb G` | 手动方式：专家流式读取，专家缓存 G GB |
| `--direct` | 读盘绕过系统文件缓存（测量真实内存占用时必须开启） |
| `--ctx N` | 上下文长度（KV cache 按此预先分配） |
| `--cache-policy lfu\|lru` | 专家缓存淘汰策略（默认衰减 LFU） |
| `--no-overlap` | 关闭"边读边算"（对照测试用） |
| `--prefetch D --prefetch-min-p P` | 预测后 D 层的专家并提前读取（默认关闭，收益未确认） |
| `--exact` | 使用 fp32 精确计算路径（慢，用于验证） |
| `--temp` `--top-k` `--top-p` `--seed` `--greedy` | 采样参数（chat） |

启动时会打印内存计划；若计划超过可用内存的 95%，程序拒绝启动。

## 测试

```bash
python -m venv .venv && .venv/Scripts/python.exe -m pip install numpy gguf
make fixtures    # 生成 3 个随机权重的小测试模型（需要任意一个 Qwen3 GGUF 作为分词器来源，见 Makefile 的 TOKSRC）
make test        # 冒烟、内核、反量化、与参考实现对照、48 种流式组合逐位一致、分词器、多轮对话
```

`make test` 只用小模型，几分钟即可完成。真实模型上的对照与测速脚本在 `tools/` 和 `bench/`
（脚本中 llama.cpp 和模型的路径是本机路径，使用前需修改）。

## 目录结构

```text
src/           引擎源码：gguf.c（GGUF 解析）、quant.c / kernels.c（反量化与整数内核）、model.c（前向、
               流式读取、专家缓存、主干环形槽）、aio.h（异步读）、tok.c（分词器）、chat.c（对话）、main.c（命令行）
tests/         C 单元测试与工具：test_kernels.c、devbw.c（测 SSD 带宽）、memhog.c（锁定内存模拟小内存机器）、peakrun.c
tools/         Python：ref_qwen3moe.py（numpy 参考实现）、cmp_*.py（对照）、test_*.py（回归）、sim_cache.py（缓存策略模拟）、
               make_tiny.py（生成测试模型）、gen_unicode.py（生成 Unicode 分类表）
bench/         测速脚本与原始结果；traces/ 为专家访问轨迹
chat_8gb.bat   8 GB 档对话启动脚本
chat_4gb.bat   4 GB 档对话启动脚本
```

## 文档

| 文件 | 内容 |
|---|---|
| [PLAN.md](PLAN.md) | 改造计划与各步骤进度 |
| [Process.md](Process.md) | 第 0–10 步每一步改了什么、结果、不严谨之处 |
| [Optimize.md](Optimize.md) | 提速优化（边读边算、按把握程度预取）的 A/B 测试记录 |
| [bench/BASELINE.md](bench/BASELINE.md) | llama.cpp 基准测试记录 |

## 已知局限

- 只在一台机器上测试过；测量噪声较大，重复次数少（详见 Process.md）。
- 没有做输出质量评测（困惑度、任务准确率）；只验证了计算与参考实现一致。
- 只测试过几百 token 以内的上下文；KV cache 为 fp32，长上下文需要更多内存。
- 用户输入中的特殊 token 文本（如 `<|im_end|>`）会被当作控制符解析。
- 仅支持 Qwen3-MoE 架构（`qwen3moe`）的 GGUF，量化类型支持 F32/F16/Q4_0/Q4_1/Q8_0/Q6_K。

## 致谢

- [kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c)：流式读取 MoE 专家、固定前缀 + 环形槽、
  "内存优先给主干"、启动前内存预算、逐位一致验证等思路。本项目未复制其代码。
- [llama.cpp](https://github.com/ggml-org/llama.cpp)：GGUF 格式、量化工具，以及作为对照基准。
- [gguf-py](https://github.com/ggml-org/llama.cpp/tree/master/gguf-py)：参考实现中用于独立反量化。
- 模型：[Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507)，
  使用请遵守其模型许可。

## 许可

尚未选择许可证。
