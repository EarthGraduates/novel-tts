# AMD 消费级显卡部署 TTS 模型实战：从踩坑到实时推理

> 硬件：AMD Radeon RX 7700 XT (RDNA3, 12GB VRAM) · ROCm 7.1 · PyTorch 2.12.1
>
> 如果你也有一块 AMD 显卡，想跑中文 TTS，这篇文章是写给你的。
>
> **⚠️ 本文基于 Claude Session 记录整理，部分数值可能存在出入，仅供参考。**

---

## 目录

1. [缘起：为什么要在 AMD 显卡上跑 TTS](#1-缘起为什么要在-amd-显卡上跑-tts)
2. [候选引擎横评：四个模型，四种体验](#2-候选引擎横评四个模型四种体验)
3. [Qwen3-TTS 优化全记录：从 RTF 1.59 到 0.76](#3-qwen3-tts-优化全记录从-rtf-159-到-076)
4. [ROCm 环境搭建的那些坑](#4-rocm-环境搭建的那些坑)
5. [novel-tts：把小说变成有声书的工具链](#5-novel-tts把小说变成有声书的工具链)
6. [总结与建议](#6-总结与建议)

---

## 1. 缘起：为什么要在 AMD 显卡上跑 TTS

我想把喜欢的中文小说转成有声书。需求很简单：拿来一本 TXT，走完流程，拿到可以听的 WAV。

家里有一台台式机，平时是给孩子玩游戏的，装了 AMD RX 7700 XT。既然有现成的硬件，就没必要为了跑个 TTS 专门去买 NVIDIA 显卡——AMD 显卡的性价比本来就高，同价位下显存也更充裕（12GB vs RTX 4060 Ti 的 8GB），对于跑推理来说，显存容量往往比算力更重要。

但真正开始折腾才发现，市面上中文 TTS 的部署教程几乎全部以 NVIDIA/CUDA 为前提。AMD 消费级显卡（RX 6000/7000 系列）的资料少得可怜——搜"ROCm TTS 部署"，出来的不是零几年的论坛帖子，就是"理论上支持"的官方文档。RDNA3 架构的 AI 生态其实在快速改善——ROCm 7.1 已经官方支持 gfx1101，PyTorch 的 ROCm 版本也越来越稳定——但缺的就是有人把这条路完整走一遍。

> 总要有人来做这件事。我踩过的坑，别人就不用再踩了。

我的硬件环境：

| 项目 | 详情 |
|------|------|
| GPU | AMD Radeon RX 7700 XT (gfx1101, RDNA3) |
| 显存 | 12 GB GDDR6 |
| ROCm | 7.1.52802 |
| PyTorch | 2.12.1+rocm7.1 |
| Python | 3.10.20 |
| OS | Ubuntu 26.04 |

---

## 2. 候选引擎横评：四个模型，四种体验

我先后测试了四个中文 TTS 引擎。每个都有独特的优势和致命的问题。

### 2.1 ChatTTS：参数最丰富，但自毁音质

ChatTTS 是我第一个尝试的引擎。它对 AMD 显卡支持很好，不需要额外折腾就能跑起来，速度也不慢。

它的最大亮点是**可调参数最丰富**——可以加入笑声、调整停顿、控制韵律。如果这些功能正常工作，它会是最好的选择。

但 ChatTTS 有一个致命问题：**团队为了防止模型被商用，在生成的音频中加入了噪音**。效果被刻意劣化后，音质完全达不到可用标准。不管参数怎么调，底噪始终存在。这条路走不通。

> **结论**：功能最强，但自废武功。不推荐。

### 2.2 F5-TTS：音质好，但设计上不适合长篇朗读

接着试了 F5-TTS。音质确实比 ChatTTS 好得多——清晰、自然、没有奇怪的噪音。

但它有一个设计层面的问题：**F5-TTS 是语音克隆模型，必须传入参考音频**。它没有内置的默认音色，所有生成都依赖一段参考音频来"模仿"目标说话人。

这意味着什么？如果你要朗读一整本小说（几十万字、几千个段落），每个段落都要用同一段参考音频。但参考音频的时长有限（通常 5-10 秒），模型从这短短几秒里提取的音色特征，在几千次生成中会产生漂移——**音色不稳定，前后听起来不像同一个人**。

除此之外还踩了几个坑：

- **参考音频内容泄露**：生成的开头偶尔会出现参考音频里的内容。我用了一段"你好，欢迎使用F5-TTS语音合成系统"作为参考，结果小说朗读开头冒出一句"大自然后母"——完全莫名其妙。
- **分词问题**："上山下乡"被逐字拆开读，中间还有停顿。这是训练数据覆盖不足的表现。
- **torchcodec 依赖 NVIDIA**：F5-TTS 的 torchcodec 依赖 `libnvrtc.so`（NVIDIA 专有库），在 AMD 上得 monkey-patch 用 soundfile 替代。

> **结论**：适合短文本语音克隆，不适合长篇朗读。音色不稳定是硬伤。

### 2.3 Qwen3-TTS：本地部署的最佳选择

阿里通义千问团队的 Qwen3-TTS 是转折点。它有三个关键优势：

1. **内置 9 个中文音色**（Vivian, Serena, Uncle_Fu, Ryan, Aiden, Eric, Dylan 等），不需要参考音频
2. **指令式语气控制**：旁白/温柔/愤怒/欢快/悲伤/严肃，一行文本指令即可
3. **也支持音色克隆**：传一段参考音频就能模仿新音色

0.6B 的小模型效果就相当不错——中文朗读自然流畅，语气切换明显。它不需要参考音频的设计意味着**全书朗读音色完全一致**，这对有声书来说是基本要求。

但它有一个小毛病：**不是所有汉字都会读**。一些比较敏感或粗俗的字眼会被跳过。对于网络小说来说偶尔会遇到，但大部分场景不影响使用。

性能方面，优化前 RTF（Real-Time Factor，实时率）是 1.59——比实时慢 60%。优化后降到 0.76，比实时快 30%。以《XX后裔》为例，经过 format-novel 处理后拆分为 3638 段，目前在 RX 7700 XT 上已稳定生成约 1500 段，暂无问题。优化过程见第 3 节。

> **结论**：本地部署首选。音色稳定 + 语气可控 + 不挑参考音频。0.6B 模型够用。

### 2.4 Edge TTS：不折腾的首选

微软 Edge TTS 是云端方案——免费、无需 GPU、音质好、什么字都能读。

我用 novel-tts 的 Edge TTS 引擎跑了一本百万字的小说，没有遇到调用频率限制。生成速度大约 32 秒/段（含网络延迟），和优化后的 Qwen3-TTS 64 秒/段相比更快。

唯一的"代价"是需要联网，以及从技术层面上说你依赖微软的服务。但如果你不想折腾 ROCm 环境、不想编译 flash-attention、不想调优参数——Edge TTS 就是最好的选择。

> **结论**：不想折腾就用这个。效果最好，成本为零。

### 2.5 横评总结

| 引擎 | 音质 | 音色稳定性 | AMD 兼容性 | 速度 | 推荐场景 |
|------|------|-----------|-----------|------|---------|
| ChatTTS | ⭐⭐ (有噪音) | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | 不推荐 |
| F5-TTS | ⭐⭐⭐⭐ | ⭐⭐ (不稳定) | ⭐⭐ (需 monkey-patch) | ⭐⭐ | 短文本克隆 |
| **Qwen3-TTS 0.6B** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ (需优化) | ⭐⭐⭐⭐ | **本地长篇朗读** |
| **Edge TTS** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ (无 GPU) | ⭐⭐⭐⭐ | **不折腾首选** |

---

## 3. Qwen3-TTS 优化全记录：从 RTF 1.59 到 0.76

这是本文的核心。Qwen3-TTS 0.6B 刚跑起来时，RTF 是 1.59——生成 1 秒音频需要 1.59 秒。一段 135 字的中文要等 60 秒。以《XX后裔》（处理后 3638 段）为例，这个速度意味着全书需要约 60 小时的生成时间，完全不可接受。

优化后 RTF 降到 0.76，135 字只需 27 秒。下面是完整的优化过程。

### 3.1 基线测试：float32

```
dtype:            float32
bench_RTF:        1.59
VRAM:             6.45 GB
135 字生成:       ~60s
50 字生成:        ~28s
```

用的是 `bench_dtype.py` 脚本，测试三段文本（12/50/135 字），每个 dtype 在独立子进程中运行以确保干净状态。

### 3.2 float16 翻车：AMD 上的 HSA 硬件异常

第一个想到的优化自然是半精度。结果：

```
dtype: float16
结果: 💥 HSA_STATUS_ERROR_EXCEPTION —— 直接崩溃
```

**float16 在 gfx1101 (RDNA3) 上不可用。** 这不是 Qwen3-TTS 的问题，是 ROCm 在 RDNA3 上对 fp16 的支持不完整。音频生成过程中某步运算触发了硬件异常，进程直接退出。

教训：**在 AMD 显卡上，不要假设 float16 能用。先测 bf16。**

### 3.3 bfloat16：决定性优化

```
dtype:            bfloat16
bench_RTF:        0.76  (-52%)
VRAM:             3.24 GB (-50%)
135 字生成:       ~27s  (-55%)
50 字生成:        ~11s  (-60%)
```

bfloat16 在 RDNA3 上有硬件支持，既快又稳。这是整个优化过程中**收益最大的单项措施**——时间减半、显存减半、音质无损。

> 如果你的 AMD 显卡是 RDNA3 (RX 7000 系列)，**第一步永远是切到 bf16**。

### 3.4 SDPA 的教训：单独开反而慢

ROCm 的 SDPA（Scaled Dot-Product Attention）experimental 模式可以启用更高效的 attention 计算路径。我天真地以为开了就会快。

结果：

| 配置 | bench_total | RTF | SDPA Warnings |
|------|-------------|-----|---------------|
| baseline | 90.0s | 1.59 | 2 |
| **SDPA-experimental 单独开** | **96.1s** | **1.59** | **0** ← 反而慢了 7%！ |
| SDPA-exp + Triton Autotune | 83.4s | 1.60 | 0 |

**单独开 SDPA experimental 不仅没加速，反而慢了 7%。** 必须同时启用 `FLASH_ATTENTION_TRITON_AMD_AUTOTUNE=TRUE`，让 Triton 的自动调优找到适合当前 GPU 的 kernel 配置，才能拿到正向收益。

教训：**ROCm 的优化措施不是独立开关——它们之间有依赖关系。开 A 必须配 B，否则可能负优化。**

### 3.5 torch.compile 的选择：reduce-overhead vs max-autotune

`torch.compile` 有三种模式。理论上 `max-autotune` 调优最激进，但实测：

```
eager (no compile):   16.9s baseline
reduce-overhead:      15.2s (1.11x)  ← 选这个
max-autotune:         16.1s (1.05x)  ← 反而更慢
```

`reduce-overhead` 模式专注于减少 Python/CUDA 交互的 overhead，对 TTS 这种中小模型效果最好。`max-autotune` 的激进 kernel fusion 在 0.6B 模型上没有额外收益，反而增加了编译时间。

实际代码中就一行：

```python
model.model = torch.compile(model.model, mode="reduce-overhead")
```

首次运行会触发 JIT 编译（warmup），后续推理自动享受加速。

### 3.6 MIOPEN_FIND_MODE=2

MIOpen 是 AMD 的深度学习卷积库。`MIOPEN_FIND_MODE=2` 让它在首次运行时搜索最优卷积算法并缓存，后续调用直接使用。

这个优化的效果集中在 speech tokenizer 的卷积层，大约带来 5 倍的卷积加速。对整个推理流程的贡献约 5%。

### 3.7 最终效果

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| RTF | 1.59 | **0.76** | 2.1× |
| 135 字 | ~60s | **~27s** | -55% |
| 50 字 | ~28s | **~11s** | -60% |
| 显存 | 6.45 GB | **3.24 GB** | -50% |

环境变量汇总（已内置到 novel-tts 中）：

```python
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.1")
os.environ.setdefault("MIOPEN_FIND_MODE", "2")
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_AUTOTUNE", "TRUE")
```

---

## 4. ROCm 环境搭建的那些坑

### 4.1 flash-attention 编译失败：aiter 的版本号解析 Bug

flash-attention 是优化 attention 计算的关键库。在 ROCm 上编译时，AMD 官方的 aiter 子模块会解析系统 HIP 版本来决定编译选项。

报错：

```
File "cpp_extension.py", line 177, in <genexpr>
    ROCM_VERSION = tuple(int(v) for v in HIP_VERSION.split(".")[:2])
ValueError: invalid literal for int() with base 10: 'HIP version: 7'
```

根因：`hipconfig` 返回的 HIP 版本字符串是 `"HIP version: 7.0.xxx"`，而 aiter 的代码直接用 `split(".")` 分割后尝试 `int("HIP version: 7")`——字符串里的 `"HIP version: "` 前缀导致转换失败。

修复：在 `flash-attention/third_party/aiter/aiter/jit/utils/cpp_extension.py` 第 177 行，把版本解析硬编码为：

```python
ROCM_VERSION = (6, 0)  # 直接写死，绕过 split 解析
```

这不是优雅的修复，但它能让你跳过这个坑继续往前走。

### 4.2 HSA_OVERRIDE_GFX_VERSION 为什么必须设

ROCm 7.1 官方支持 gfx1100，但 RX 7700 XT 实际是 gfx1101。如果不设置 `HSA_OVERRIDE_GFX_VERSION=11.0.1`，ROCm 运行时会拒绝在 gfx1101 上执行。设了这个变量后，运行时会把 gfx1101 当作 gfx1100 来处理——两者架构几乎相同，不会出问题。

### 4.3 flash_attn 走的是 Triton 后端，不是 HIP Kernel

标准的 `pip install flash-attn` 在 gfx1101 上不会编译出 HIP native kernel。它会 fallback 到 Triton 后端。Triton 的性能在大多数场景下够用，但如果你想要 HIP native kernel 的极致性能，需要从 `navi_support` 分支源码编译——目前我还没走到这一步。

### 4.4 PyTorch 版本兼容性

| 组件 | 版本 | 备注 |
|------|------|------|
| ROCm | 7.1 | 不要用 6.x，RDNA3 支持不完整 |
| PyTorch | 2.12.1+rocm7.1 | 从 pytorch.org 的 ROCm index 安装 |
| triton-rocm | 3.7.1 | 随 PyTorch 自动安装 |
| transformers | 4.57.3 | Qwen3-TTS 架构需要 `trust_remote_code=True` |

---

## 5. novel-tts：把小说变成有声书的工具链

优化 Qwen3-TTS 只是手段，目的是能稳定、省心地生成有声书。我把整个流程做成了 CLI 工具 [novel-tts](https://github.com/EarthGraduates/novel-tts)。

### 5.1 7 步工作流

```
init → parse → manifest → [view] → apply → generate → status
  │       │         │          │         │         │
  │       │         │          │         │         └─ 进度统计 + 完成率
  │       │         │          │         └─ 生成 WAV + 断点续传 + ETA + 日志 + 章节拼接
  │       │         │          └─ 应用 manifest 修改到 novel.json
  │       │         ├─ 查看章节详情 + 标记 SPLIT 拆分点
  │       │         └─ 编辑章节操作（K/M/S/X），循环直到满意
  │       └─ 编码检测 → 章节识别 → 句子切分 → 段落分组
  └─ 自动扫描 data/ → 选小说 → 选引擎 → 选音色 → 选风格 → 试听 → 保存
```

核心设计原则：

- **每段生成后立即写盘**：Ctrl+C 不会丢进度
- **JSON 状态 × 磁盘文件对账**：断点续传自动识别已完成段落
- **静默模式**：默认抑制模型加载的冗余日志
- **循环编辑**：manifest 不满意可继续修改，直到确认
- **生成日志（CSV）**：每段的 timestamp / chapter / chars / duration 全部记录
- **单章生成**：`./novel-tts generate 0005` 只生成指定章节，方便测试

### 5.2 双引擎支持

| 引擎 | 类型 | 需要 GPU | 音色 | 推荐场景 |
|------|------|---------|------|---------|
| **Qwen3-TTS 0.6B** | 本地 | ✅ AMD ROCm | 9 个内置音色 + 6 种语气 | 离线可用、音色稳定 |
| **Edge TTS** | 云端 | ❌ 无需 | 5 个微软中文音色 | 零配置、音质最好、不挑字 |

引擎切换只需在 `init` 时选择，后续完全透明。

### 5.3 实战数据：《XX后裔》进行中

- 原始小说经过 format-novel 处理后，novel-tts 拆分为 3638 段
- Qwen3-TTS 0.6B 目前已稳定生成约 1500 段，运行正常
- Edge TTS 引擎也同步测试了另一本小说，生成约 6000 段落无调用限制警告

---

## 6. 总结与建议

### 6.1 AMD 消费级显卡跑 TTS 推理：完全可行

RDNA3 (RX 7000 系列) + ROCm 7.1 + PyTorch 2.12 的组合已经足够稳定。12GB 显存跑 0.6B 模型只占 3.24GB，还有大量余量。

### 6.2 优化有顺序

```
第一步：bfloat16          → 收益最大（-50% 时间，-50% 显存）
第二步：MIOPEN_FIND_MODE=2 → 简单、安全
第三步：SDPA + Autotune    → 必须同时开，单独开反而慢
第四步：torch.compile      → reduce-overhead 模式
不要碰：float16            → 在 RDNA3 上直接崩溃
```

### 6.3 给你的建议

- **只想听书** → Edge TTS，零折腾
- **想折腾 + 离线可用** → Qwen3-TTS 0.6B，跟着本文的优化步骤走
- **AMD 显卡别卖** → ROCm 生态在快速改善，RDNA3 的支持已经不是"理论可行"而是"实际可用"

novel-tts 项目地址：[github.com/EarthGraduates/novel-tts](https://github.com/EarthGraduates/novel-tts)

---

> 写于 2026 年 7 月，基于 AMD RX 7700 XT 实测。如果你也在 AMD 显卡上跑 TTS，欢迎提 Issue 交流踩坑经验。
