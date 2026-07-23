# novel-tts

> 小说转有声读物工具 —— 支持 AMD 消费级显卡本地运行 + 微软 Edge TTS 云端引擎

> **📖 在 AMD 显卡上部署 TTS 的完整踩坑记录？** 见 [《AMD 消费级显卡部署 TTS 模型实战》](docs/amd-consumer-gpu-tts-deployment.md) —— 从 ChatTTS、F5-TTS、Qwen3-TTS 到 Edge TTS 的引擎横评，以及 Qwen3-TTS 0.6B 在 RX 7700 XT 上从 RTF 1.59 优化到 0.76 的全过程。

---

## 前置步骤：格式化小说

原始网络小说 TXT 文件含广告、乱码、编码混乱、硬换行等"版面杂质"，直接喂给 novel-tts 会导致章节识别错误。

使用 **format-novel** 技能（`docs/SKILL.md`）先清洗排版：

```
/format-novel /path/to/原始小说.txt
```

输出为符合 novel-tts 解析规范的标准干净文本，放到 `data/` 目录下。初始化时 `novel-tts init` 会自动扫描 `data/` 目录下的 `.txt` 文件供选择。

---

## 快速开始

```bash
# 1. 初始化项目（自动扫描 data/ 目录 → 选小说 → 选引擎 → 选音色 → 试听）
./novel-tts init

# 2. 解析小说 → 自动检测章节/段落
./novel-tts parse

# 3. 编辑章节清单（K保留 / M合并 / S拆分 / X跳过）
#    不满意可循环编辑，直到确认满意
./novel-tts manifest

# 4. 查看章节详情（可选 — 标记 SPLIT 拆分点）
./novel-tts view 0001

# 5. 应用修改
./novel-tts apply

# 6. 生成音频（生成前确认进度 → 断点续传 → ETA → 日志）
./novel-tts generate

# 7. 查看进度
./novel-tts status
```

6 步走完，有声书在 `output/<书名>/` 下。

---

## 双引擎

novel-tts 支持两个 TTS 引擎，`init` 时选择：

| 引擎 | 类型 | 需要 GPU | 音色 | 推荐场景 |
|------|------|---------|------|---------|
| **Qwen3-TTS 0.6B** | 本地 | ✅ AMD ROCm | 9 个内置音色 + 语气控制 | 离线可用、音色稳定 |
| **Edge TTS** | 云端 | ❌ 无需 | 5 个微软中文音色 | 零配置、音质最好 |

引擎切换只需重新 `init` 选择，后续工作流完全透明。

### Qwen3-TTS（本地）

- 阿里通义千问团队，0.6B 模型，效果够用
- 9 个内置音色：Vivian, Serena, Uncle_Fu, Ryan, Aiden, Eric, Dylan, Ono_Anna, Sohee
- 6 种语气指令：旁白 / 温柔 / 愤怒 / 欢快 / 悲伤 / 严肃
- 支持音色克隆（传参考音频）
- 首次加载约 14s（含 torch.compile warmup），后续推理走编译缓存

### Edge TTS（云端）

- 微软免费 TTS 服务，无需 GPU
- 5 个中文音色：晓晓、云希、云健、晓伊、云扬
- 7 种风格：通用 / 欢快 / 悲伤 / 愤怒 / 温柔 / 严肃 / 旁白
- 音质好、什么字都能读、不挑硬件
- 实测百万字小说未触发调用限制

---

## 工作流

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

### 关键设计

- **断点续传**：每段生成后立即写盘（JSON + WAV），Ctrl+C 不丢进度。重新运行时自动对账 JSON 状态与磁盘文件。
- **静默模式**：默认抑制模型加载的冗余日志，设 `NOVEL_TTS_VERBOSE=1` 可恢复。
- **循环编辑**：manifest 编辑后不满意可继续修改，直到确认。
- **单章生成**：`./novel-tts generate 0005` 只生成指定章节。
- **自动衔接**：交互菜单中每步完成后按 Enter 进入下一步。

---

## 硬件 & 环境

| 项目 | 详情 |
|------|------|
| GPU | AMD Radeon RX 7700 XT (gfx1101, RDNA3), 12 GB VRAM |
| ROCm | 7.1.52802 |
| PyTorch | 2.12.1+rocm7.1 |
| Python | 3.10.20 |
| OS | Ubuntu 26.04 |
| 模型 | Qwen3-TTS-12Hz-0.6B-CustomVoice |

### 安装

```bash
# 创建 conda 环境
conda create -n qwen3-tts python=3.10 -y
conda activate qwen3-tts

# 安装 PyTorch (ROCm)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm7.0

# 安装 Qwen3-TTS
pip install qwen-tts

# 安装 Edge TTS（云端引擎）
pip install edge-tts

# 安装项目依赖
pip install soundfile numpy transformers accelerate

# 安装 flash-attention（Triton 后端, RDNA3）
pip install flash-attn triton
```

> **注意**：如果 flash-attention 编译失败（aiter HIP 版本解析 Bug），见下方 [ROCm 踩坑](#rocm-踩坑)。

---

## 性能优化

在 AMD RX 7700 XT (RDNA3) 上的优化历程。测试文本：3 段中文（12 / 50 / 135 字）。

### 优化前 → 优化后

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| **推理速度 (RTF)** | 1.59 | **0.76** | **2.1×** |
| **135 字生成** | ~60s | **~27s** | **-55%** |
| **显存占用** | 6.45 GB | **3.24 GB** | **-50%** |
| **50 字生成** | ~28s | **~11s** | **-60%** |
| SDPA Warnings | 每次 2 个 | **0** | |

### 优化措施

| # | 措施 | 效果 | 风险 |
|---|------|------|------|
| 1 | `torch_dtype=torch.bfloat16` | **-54% 时间, -50% 显存** | 低 — bf16 在 RDNA3 有硬件支持，音频质量正常 |
| 2 | `FLASH_ATTENTION_TRITON_AMD_AUTOTUNE=TRUE` | **-7% 时间**（配合 #3 使用） | 低 — 首次运行 Triton autotune warmup |
| 3 | `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` | 消除 SDPA experimental warning，配合 #2 才有正向收益 | 低 |
| 4 | `MIOPEN_FIND_MODE=2` | ~5× 卷积加速 | 低 |
| 5 | `torch.compile(mode="reduce-overhead")` | ~5% 加速 | 低 — 首次触发 JIT 编译 |

### 关键发现

1. **bfloat16 是决定性优化**。float16 在 gfx1101 上直接 `HSA_STATUS_ERROR_EXCEPTION`（硬件异常），不可用。bf16 既快又稳。
2. **单独开 SDPA experimental 反而慢 7%**。必须搭配 Triton Autotune 才能拿到正向收益。
3. **flash_attn 走 Triton 后端**（非 HIP native kernel）。标准 pip 安装的 flash_attn 2.8.4 在 gfx1101 上没有编译出 HIP kernel，fallback 到 Triton。Autotune 能从 Triton 路径中榨出余量。

### ROCm 踩坑

#### flash-attention 编译失败

AMD 的 aiter 子模块解析 `hipconfig` 输出的 HIP 版本字符串时出错：

```
ValueError: invalid literal for int() with base 10: 'HIP version: 7'
```

修复：编辑 `flash-attention/third_party/aiter/aiter/jit/utils/cpp_extension.py` 第 177 行，硬编码：

```python
ROCM_VERSION = (6, 0)  # 绕过 split 解析
```

#### HSA_OVERRIDE_GFX_VERSION

RX 7700 XT 是 gfx1101，ROCm 7.1 官方支持 gfx1100。需设置 `HSA_OVERRIDE_GFX_VERSION=11.0.1` 让运行时兼容处理。已内置到引擎初始化代码中。

---

## benchmark 数据

### bf16 vs float32 vs float16

`benchmarks/bench_dtype.py` — 2026-07-21

| dtype | bench_total | RTF | VRAM | 音频正常 |
|------|-------------|-----|------|---------|
| float32 | 87.9s | 1.59 | 6.45 GB | ✓ |
| float16 | 💥 崩溃 | — | — | ✗ HSA 异常 |
| **bfloat16** | **40.4s** | **0.76** | **3.24 GB** | ✓ |

### SDPA Experimental + Autotune

`benchmarks/bench_sdpa_experimental.py` — 2026-07-21

| 配置 | bench_total | RTF | SDPA Warnings |
|------|-------------|-----|---------------|
| baseline | 90.0s | 1.59 | 2 |
| SDPA-experimental | 96.1s | 1.59 | 0 |
| **SDPA-exp + Autotune** | **83.4s** | **1.60** | **0** |

### torch.compile 模式对比

`benchmarks/bench_compile.py` — 2026-07-21

| 模式 | 耗时 | 加速比 |
|------|------|--------|
| eager (no compile) | 16.9s | baseline |
| **reduce-overhead** | **15.2s** | **1.11×** |
| max-autotune | 16.1s | 1.05× |

---

## 解析规则

### 章节识别

```
正则: ^第[零一二三四五六七八九十百千\d]+[章回节集]
锚定行首（避免"详见第一章"误判）
标题长度 ≤ 50 字
3 行内重复标题自动去重
```

### 段落切分

```
按空行切段 → 每个段落内部按 。！？… 切句
→ 组装回段落，确保每段 ≤ 500 字
→ 超出则在中点句子边界拆分
→ 引号内句子不拆分（保持对话完整）
```

### 编码检测

```
UTF-8 → GBK → GB18030 → 兜底 UTF-8
```

---

## 文件结构

```
novel-tts                          # CLI 入口脚本
tts_core/
├── config.py                      # JSON 配置读写
├── parser.py                      # TXT → novel.json 解析引擎
├── qwen3tts_utils.py             # Qwen3-TTS 模型封装（兼容层）
├── engines/                       # TTS 引擎模块
│   ├── __init__.py                # 引擎注册表 + 统一接口
│   ├── qwen3_engine.py            # Qwen3-TTS（本地 GPU）
│   └── edge_engine.py             # Edge TTS（云端）
└── commands/
    ├── init_cmd.py                # 项目初始化（自动扫描 data/，双引擎选择）
    ├── parse_cmd.py               # 文本解析（引号平衡 + 鲁棒章节检测）
    ├── manifest_cmd.py            # 章节清单编辑（循环直到满意）
    ├── view_cmd.py                # 章节详情 + 标记 SPLIT 拆分点
    ├── apply_cmd.py               # 应用 manifest 修改（含 SPLIT）
    ├── generate_cmd.py            # 音频生成（静默模式 + 确认步骤 + 断点续传 + 日志）
    └── status_cmd.py              # 进度查看
benchmarks/                        # 性能基准脚本
├── bench_dtype.py                 # dtype 对比（float32/float16/bfloat16）
├── bench_dtype_worker.py
├── bench_sdpa_experimental.py     # SDPA/Autotune 对比
├── bench_sdpa_worker.py
├── bench_compile.py               # torch.compile 模式对比
└── bench_miopen.py                # MIOPEN_FIND_MODE 基准
data/                              # 格式化后的小说 TXT（init 自动扫描）
docs/                              # 文档
├── SKILL.md                       # format-novel 技能说明
├── AI_FORMAT_PROMPT.md
├── MODEL_INVENTORY.md
└── amd-consumer-gpu-tts-deployment.md  # AMD 部署实战文章
novels/
├── <书名>_config.json             # 项目配置（书路径 + 引擎 + 音色）
├── <书名>_novel.json              # 解析结果（目录树 + 句子列表）
├── <书名>_manifest.txt            # 章节操作清单
├── <书名>_manifest_<ID>.txt       # 章节 SPLIT 标记
└── output/<书名>/
    ├── generate.log               # 生成日志（CSV）
    ├── <章节标题>.wav             # 章节拼接音频
    └── <章节ID>/
        └── p_<ID>_<order>.wav     # 段落音频
```

---

## 已知限制

- **不支持实时流式朗读** — Qwen3-TTS SDK 无 streaming API，只能一次性生成
- **Windows 未测试** — ROCm 环境仅在 Ubuntu 26.04 上验证
- **仅 RX 7700 XT (gfx1101) 验证** — 其他 RDNA3 卡（7900 XT/XTX）理论上兼容，未实测
- **flash_attn 无 HIP native kernel** — 当前走 Triton 后端，HIP kernel 需源码编译 navi_support 分支
- **Qwen3-TTS 部分汉字不读** — 敏感或粗俗字眼会被跳过

---

## License

Apache-2.0 (Qwen3-TTS 模型许可)
