# novel-tts

> 小说转有声读物工具 —— 在 AMD 消费级显卡上运行 Qwen3-TTS

## 快速开始

```bash
# 1. 初始化项目（选小说 + 选音色 + 试听）
./novel-tts init

# 2. 解析小说 → 自动检测章节/段落
./novel-tts parse

# 3. 编辑章节清单（K保留 / M合并 / S拆分 / X跳过）
./novel-tts manifest

# 4. 应用修改
./novel-tts apply

# 5. 生成音频（支持断点续传）
./novel-tts generate

# 6. 查看进度
./novel-tts status
```

6 步走完，有声书在 `output/<书名>/` 下。

---

## 工作流

```
init → parse → manifest → apply → generate → status
  │       │         │          │         │
  │       │         │          │         └─ 生成 WAV + 断点续传 + ETA + 日志
  │       │         │          └─ 应用 manifest 修改到 novel.json
  │       │         └─ 编辑章节操作（K/M/S/X）+ 标记拆分点
  │       └─ 编码检测 → 章节识别 → 句子切分 → 段落分组
  └─ 选小说 TXT → 选音色 → 选风格 → 试听 → 保存配置
```

---

## 硬件 & 环境

| 项目 | 详情 |
|---|---|
| GPU | AMD Radeon RX 7700 XT (gfx1101, RDNA3), 12 GB VRAM |
| ROCm | 7.1.52802 |
| PyTorch | 2.12.1+rocm7.1 |
| Python | 3.10.20 |
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

# 安装依赖
pip install soundfile numpy transformers accelerate

# 安装 flash-attention (Triton 后端, RDNA3)
pip install flash-attn triton
```

---

## 性能优化

在 AMD RX 7700 XT (RDNA3) 上的优化历程。测试文本：3 段中文（12 / 50 / 135 字）。

### 优化前 → 优化后

| 指标 | 优化前 | 优化后 | 提升 |
|---|---|---|---|
| **推理速度 (RTF)** | 1.59 | **0.76** | **2.1×** |
| **135 字生成** | ~60s | **~27s** | **-55%** |
| **显存占用** | 6.45 GB | **3.24 GB** | **-50%** |
| **50 字生成** | ~28s | **~11s** | **-60%** |
| SDPA Warnings | 每次 2 个 | **0** | |

### 优化措施

| # | 措施 | 效果 | 风险 |
|---|---|---|---|
| 1 | `torch_dtype=torch.bfloat16` | **-54% 时间, -50% 显存** | 低 — bf16 在 RDNA3 有硬件支持，音频质量正常 |
| 2 | `FLASH_ATTENTION_TRITON_AMD_AUTOTUNE=TRUE` | **-7% 时间**（配合 #3 使用） | 低 — 首次运行 Triton autotune warmup |
| 3 | `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` | 消除 SDPA experimental warning，配合 #2 才有正向收益 | 低 |
| 4 | `MIOPEN_FIND_MODE=2` | ~5× 卷积加速 | 低 |
| 5 | `torch.compile(mode="reduce-overhead")` | ~5% 加速 | 低 — 首次触发 JIT 编译 |

### 关键发现

1. **bfloat16 是决定性优化**。float16 在 gfx1101 上直接 `HSA_STATUS_ERROR_EXCEPTION`（硬件异常），不可用。bf16 既快又稳。
2. **单独开 SDPA experimental 反而慢 7%**。必须搭配 Triton Autotune 才能拿到正向收益。
3. **flash_attn 走 Triton 后端**（非 HIP native kernel）。标准 pip 安装的 flash_attn 2.8.4 在 gfx1101 上没有编译出 HIP kernel，fallback 到 Triton。Autotune 能从 Triton 路径中榨出余量。

### 环境变量（已内置到 `novel-tts` 和 `qwen3tts_utils.py`）

```python
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.1")
os.environ.setdefault("MIOPEN_FIND_MODE", "2")
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_AUTOTUNE", "TRUE")
```

---

## benchmark 数据

### bf16 vs float32 vs float16

`bench_dtype.py` — 2026-07-21

| dtype | bench_total | RTF | VRAM | 音频正常 |
|---|---|---|---|---|
| float32 | 87.9s | 1.59 | 6.45 GB | ✓ |
| float16 | 💥 崩溃 | — | — | ✗ HSA 异常 |
| **bfloat16** | **40.4s** | **0.76** | **3.24 GB** | ✓ |

### SDPA Experimental + Autotune

`bench_sdpa_experimental.py` — 2026-07-21

| 配置 | bench_total | RTF | SDPA Warnings |
|---|---|---|---|
| baseline | 90.0s | 1.59 | 2 |
| SDPA-experimental | 96.1s | 1.59 | 0 |
| **SDPA-exp + Autotune** | **83.4s** | **1.60** | **0** |

---

## 解析规则

### 章节识别

```
正则: ^第[零一二三四五六七八九十百千\d]+[章回节集]
锚定行首（避免"详见第一章"误判）
标题长度 ≤ 50 字
```

### 段落切分

```
按空行切段 → 每个段落内部按 。！？… 切句
→ 组装回段落，确保每段 ≤ 500 字
→ 超出则在中点句子边界拆分
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
├── qwen3tts_utils.py             # Qwen3-TTS 模型封装 + ROCm 优化
└── commands/
    ├── init_cmd.py                # 项目初始化
    ├── parse_cmd.py               # 文本解析
    ├── manifest_cmd.py            # 章节清单编辑
    ├── view_cmd.py                # 章节详情 + 标记拆分点
    ├── apply_cmd.py               # 应用 manifest 修改
    ├── generate_cmd.py            # 音频生成 + 断点续传 + 日志
    └── status_cmd.py              # 进度查看
bench_sdpa_experimental.py         # SDPA/Autotune benchmark
bench_dtype.py                     # dtype 对比 benchmark
bench_compile.py                   # torch.compile 对比 benchmark
bench_miopen.py                    # MIOPEN_FIND_MODE benchmark
novels/
├── <书名>_config.json             # 项目配置（书路径 + 音色）
├── <书名>_novel.json              # 解析结果（目录树 + 句子列表）
├── <书名>_manifest.txt            # 章节操作清单
└── output/<书名>/
    ├── generate.log               # 生成日志（CSV）
    ├── <章节名>.wav               # 章节拼接音频（以章节标题命名）
    └── <章节ID>/
        └── p_<ID>_<order>.wav     # 段落音频
```

---

## 已知限制

- **不支持实时流式朗读** — Qwen3-TTS SDK 无 streaming API，只能一次性生成
- **Windows 未测试** — ROCm 环境仅在 Ubuntu 22.04 上验证
- **仅 RX 7700 XT (gfx1101) 验证** — 其他 RDNA3 卡（7900 XT/XTX）理论上兼容，未实测
- **flash_attn 无 HIP native kernel** — 当前走 Triton 后端，HIP kernel 需源码编译 navi_support 分支

---

## License

Apache-2.0 (Qwen3-TTS 模型许可)
