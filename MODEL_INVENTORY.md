# TTS 项目模型与依赖清单

> 更新时间: 2026-07-21

---

## 一、硬件环境

| 项目 | 详情 |
|------|------|
| GPU | AMD Radeon RX 7700 XT (RDNA3, gfx1100) |
| ROCm | 7.1 (PyTorch 2.12.1+rocm7.1) |
| Python | 3.10.20 |
| OS | Ubuntu 22.04 (Linux 7.0.0-27-generic) |

---

## 二、Conda 环境

### 1. `qwen3-tts`（Qwen3-TTS 主环境）

| 包名 | 版本 | 说明 |
|------|------|------|
| qwen-tts | 0.1.1 | Qwen3-TTS 官方包（阿里通义千问团队） |
| torch | 2.12.1+rocm7.1 | PyTorch ROCm 版 |
| torchaudio | 2.11.0+rocm7.1 | |
| transformers | 4.57.3 | ⚠️ qwen3_tts 架构未注册，需要 trust_remote_code |
| tokenizers | 0.22.2 | |
| accelerate | 1.12.0 | |
| triton-rocm | 3.7.1 | |
| chattts | 0.2.5 | 旧依赖（计划删除） |
| f5-tts | 1.1.21 | 旧依赖 |
| ema-pytorch | 0.7.9 | |
| soundfile | (已安装) | |
| einops | (已安装) | |
| gradio | (已安装) | |
| librosa | (已安装) | |
| onnxruntime | (已安装) | |
| sox | (已安装) | |

⚠️ **flash-attn 未安装** —— 先尝试 PyTorch 手动版推理，能跑再决定是否编译

### 2. `chattts`（旧环境，计划清理）

| 包名 | 版本 |
|------|------|
| chattts | 0.2.5 |
| f5-tts | 1.1.21 |
| torch | 2.12.1+rocm7.1 |
| transformers | 5.13.1 |
| qwen-tts | ❌ 未安装 |

### 3. `base`（系统默认）

- miniconda3 基础环境，无 TTS 相关包

---

## 三、HuggingFace 模型缓存

> 缓存路径: `~/.cache/huggingface/hub/`

### 1. Qwen3-TTS-12Hz-0.6B-CustomVoice 🟡

| 字段 | 值 |
|------|-----|
| 路径 | `models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice` |
| 磁盘占用 | 656 MB |
| 快照 | `85e237c12c027371202489a0ec509ded67b5e4b5` |

**已缓存文件:**

| 文件 | 状态 |
|------|------|
| config.json | ✅ |
| generation_config.json | ✅ |
| tokenizer_config.json | ✅ |
| preprocessor_config.json | ✅ |
| vocab.json | ✅ |
| merges.txt | ✅ |
| speech_tokenizer/config.json | ✅ |
| speech_tokenizer/configuration.json | ✅ |
| speech_tokenizer/preprocessor_config.json | ✅ |
| speech_tokenizer/model.safetensors (682MB) | ✅ |
| **model.safetensors（主模型权重）** | ❌ **缺失！** |

**支持的功能:**
- Speaker: Vivian, Serena, Uncle_Fu, Ryan, Aiden, Ono_Anna, Sohee, Eric, Dylan
- 语言: 中文/英文/日/韩/德/法/意/西/葡/俄 + 北京话/四川话
- Eric → 四川话, Dylan → 北京话

### 2. SWivid/F5-TTS ✅

| 字段 | 值 |
|------|-----|
| 路径 | `models--SWivid--F5-TTS` |
| 磁盘占用 | 1.3 GB |
| 状态 | **完整** |

### 3. openai/whisper-large-v3-turbo ✅

| 字段 | 值 |
|------|-----|
| 路径 | `models--openai--whisper-large-v3-turbo` |
| 磁盘占用 | 1.6 GB |
| 状态 | **完整**（F5-TTS 的参考音频转写依赖） |

### 4. charactr/vocos-mel-24khz ✅

| 字段 | 值 |
|------|-----|
| 路径 | `models--charactr--vocos-mel-24khz` |
| 磁盘占用 | 52 MB |
| 状态 | **完整**（F5-TTS 声码器） |

---

## 四、项目文件结构

```
/home/phoenix/ClaudeProjects/TTS/
├── tts_core/                          # TTS CLI 核心代码
│   ├── __init__.py
│   ├── config.py                      # 配置读写（novel_config.json）
│   ├── parser.py                      # 命令行参数解析
│   ├── chattts_utils.py               # ChatTTS 接口（旧）
│   ├── f5tts_utils.py                 # F5-TTS 接口（当前）
│   └── commands/
│       ├── __init__.py
│       ├── init_cmd.py                # 初始化小说项目
│       ├── parse_cmd.py               # TXT → novel.json 解析
│       ├── manifest_cmd.py            # 生成 manifest
│       ├── generate_cmd.py            # 生成音频（当前调用 F5-TTS）
│       ├── apply_cmd.py               # 应用修订到 novel.json
│       ├── status_cmd.py              # 状态查询
│       └── view_cmd.py                # 查看内容
├── gen_f5_chapter.py                  # F5-TTS 独立测试脚本
├── test_f5_tts.py                     # F5-TTS 测试
├── 大力金刚掌-茅山后裔.txt             # 小说源文件
├── novels/
│   ├── 大力金刚掌-茅山后裔_config.json
│   ├── 大力金刚掌-茅山后裔_novel.json
│   ├── 大力金刚掌-茅山后裔_manifest.txt
│   ├── 大力金刚掌-茅山后裔_manifest_0002.txt
│   ├── tmp/
│   │   ├── preview_preset_1.wav
│   │   └── preview_preset_3.wav
│   └── output/                        # 生成的音频输出
│       ├── f5_tts_ch01_p01.wav            # F5-TTS 第一章第一段
│       ├── f5_tts_ch01_p01_v2.wav         # F5-TTS v2
│       ├── f5_tts_cli_test.wav            # F5-TTS CLI 测试
│       ├── qwen3_0.6b_angry.wav           # Qwen3 0.6B 愤怒语气
│       ├── qwen3_0.6b_gentle.wav          # Qwen3 0.6B 温柔语气
│       ├── qwen3_0.6b_narration.wav       # Qwen3 0.6B 旁白
│       ├── qwen3_0.6b_customvoice.wav     # Qwen3 0.6B CustomVoice (0字节-失败)
│       ├── qwen3_1.7b_chapter.wav         # Qwen3 1.7B 章节
│       └── qwen3_1.7b_narration.wav       # Qwen3 1.7B 旁白
├── f5_tts_test_hello.wav
├── f5_tts_test_weather.wav
└── tests/
    └── infer_cli_basic.wav
```

---

## 五、当前问题

### 🔴 Qwen3-TTS 0.6B 模型不完整
- 主模型权重 `model.safetensors` 未下载（snapshot 中缺失）
- 只有 `speech_tokenizer/model.safetensors`（682MB）完整
- 原因：7/19 下载中断，残留 `.incomplete` 文件和 `.no_exist` 标记
- 需要重新下载 `model.safetensors`

### 🟡 flash-attn 未编译
- `/tmp/flash-attention/` 源码在重启后丢失
- 但 Qwen3-TTS 可以降级到 PyTorch 手动版运行（仅影响推理速度）
- 如果 0.6B 推理速度可接受，可以暂不编译

### 🟢 F5-TTS 环境完整可用
- 模型缓存完整，CLI 可正常使用

### 🔵 Qwen3-TTS 1.7B
- 之前生成过样例（`qwen3_1.7b_chapter.wav`、`qwen3_1.7b_narration.wav`）
- 但无本地缓存（可能被清理或从未持久化）
- 需要确认当时用的是哪个具体模型名

---

## 六、下一步

1. 修复 0.6B 模型缓存 → 重下载 `model.safetensors`
2. 测试 Qwen3-TTS 推理是否正常（无 flash-attn）
3. 如推理速度可接受 → 跳过 flash-attn 编译
4. 将 TTS CLI 从 F5-TTS 切换到 Qwen3-TTS
