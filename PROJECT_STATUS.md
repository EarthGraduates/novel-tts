# novel-tts v1.0

> 2026-07-21

---

## 一、项目概述

novel-tts 是一个小说转有声读物工具，使用 Qwen3-TTS 0.6B CustomVoice 引擎，
在 AMD RX 7700 XT (ROCm 7.1) 上运行。

## 二、当前状态：✅ v1.0 打版

### 已完成

| 模块 | 文件 | 状态 |
|------|------|:--:|
| CLI 入口 | `novel-tts` | ✅ |
| 菜单循环 | `novel-tts` | ✅ init→parse→manifest→apply→generate 自动衔接 |
| 项目初始化 | `init_cmd.py` | ✅ 选小说/音色/风格/试听 |
| 文本解析 | `parse_cmd.py` | ✅ GBK/UTF-8 自动检测，章节/段落拆分 |
| 章节清单 | `manifest_cmd.py` | ✅ 编辑章节合并/拆分/跳过 |
| 章节详情 | `view_cmd.py` | ✅ 查看/标记拆分点 |
| 应用修改 | `apply_cmd.py` | ✅ manifest → novel.json |
| 音频生成 | `generate_cmd.py` | ✅ 断点续传、文件校验、单章过滤 |
| 进度查看 | `status_cmd.py` | ✅ |
| 引擎适配 | `qwen3tts_utils.py` | ✅ Qwen3TTSModel 封装 |
| 配置文件 | `config.py` | ✅ JSON 读写 |

### 性能优化

| 优化 | 效果 | 状态 |
|------|------|:--:|
| MIOPEN_FIND_MODE=2 | ~5x 加速 | ✅ CLI 默认启用 |
| TORCH_BLAS_PREFER_HIPBLASLT=0 | 消除警告 | ✅ CLI 默认启用 |
| flash-attn (Triton AMD) | +33% 加速 | ✅ 已安装（需 patch aiter） |
| torch.compile(reduce-overhead) | +5% 加速 | ✅ 模型加载时自动启用 |
| RTF (实时率) | ~1.6 | 短文本 ~5s 生成 |

### 硬件 & 模型

| 项目 | 详情 |
|------|------|
| GPU | AMD Radeon RX 7700 XT (gfx1100) |
| ROCm | 7.1.52801 (Ubuntu apt) |
| PyTorch | 2.12.1+rocm7.1 |
| 模型 | Qwen3-TTS-12Hz-0.6B-CustomVoice (2.4G) |
| 备用模型 | Qwen3-TTS-12Hz-1.7B-CustomVoice (4.3G) |
| Conda 环境 | `qwen3-tts` (15G) |

### Git

| 项目 | 值 |
|------|-----|
| 仓库 | 本地 git 已初始化 |
| 分支 | master |
| 提交数 | 6 次 |
| .gitignore | ✅ WAV/JSON/TXT/模型排除 |

---

## 三、待完成

### 功能

- [ ] **generate 完整测试** — 生成第一章全部 17 段，验证拼章
- [ ] **质量检查** — 听生成的音频，确认语气/音色满意
- [ ] **速度问题** — 长文本（99字 ~40s），全本书 9385 段需要优化
- [ ] **init 路径记忆** — 多次测试中已有「已配置的小说」列表 ✅

### 性能（可选）

- [ ] SDPA 替代 flash-attn 测试（可能更稳定）
- [ ] INT8 量化（2.5x 加速，需验证音质）
- [ ] 批量推理（`generate_batch`）

### 工程

- [ ] GitHub 仓库创建 & push
- [ ] `pyproject.toml` / `setup.py`（pip install 支持）
- [ ] 单元测试

---

## 四、数据统计

| 指标 | 值 |
|------|-----|
| 测试小说 | 大力金刚掌-茅山后裔 |
| 编码 | GBK |
| 章节 | 362 章 |
| 句子 | 40,518 句 |
| 段落 | 9,385 段 |
| 音频格式 | 24kHz WAV |
| 已生成（F5-TTS 旧） | 17 段 |

---

## 五、关键决策记录

1. **选 Qwen3-TTS 而非 F5-TTS** — 不需要参考音频，内置音色，语气可控
2. **选 0.6B 而非 1.7B** — 速度接近但显存更小，音质差异不大
3. **删掉 ChatTTS** — 效果差，环境已清理
4. **保留 F5-TTS 代码** — `f5tts_utils.py` 未删除，但不再调用
5. **ROCm 用 Ubuntu apt 安装** — 路径散落在 `/usr`，非 `/opt/rocm`
