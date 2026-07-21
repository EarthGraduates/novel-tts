# novel-tts v1.0 PRD

> 产品需求文档 · 2026-07-21

---

## 一、产品概述

### 1.1 产品定位

novel-tts 是一个**命令行小说转有声读物工具**，让用户在本地 AMD 消费级显卡上，将中文小说 TXT 自动转为分章节的有声书。

### 1.2 目标用户

三层画像，递进覆盖：

| 层级 | 用户 | 需求 | 优先级 |
|---|---|---|---|
| **A** | 我自己 | 把喜欢的小说转成有声书，自己听 | P0 |
| **B** | 同好看书人 | 有同样需求的中文读者，需要易上手 | P1 |
| **C** | 开发者/极客 | 有 ROCm/AMD GPU，想知道 Qwen3-TTS 怎么在 RDNA3 上跑通 | P2 |

- A 是当下实际用户，驱动功能决策
- B 决定产品体验标准（界面友好、文档清晰、安装门槛低）
- C 决定技术文档的深度（README 中的性能优化记录、benchmark 数据）

### 1.3 核心价值

> **拿来一本 TXT，走完 6 步，拿到有声书。**

不需要 GPU 服务器、不需要参考音频、不需要手动配置角色音色——Qwen3-TTS 内置 9 个中文音色，指令式语气控制（旁白/温柔/愤怒/欢快……）。

---

## 二、v1.0 目标与范围

### 2.1 完成定义

- [x] 完整 6 步工作流（init → parse → manifest → apply → generate → status）
- [x] 在 AMD RX 7700 XT 上稳定运行，推理速度 RTF ≤ 0.8
- [x] 跑通一本完整小说（《茅山后裔》，362 章 / 9,385 段）
- [x] README 包含安装指南、使用说明、性能对比数据、ROCm 踩坑记录
- [x] CLI 交互友好（菜单、自动衔接、操作提示）

### 2.2 不在 v1.0 范围

- Web UI / GUI
- 批量推理（一次生成多段）
- INT8 量化
- Windows 支持
- 自动化测试套件
- HIP native kernel flash-attention（当前 Triton 后端够用）

---

## 三、用户故事

### 主线一：项目初始化（一次性操作）

**故事**: 用户拿来一本小说 TXT，完成初始化。

```
1. 运行 ./novel-tts init
2. 选择小说 TXT 文件（或复用已有项目）
3. 选择朗读音色（9 个内置 speaker，标注性别）
4. 选择朗读风格（旁白/温柔/愤怒/欢快/悲伤/严肃）
5. 试听
6. 满意 → 保存配置
```

**规则**:
- 编码自动检测: UTF-8 → GBK → GB18030
- 章节识别: 行首 + `第X章/第X回` + 标题长度 ≤ 50 字
- 段落切分: 按空行切段 → 按 `。！？…` 切句 → 组装 ≤ 500 字
- manifest 编辑: 默认 nano 编辑器（`$EDITOR` 可覆盖）

**验收标准**:
- [ ] 能正确解析《茅山后裔》GBK 编码，362 章全部识别
- [ ] 不会把正文中的 "详见第一章" 误判为章节标题
- [ ] 段落按自然段切分，每段 ≤ 500 字
- [ ] manifest 在 nano 中正确显示 K / M / S / X 操作说明

---

### 主线二：音频生成（重复操作）

**故事**: 用户运行生成，支持断点续传，最终拿到完整有声书。

```
1. 运行 ./novel-tts generate
2. 自动对账：JSON 状态 vs 磁盘文件 → 识别已完成/待生成
3. 如有已完成，提示 [1] 断点续传 [2] 重新生成
4. 逐段生成 + 实时显示进度 + ETA
5. 每章完成后自动拼接 chapter.wav
6. 所有错误段自动重试一次，仍失败标记 failed
```

**规则**:
- 每次生成完一段立即 `save_novel()` + 写日志 → Ctrl+C 安全
- 日志文件: `novels/output/<书名>/generate.log`（CSV）
- ETA: 取最近 10 段平均耗时 × 剩余段数
- 断点续传: `os.walk` 扫描已有 WAV + JSON status 对账

**验收标准**:
- [ ] 生成一段后 Ctrl+C，重新运行能正确续传
- [ ] ETA 在生成 10 段后开始显示，误差在 ±30% 以内
- [ ] 日志记录每段的 timestamp / chapter / order / chars / duration / status
- [ ] 完整一章生成的段落 WAV 能正确拼接为 chapter.wav

---

### 主线三：ROCm/AMD 部署（技术故事）

**故事**: 开发者在 AMD RX 7000 系列显卡上部署 Qwen3-TTS，需要知道坑和优化方案。

**覆盖内容**（见 README）:
- 环境安装步骤（ROCm 7.1 + PyTorch 2.12 + flash-attn）
- bf16 vs float16 vs float32 对比数据
- SDPA experimental + Triton Autotune 的配合关系
- 单独开 SDPA experimental 反而慢 7% 的教训
- 各优化措施的累计效果

---

## 四、核心流程

```
┌──────┐    ┌───────┐    ┌──────────┐    ┌───────┐    ┌──────────┐    ┌────────┐
│ init │───▶│ parse │───▶│ manifest │───▶│ apply │───▶│ generate │───▶│ status │
└──────┘    └───────┘    └──────────┘    └───────┘    └──────────┘    └────────┘
    │            │             │               │             │              │
    ▼            ▼             ▼               ▼             ▼              ▼
 选小说      编码检测      章节操作       应用修改      生成音频        查看进度
 选音色      章节识别      K/M/S/X       重新编号      断点续传        完成统计
 试听        句子切分      编辑器         重建目录      章节拼接
 保存        段落分组                                  生成日志
                                                        ETA
```

---

## 五、数据模型

### novel.json（核心数据结构）

```json
{
  "novel_title": "茅山后裔",
  "source_file": "/path/to/novel.txt",
  "encoding": "gbk",
  "voice_model": "single",
  "voice_profile": {
    "engine": "qwen3-tts",
    "speaker": "Uncle_Fu",
    "instruct": "narration",
    "language": "chinese"
  },
  "toc": [{
    "volume": 1,
    "title": "正文",
    "parts": [{
      "part": 1,
      "chapters": [{
        "id": "0001",
        "title": "第一章 楔子",
        "status": "pending",
        "audio_path": "",
        "sentence_range": [1, 150],
        "paragraphs": [[2, 5], [6, 10], ...]
      }]
    }]
  }],
  "sentences": [{
    "id": "0001-000000",
    "order": 1,
    "chapter_id": "0001",
    "type": "chapter_title",
    "text": "第一章 楔子",
    "status": "pending",
    "audio_path": ""
  }]
}
```

### 关键规则

- **段落边界**: 每个段落的**首句**携带 `status` / `audio_path` 字段，同段落其他句子无此字段
- **段落识别**: `sentences` 中 `"status" in s` 的句子即为段落首句
- **章节拼接**: 一章所有段落 status=done 且文件存在 → `concat → chapter.wav`

---

## 六、功能清单

### 6.1 init — 项目初始化

| 功能 | 状态 |
|---|---|
| 选择小说 TXT 文件 | ✅ |
| 记住已配置的小说列表 | ✅ |
| 选择朗读音色（9 个内置 speaker，含性别标注） | ✅ |
| 选择朗读风格（旁白/温柔/愤怒/欢快/悲伤/严肃） | ✅ |
| 生成试听音频 | ✅ |
| 满意后保存配置 | ✅ |
| 覆盖已有配置确认 | ✅ |

### 6.2 parse — 文本解析

| 功能 | 状态 |
|---|---|
| 编码自动检测（UTF-8 → GBK → GB18030） | ✅ |
| 章节标题识别（`^第X章` + ≤50 字 + 非 front_matter） | ✅ |
| 重复章节标题过滤（3 行内去重） | ✅ |
| front_matter 识别（作品相关/人物表/序/前言） | ✅ |
| 卷/部与章节区分 | ✅ |
| 句子切分（`。！？…` 主切 + `，；：` 超长辅助切） | ✅ |
| 段落检测（空行分割，≤500 字自动拆分） | ✅ |
| 输出 novel.json + manifest.txt | ✅ |
| CLI 显示解析规则 | ✅ |
| 已有 novel.json 可选择沿用或重新解析 | ✅ |

### 6.3 manifest — 章节清单

| 功能 | 状态 |
|---|---|
| 打开 manifest.txt 供编辑（默认 nano） | ✅ |
| 操作类型: K(keep) / M(merge) / S(split) / X(skip) | ✅ |
| 显示操作说明 | ✅ |

### 6.4 view — 章节详情

| 功能 | 状态 |
|---|---|
| 查看/编辑章节详情 | ✅ |
| 自动换行长文本（~80 字） | ✅ |
| 插入 `>>> SPLIT <<<` 标记拆分点 | ✅ |
| 显示拆分行提示 | ✅ |

### 6.5 apply — 应用修改

| 功能 | 状态 |
|---|---|
| 解析 manifest.txt 操作码 | ✅ |
| 解析 manifest_<ID>.txt SPLIT 标记 | ✅ |
| 应用 merge（合并章节） | ✅ |
| 应用 split（拆分章节） | ✅ |
| 应用 skip（标记为 front_matter） | ✅ |
| 自动重新编号章节/句子 | ✅ |
| 重新生成 manifest.txt | ✅ |
| 操作前备份 novel.json + manifest.txt | ✅ |

### 6.6 generate — 音频生成

| 功能 | 状态 |
|---|---|
| 文件对账（JSON status × 磁盘 WAV） | ✅ |
| 断点续传（自动或手动重置） | ✅ |
| 单章过滤（`generate 0005` 只生成第五章） | ✅ |
| 每段完成后立即写 status + save_novel | ✅ |
| 生成日志（CSV: timestamp/chapter/order/chars/duration/status） | ✅ |
| ETA 估算（最近 10 段平均速度） | ✅ |
| 失败自动重试一次 | ✅ |
| 错误段汇总 + 二次重试 | ✅ |
| 章节拼接（段落 done → chapter.wav） | ✅ |
| torch.compile(reduce-overhead) | ✅ |
| bfloat16 推理 | ✅ |
| Triton Autotune | ✅ |

### 6.7 status — 进度查看

| 功能 | 状态 |
|---|---|
| 段落完成统计 + 百分比 | ✅ |
| 章节拼接统计 | ✅ |
| 错误/失败提示 + 建议操作 | ✅ |
| 全部完成提示 + 输出路径 | ✅ |
| 多书时自动检测并显示所有 | ✅ |

### 6.8 CLI 体验

| 功能 | 状态 |
|---|---|
| 交互式菜单（9 选项 + 退出） | ✅ |
| 工作流自动衔接（init→parse→manifest→apply→generate） | ✅ |
| 步骤间按 Enter 继续 / 输入其他返回菜单 | ✅ |
| 直接模式（`./novel-tts generate 0001`） | ✅ |
| Ctrl+C 安全退出 | ✅ |
| CLI 输出中文友好 | ✅ |

---

## 七、性能目标

| 指标 | v1.0 目标 | 实际达成 |
|---|---|---|
| RTF（实时率） | ≤ 1.0 | **0.76** ✅ |
| 显存占用 | ≤ 8 GB | **3.24 GB** ✅ |
| 单段生成 (200字) | ≤ 10s | ~4s |
| 全书生成 (9385段) | ≤ 3h | ~2h |

优化措施详见 README。

---

## 八、用户故事验收场景

### 场景 1: 新人首次使用

1. 用户 clone 项目，安装 conda 环境
2. 运行 `./novel-tts` 看到交互菜单
3. 选 [1] init → 输入小说 TXT 路径
4. 选朗读者 → 选风格 → 试听 → 保存
5. 系统自动衔接: "下一步 parse，按 Enter 继续"
6. 按 Enter → parse 完成 → "下一步 manifest"
7. 按 Enter → 编辑器打开 manifest → 不修改直接保存退出
8. "下一步 apply" → 按 Enter → "无修改"
9. "下一步 generate" → 按 Enter → 开始生成
10. 看到进度条 + ETA → 跑完 → "全部完成！"

### 场景 2: 断点续传

1. 用户在生成中按 Ctrl+C
2. 重新运行 `./novel-tts generate`
3. 系统显示: "已完成: 42 段 (12%)" → [1] 断点续传 [2] 重新生成
4. 选 [1] → 从第 43 段继续

### 场景 3: 章节编辑

1. 用户发现某两章很短应该合并
2. `./novel-tts manifest` → 把第 0002 章的 `K` 改成 `M`
3. 保存退出 → `./novel-tts apply` → 看到 "merge: 0002 → 0001"
4. `./novel-tts generate` — 之前已完成的不受影响

---

## 九、非功能需求

| 类别 | 要求 |
|---|---|
| 可靠性 | Ctrl+C 不丢进度（每段生成后立即写盘） |
| 可观测性 | 生成日志 CSV + stdout 进度 + ETA |
| 兼容性 | 编码 GBK/UTF-8/GB18030 自动检测 |
| 可维护性 | 命令模块独立，单文件职责清晰 |
| 文档 | README 含安装/使用/benchmark/优化记录 |

---

## 十、已知问题 & v1.1 候选

| 项目 | 优先级 |
|---|---|
| 交互式 manifest（非编辑器，CLI 逐章确认 K/M/S/X） | P1 |
| `novel-tts install` 一键环境安装脚本 | P1 |
| 长段落（>500 字）TTS 调用失败时的自动降级拆分 | P2 |
| 说话人角色映射（不同角色用不同 speaker） | P2 |
| ROCm 部署踩坑文章 | P2 |
| HIP native kernel flash-attention（navi_support fork） | P3 |
| 自动化测试套件 | P3 |
| GitHub release | P3 |
