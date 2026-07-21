"""F5-TTS model loading and speaker management utilities.

Replaces chattts_utils.py — uses F5-TTS for higher-quality Chinese speech synthesis.

All heavy imports (torch, F5TTS, numpy, soundfile) are lazy — only loaded
when a function that needs them is called, not at module import time.

Monkey-patches torchaudio to use soundfile backend for AMD ROCm compatibility
(torchcodec requires NVIDIA libnvrtc.so.13).

F5-TTS 工作原理:
  - 必须提供参考音频 (ref_file) + 参考音频文本 (ref_text)，用于音色克隆
  - 模型在 mel 频谱域生成 "参考部分 + 目标部分"，然后切掉参考部分
    (源码: utils_infer.py L508 `generated = generated[:, ref_audio_len:, :]`)
  - 因此输出音频中**不包含**参考音频，无需额外 trim
  - cross_fade_duration 只用于相邻文本批次之间的平滑拼接，与参考音频无关
"""

import os
import sys
import numpy as np

# AMD ROCm environment
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.1")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault("MIOPEN_FIND_MODE", "2")
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")
# NOTE: Do NOT set TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 here.
# AOTriton accelerated SDPA only works on MI200/MI300X/Navi31 GPUs.
# On other AMD GPUs it causes RuntimeError. Let PyTorch use its own fallback.

# ---- 内部状态 ----
_f5tts_instance = None
_patches_applied = False
_imports_ready = False


# ═══════════════════════════════════════════════════════════════════════
# torchaudio Monkey-Patch (AMD ROCm 兼容)
# ═══════════════════════════════════════════════════════════════════════
#
# torchaudio 2.11 默认使用 torchcodec 后端，依赖 NVIDIA libnvrtc.so.13。
# AMD GPU 上没有这个库，所以用 soundfile 替换 torchaudio.load/save。
# 必须在 import f5_tts 之前执行，否则 F5-TTS 内部的 torchaudio 调用会失败。
#

def _apply_torchaudio_patch():
    """Monkey-patch torchaudio to use soundfile backend."""
    global _patches_applied
    if _patches_applied:
        return

    import torch
    import soundfile as sf
    import torchaudio

    def _sf_load(uri, frame_offset=0, num_frames=-1, normalize=True,
                 channels_first=True, **kwargs):
        """soundfile 版 torchaudio.load — 读取 WAV 为 Tensor."""
        data, sample_rate = sf.read(uri, dtype="float32", always_2d=True)
        data = torch.from_numpy(data)  # [time, channel]
        if frame_offset > 0:
            data = data[frame_offset:]
        if num_frames > 0:
            data = data[:num_frames]
        if channels_first:
            data = data.transpose(0, 1)  # [channel, time]
        return data, sample_rate

    def _sf_save(uri, src, sample_rate, channels_first=True, **kwargs):
        """soundfile 版 torchaudio.save — 写 Tensor 为 WAV."""
        if isinstance(src, torch.Tensor):
            src = src.detach().cpu()
            if src.ndim == 2 and channels_first:
                src = src.transpose(0, 1)  # [channel, time] -> [time, channel]
            src = src.numpy()
        sf.write(uri, src, sample_rate)

    torchaudio.load = _sf_load
    torchaudio.save = _sf_save
    torchaudio.load_with_torchcodec = _sf_load
    torchaudio.save_with_torchcodec = _sf_save

    _patches_applied = True


def _ensure_imports():
    """延迟导入重型依赖（torch, F5TTS, soundfile），避免模块加载时触发 GPU 初始化。"""
    global _imports_ready
    if not _imports_ready:
        _apply_torchaudio_patch()
        _imports_ready = True


# ═══════════════════════════════════════════════════════════════════════
# 模型加载
# ═══════════════════════════════════════════════════════════════════════

def get_f5tts(model="F5TTS_v1_Base"):
    """获取或创建 F5TTS 单例（首次调用时加载模型到 GPU）。

    Args:
        model: 模型变体，默认 F5TTS_v1_Base（中文优化）。

    Returns:
        F5TTS 实例。

    模型加载流程 (F5TTS.__init__):
      1. 从 HuggingFace 下载/加载 vocoder (Vocos, 24kHz mel → waveform)
      2. 从 HuggingFace 下载/加载 DiT backbone (F5TTS_v1_Base checkpoint)
      3. 加载字符→拼音→token 的词表
      4. 全部移到 GPU (device="cuda")
    """
    global _f5tts_instance
    _ensure_imports()
    from f5_tts.api import F5TTS
    if _f5tts_instance is None:
        _f5tts_instance = F5TTS(model=model)
    return _f5tts_instance


# ═══════════════════════════════════════════════════════════════════════
# 默认参考音频
# ═══════════════════════════════════════════════════════════════════════

def get_default_ref_audio():
    """F5-TTS 内置中文参考音频路径 (女声, ~6.8s)。

    内容: "有的人称呼我为大自然，也有人叫我大自然母亲。"
    路径: f5_tts/infer/examples/basic/basic_ref_zh.wav
    """
    from importlib.resources import files
    return str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_zh.wav"))


def get_default_ref_text():
    """内置中文参考音频的对应文本。

    F5-TTS 需要 ref_text 来告诉模型参考音频说的是什么，
    模型用这个信息做音色-文本对齐。
    """
    return "有的人称呼我为大自然，也有人叫我大自然母亲。"


# ═══════════════════════════════════════════════════════════════════════
# 推理函数
# ═══════════════════════════════════════════════════════════════════════

def infer_paragraph(f5tts, text, ref_file, ref_text,
                    seed=42, nfe_step=32, speed=1.0,
                    remove_silence=False, cross_fade_duration=0.15):
    """用 F5-TTS 生成一段文本的语音。

    这是最核心的调用，generate_cmd.py 对每个段落都调用此函数。

    F5TTS.infer() 内部流程 (f5_tts/api.py → utils_infer.py):
    ═══════════════════════════════════════════════════════════════════
    1. chunk_text()
       将 gen_text 按句号/换行切成多个 batch，每个 batch 独立推理，
       然后拼接。长文本自动分 batch 避免 OOM。

    2. infer_batch_process() — 对每个 batch:
       a. _infer_basic():
          - ref_text + gen_text 拼接后转拼音
          - 计算 duration (mel 帧数): ref_audio_len + 按比例估算的 gen_len
          - model_obj.sample(cond=ref_audio, text=[...], duration=...)
            → 生成 mel spectrogram, shape [1, duration, mel_dim]
            cond 是参考音频的 mel，提供音色条件
          - generated[:, ref_audio_len:, :]  ← 切掉参考部分！
            只保留生成部分: shape [1, gen_duration, mel_dim]
          - vocoder.decode(mel) → waveform (24000 Hz)
       b. 返回 numpy waveform (1D array)

    3. cross-fade 拼接所有 batch:
       - 相邻 batch 之间做线性淡入淡出 (cross_fade_duration 秒)
       - 默认 0.15s，设为 0 则直接 concat
       - 返回 final_wave

    Args:
        f5tts:   F5TTS 实例（由 get_f5tts() 返回）。
        text:    要合成的文本（一个段落的全部句子拼接）。
        ref_file:  参考音频路径 (WAV)，决定输出音色。
        ref_text:  参考音频的文字内容，模型做对齐用。
        seed:       随机种子，固定值保证可复现。
        nfe_step:   ODE 积分步数 (16-64)。越高音质越好但越慢，32 是平衡点。
        speed:      语速 (0.5-2.0)，1.0 为正常。
        remove_silence: 是否去掉生成音频中的静音段。
        cross_fade_duration: batch 间淡入淡出秒数，默认 0.15。

    Returns:
        (wav_numpy_1d, sample_rate) — sample_rate 固定为 24000。
    """
    # --- 调用 F5TTS 推理 ---
    wav, sr, _ = f5tts.infer(
        ref_file=ref_file,          # 参考音频 → 决定音色
        ref_text=ref_text,           # 参考音频文本 → 对齐用
        gen_text=text,               # 要生成的文本
        seed=seed,                   # 随机种子
        nfe_step=nfe_step,           # 质量/速度权衡
        speed=speed,                 # 语速
        remove_silence=remove_silence,
        cross_fade_duration=cross_fade_duration,  # batch 间平滑拼接
        # show_info 默认=print，会打印 ref_text / gen_text / batch 信息
        # progress 默认=tqdm，显示生成进度条
        # target_rms=0.1, cfg_strength=2 (默认值，未覆盖)
    )

    # --- 后处理：转 numpy + squeeze ---
    # wav 可能是 torch.Tensor (GPU) 或 numpy，统一处理
    if isinstance(wav, __import__("torch").Tensor):
        wav = wav.squeeze().cpu().numpy()
    elif isinstance(wav, np.ndarray):
        wav = wav.squeeze()
    else:
        wav = np.squeeze(np.array(wav))

    # 注: F5-TTS 已在 mel 域切掉了参考音频
    # (utils_infer.py L508: generated = generated[:, ref_audio_len:, :])
    # 因此这里 wav 只包含生成的文本内容，无需额外 trim。

    # --- 音量归一化 ---
    # 防止削波，确保峰值不超过 ±1.0
    wav = wav / max(abs(wav.max()), abs(wav.min()), 1e-8)
    wav = np.clip(wav, -1.0, 1.0)
    return wav, sr


def generate_preview(text="你好，欢迎使用语音合成。",
                     ref_file=None, ref_text=None,
                     seed=42, nfe_step=32, speed=1.0,
                     remove_silence=False, cross_fade_duration=0.15):
    """生成短试听音频，用于 init 交互式选音色时的预览。

    内部调用 infer_paragraph() 相同流程，只是 text 更短（默认 12 字）。

    Args:
        text: 试听文本。
        ref_file, ref_text: 参考音频。
        seed, nfe_step, speed, remove_silence, cross_fade_duration:
            与 infer_paragraph() 相同。

    Returns:
        (wav_numpy, sample_rate) tuple。
    """
    f5tts = get_f5tts()

    wav, sr, _ = f5tts.infer(
        ref_file=ref_file,
        ref_text=ref_text,
        gen_text=text,
        seed=seed,
        nfe_step=nfe_step,
        speed=speed,
        remove_silence=remove_silence,
        cross_fade_duration=cross_fade_duration,
    )
    # 后处理
    if isinstance(wav, __import__("torch").Tensor):
        wav = wav.squeeze().cpu().numpy()
    elif isinstance(wav, np.ndarray):
        wav = wav.squeeze()
    else:
        wav = np.squeeze(np.array(wav))

    # 注: 无需 trim 参考音频，F5-TTS 内部已处理

    # 归一化
    wav = wav / max(abs(wav.max()), abs(wav.min()), 1e-8)
    wav = np.clip(wav, -1.0, 1.0)
    return wav, sr


def save_preview_wav(audio, path, sample_rate=24000):
    """保存预览音频为 WAV 文件。"""
    import soundfile as sf
    sf.write(path, audio, sample_rate)
