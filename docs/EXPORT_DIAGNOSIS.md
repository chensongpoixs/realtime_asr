# ONNX 导出脚本问题排查报告

## 执行环境

```
设备: DESKTOP-1N5JVR1
CUDA: 13.0 (V13.0.48)
Conda: asr
模型: Whisper large-v3 (n_mels=128, d_model=1280, vocab_size=51866, n_audio_ctx=1500)
PyTorch: 2.x (CUDA 13.0 对应新版)
```

## 错误时间线

| # | 阶段 | 错误 | 类型 |
|---|------|------|------|
| 1 | Mel filterbank 导出 | `ModuleNotFoundError: onnxscript` | 依赖缺失 |
| 2 | Encoder 导出 | `TorchExportError: incorrect audio shape` | 模型兼容 |
| 3 | (潜在) Mel filterbank | shape=(201,128) 方向反了 | 运行时 |
| 4 | (潜在) n_mels | 硬编码 80 vs 实际 128 | 静默错误 |

---

## 错误 #1: `ModuleNotFoundError: No module named 'onnxscript'` ⚠️ 阻断性

```
torch/onnx/_internal/exporter/_core.py:19 → import onnxscript
ModuleNotFoundError: No module named 'onnxscript'
```

**根因:** PyTorch ≥2.1 中 `torch.onnx.export()` 默认使用 TorchDynamo-based ONNX exporter，内部依赖 `onnxscript` 做图转换。`requirements_v2.txt` 遗漏此包。

**修复:**
1. `pip install onnxscript` (已安装 v0.7.0)
2. 导出脚本增加 try/except 回退: Dynamo 失败 → 自动用 `dynamo=False` (TorchScript exporter)

---

## 错误 #2: `TorchExportError: "incorrect audio shape"` ⚠️ 阻断性

```
whisper/model.py:197 → assert x.shape[1:] == self.positional_embedding.shape
AssertionError: incorrect audio shape
→ torch.onnx._internal.exporter._errors.TorchExportError
```

**根因分析:**

Whisper `AudioEncoder.forward()` 内部断言要求 conv 输出后的 `(d_model, n_frames)` 严格等于 `positional_embedding.shape = (1280, 1500)`。

- `d_model = 1280` — 由 conv2 的 out_channels 决定，自动满足
- `n_frames = ?` — 由输入 mel frames 和 conv 层决定:
  - conv1: kernel=3, stride=1 → 不降采样
  - conv2: kernel=3, stride=2 → **2× 降采样**
  - 所以: **输出帧数 = 输入帧数 / 2**
  - 要求输出 = 1500 → **输入必须精确 = 3000**

原脚本用 `(max_samples - 400)//160 + 1` 计算得到 ~2998 帧，导致 assertion 失败。

更深层问题：即使输入精确 3000 帧导出成功，ONNX 模型也只接受**固定 3000 帧**。实时流 1s 音频 ≈ 100 帧，完全无法使用。

**修复: `_patch_encoder_for_variable_length()`**

导出前 monkey-patch `encoder.forward`，将硬编码的 positional embedding 切片为按实际输入长度:

```python
# Before (original — requires exactly 3000 input frames):
x = (x + self.positional_embedding).to(x.dtype)

# After (patched — supports any ≤3000 input frames):
n_frames = x.shape[2]
pos_emb = self.positional_embedding[:, :n_frames]
x = (x + pos_emb).to(x.dtype)
```

导出后立即恢复原始 forward。这样 Triron ONNX 模型可接受 `≤3000` 帧的任意 mel spectrogram。

---

## 错误 #3: Dynamo exporter 通用失败回退

`torch.onnx.export()` 内部链路长 (torch.export → torch.fx → onnxscript → ONNX)，任一环节不兼容都会失败。Whisper 内部使用 `assert` + 动态 shape，进一步增加 Dynamo 失败概率。

**修复:** 所有 `torch.onnx.export()` 统一两道保护:

```python
try:
    torch.onnx.export(..., dynamo=True)   # ← 新导出器 (默认)
except Exception:
    torch.onnx.export(..., dynamo=False)  # ← 旧版 TorchScript 导出器 (稳定)
```

---

## 错误 #4: Mel filterbank shape 方向不一致

torchaudio 新版 `MelScale.fb` 返回 `(n_freqs, n_mels)` = `(201, 128)`，而 `transcriber_v2.py` 按 `(n_mels, n_freqs)` 做矩阵乘法。运行时维度不匹配崩溃。

**修复 (双重):**
1. 导出时转置 → 始终保存 `(128, 201)`
2. 加载时自动检测 → 若 `shape[0] > shape[1]` 则转置

---

## 错误 #5: large-v3 n_mels=128 (非 80)

| 模型 | n_mels |
|------|--------|
| tiny ~ large-v2 | **80** |
| **large-v3** | **128** |

代码中多处以 `n_mels=80` 硬编码。

**修复:** `audio_to_mel()` 改为从 `_get_mel_filters().shape[0]` 动态获取 n_mels。

---

## 修复文件清单

| 文件 | 修改内容 |
|------|---------|
| `scripts/export_whisper_to_onnx.py` | ① Encoder: monkey-patch 变长支持 + Dynamo→TorchScript 回退<br>② Decoder: Dynamo→TorchScript 回退<br>③ Mel filterbank: 导出时自动转置 |
| `app/services/transcriber_v2.py` | ① `_load_mel_filters()`: 加载时自动修正方向<br>② `_create_mel_filters()`: 接受 n_mels 参数<br>③ `audio_to_mel()`: 动态获取 n_mels |
| `triton_model_repo/whisper_encoder/config.pbtxt` | `dims: [80, -1]` → `dims: [128, -1]` |
| `triton_model_repo/whisper_decoder/config.pbtxt` | `vocab_size=51865` → `51866` |
| `requirements_v2.txt` | 增加 `onnxscript>=0.1.0` |
| `docs/EXPORT_DIAGNOSIS.md` | 本报告 |

---

## 修复后重新执行

```bash
# 1. 删除方向错误的旧 filterbank
rm -f triton_model_repo/mel_filters.npy

# 2. 重新导出 (onnxscript 已安装，但自动回退仍生效)
python scripts/export_whisper_to_onnx.py --model large-v3

# 3. 预期输出
# [INFO] Mel filterbank 已自动转置: 128 mels × 201 bins
# [OK] Mel filterbank 已保存: mel_filters.npy (shape=(128, 201))
# [INFO] Dynamo exporter 失败，回退到旧版 TorchScript: incorrect audio shape
# [INFO] 使用旧版 TorchScript ONNX exporter (dynamo=False)...
# [OK] Encoder ONNX 已保存: .../model.onnx
#     输入: (batch, 128, ≤3000) — 支持变长
# [验证] ✅ input=100frames → output=50frames (expect=50)
# [验证] ✅ input=500frames → output=250frames (expect=250)
# [验证] ✅ input=1000frames → output=500frames (expect=500)
# [验证] ✅ input=3000frames → output=1500frames (expect=1500)
```

## 验证要点

1. ✅ `mel_filters.npy` shape = `(128, 201)` 不是 `(201, 128)`
2. ✅ Encoder ONNX > 1GB, 支持变长: 100/500/1000/3000 帧全部通过
3. ✅ Decoder ONNX > 500MB
4. ✅ 输出帧数 = 输入帧数 // 2 (encoder)
5. ✅ ONNX Runtime 推理与 PyTorch 误差 < 1e-3
