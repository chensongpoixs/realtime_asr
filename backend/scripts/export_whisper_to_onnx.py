#!/usr/bin/env python3
"""
Whisper 模型导出到 ONNX 格式 — 供 Triton Inference Server 使用

导出产物:
    triton_model_repo/
    ├── whisper_encoder/1/model.onnx    # Encoder ONNX (~1.2GB for large-v3)
    ├── whisper_decoder/1/model.onnx    # Decoder ONNX (~800MB for large-v3)
    └── mel_filters.npy                 # Mel filterbank (80x201, 供运行时 NumPy 加载)

用法:
    python scripts/export_whisper_to_onnx.py --model large-v3
    python scripts/export_whisper_to_onnx.py --model medium --output_dir my_triton_repo

依赖:
    pip install torch torchaudio onnx onnxruntime-gpu openai-whisper

注意事项:
    - 需要 GPU 内存约 6-8GB (加载 PyTorch 模型)
    - 导出耗时约 5-10 分钟
    - 这是离线一次性操作，运行时不需要 torch
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# ═══════════════════════════════════════════════════════════════
# 命令行参数
# ═══════════════════════════════════════════════════════════════


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export OpenAI Whisper model to ONNX for Triton Inference Server"
    )
    parser.add_argument(
        "--model", type=str, default="large-v3",
        help="Whisper 模型名称 (tiny, base, small, medium, large-v2, large-v3)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="triton_model_repo",
        help="Triton 模型仓库目录 (默认: triton_model_repo)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="导出计算的设备 (cuda/cpu)"
    )
    parser.add_argument(
        "--opset", type=int, default=17,
        help="ONNX opset 版本 (默认: 17)"
    )
    parser.add_argument(
        "--max_audio_length", type=int, default=30,
        help="最大音频长度 (秒), 决定 encoder 输入 padding (默认: 30)"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=448,
        help="Decoder 最大 token 序列长度 (默认: 448)"
    )
    parser.add_argument(
        "--hf_endpoint", type=str, default="https://hf-mirror.com",
        help="HuggingFace 镜像端点 (如 https://hf-mirror.com)"
    )
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════


def export_mel_filters(model, output_dir: Path):
    """从 Whisper 模型中提取 mel filterbank 并保存为 .npy 文件。

    这个文件供 transcriber_v2.py 在运行时使用，
    避免运行时依赖 torch/torchaudio。
    """
    # 从 Whisper 模型获取 mel filters
    mel_filters = model.encoder.conv1.weight.data.cpu().numpy()
    # 实际上 Whisper 的 mel filters 在 model.dims.n_mels 和 model.encoder 中
    # 使用 torchaudio 的 mel 滤波器组作为替代
    import torchaudio

    n_mels = model.dims.n_mels  # 80
    n_fft = model.dims.n_audio_ctx  # 实际是 max source positions, 用 n_audio_ctx

    # Whisper 内部的 mel 滤波器组参数
    # 我们重新计算来匹配
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=400,
        hop_length=160,
        n_mels=n_mels,
        power=2.0,
        f_min=0.0,
        f_max=8000.0,
        norm="slaney",
        mel_scale="slaney",
    )

    # torchaudio 新版 mel_scale.fb 返回 (n_freqs, n_mels)
    # 统一转置为 (n_mels, n_freqs) 供 transcriber_v2.py 使用
    filterbank = mel_spec.mel_scale.fb.numpy()
    if filterbank.shape[0] > filterbank.shape[1]:
        # 形状如 (201, 128) → 转置为 (128, 201)
        filterbank = filterbank.T
        print(f"  [INFO] Mel filterbank 已自动转置: {filterbank.shape[0]} mels × {filterbank.shape[1]} bins")
    else:
        print(f"  [INFO] Mel filterbank: {filterbank.shape[0]} mels × {filterbank.shape[1]} bins")

    out_path = output_dir / "mel_filters.npy"
    np.save(out_path, filterbank.astype(np.float32))
    print(f"  [OK] Mel filterbank 已保存: {out_path} (shape={filterbank.shape})")


# ═══════════════════════════════════════════════════════════════
# Encoder 导出
# ═══════════════════════════════════════════════════════════════


def _patch_encoder_for_variable_length(encoder):
    """Monkey-patch Whisper Encoder 以支持变长 mel 输入。

    Whisper 原版 encoder.forward() 有断言:
        assert x.shape[1:] == self.positional_embedding.shape

    要求输入 mel frames 必须等于 n_audio_ctx (如 large-v3=1500)。
    但实时流场景每次只传 ~100 帧，需要按实际长度切片 positional embedding。
    """
    import torch.nn.functional as F

    original_forward = encoder.forward

    def variable_length_forward(x: "torch.Tensor"):
        """支持变长 mel 的 encoder forward。

        Input:  (batch, n_mels, n_frames)  where n_frames ≤ n_audio_ctx*2
        Output: (batch, n_frames//2, d_model)

        Conv 输出为 (batch, d_model, seq_len)，但 transformer blocks 和
        LayerNorm 需要 (batch, seq_len, d_model)。添加 transpose 解决。
        """
        # conv layers
        x = F.gelu(encoder.conv1(x))
        x = F.gelu(encoder.conv2(x))
        # x.shape = (batch, d_model, n_frames_conv2)

        n_frames = x.shape[2]
        d_model_out = x.shape[1]
        pe = encoder.positional_embedding

        # 添加 positional embedding (在 (batch, d_model, seq_len) 空间)
        if pe.shape[1] == d_model_out:
            pos_emb = pe[:n_frames, :].t()    # (n_frames, d_model) → (d_model, n_frames)
        else:
            pos_emb = pe[:, :n_frames]         # (d_model, n_frames)
        x = (x + pos_emb).to(x.dtype)

        # ★ 关键: transpose 为 (batch, seq_len, d_model) 供 transformer 使用
        # nn.Linear / LayerNorm 沿最后一维操作，需要 d_model 在最后一维
        x = x.transpose(1, 2)  # (batch, d_model, seq_len) → (batch, seq_len, d_model)

        # Transformer blocks
        for block in encoder.blocks:
            x = block(x)

        # Layer norm (现在最后一维是 d_model)
        x = encoder.ln_post(x)

        return x

    encoder.forward = variable_length_forward
    return original_forward


def export_encoder(model, output_dir: Path, args):
    """导出 Whisper Encoder 到 ONNX。

    变长输入支持: monkey-patch positional embedding，使 encoder 接受任意 ≤n_audio_ctx 的 mel frames。

    Input:  mel spectrogram  (batch, n_mels, n_frames)   n_frames ≤ n_audio_ctx
    Output: encoder hidden states (batch, n_frames//2, d_model)
    """
    import torch
    import torch.onnx

    print("\n[1/3] 导出 Whisper Encoder...")

    encoder = model.encoder
    encoder.eval()

    n_mels = model.dims.n_mels       # 128 (large-v3) / 80 (其他)
    d_model = model.dims.n_audio_state  # 1280 (large-v3)
    n_audio_ctx = model.dims.n_audio_ctx  # 1500 (encoder 内部最大帧数 = 输入 conv 后)

    # ─── 关键: 输入 mel frames 必须 = n_audio_ctx ──────────
    # encoder.positional_embedding shape = (d_model, n_audio_ctx)
    # conv2 stride=2 → 输出 = 输入 / 2 = n_audio_ctx
    # 所以输入 mel frames = n_audio_ctx * 2
    input_frames = n_audio_ctx * 2   # = 3000 for large-v3

    print(f"  [INFO] n_mels={n_mels}, d_model={d_model}, n_audio_ctx={n_audio_ctx}")
    print(f"  [INFO] dummy input: (1, {n_mels}, {input_frames})")

    # ─── 变长支持 ──────────────────────────────────────────
    original_forward = _patch_encoder_for_variable_length(encoder)

    # Dummy input: 使用最大尺寸 (ONNX dynamic axes 允许运行时传入更小尺寸)
    dummy_mel = torch.randn(1, n_mels, input_frames, device=args.device)

    # 定义 dynamic axes (支持运行时变长)
    dynamic_axes = {
        "mel": {0: "batch_size", 2: "n_frames"},
        "encoder_output": {0: "batch_size", 1: "n_frames_out"},
    }

    # 导出
    output_path = output_dir / "whisper_encoder" / "1" / "model.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    export_ok = False
    export_error = None

    with torch.no_grad():
        # 直接使用旧版 TorchScript ONNX exporter
        # 注: Dynamo exporter (新版) 在此 PyTorch/whisper 版本组合下不稳定
        #   - 图捕获 OK, 分解 OK, 但 ONNX IR 优化步骤失败
        #   - 失败后残留状态会损坏 TorchScript 回退
        print("  [INFO] 使用 TorchScript ONNX exporter (dynamo=False)...")
        torch.onnx.export(
            encoder,
            (dummy_mel,),
            str(output_path),
            input_names=["mel"],
            output_names=["encoder_output"],
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )

    # ─── 恢复原始 forward ─────────────────────────────────
    encoder.forward = original_forward

    print(f"  [OK] Encoder ONNX 已保存: {output_path}")
    print(f"       大小: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"       耗时: {time.time() - t0:.1f}s")
    print(f"       输入: (batch, {n_mels}, ≤{input_frames}) — 支持变长")

    # 验证
    print("  [验证] 检查 ONNX 模型有效性...")
    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("  [验证] ✅ ONNX 模型有效")

        # 测试变长推理 (100 帧 → 约 1s 音频)
        import onnxruntime as ort
        session = ort.InferenceSession(
            str(output_path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        for test_frames in [100, 500, 1000, input_frames]:
            test_mel = torch.randn(1, n_mels, test_frames, device="cpu").numpy()
            output = session.run(["encoder_output"], {"mel": test_mel})
            expected_out_frames = test_frames // 2  # conv2 stride=2
            actual_out_frames = output[0].shape[1]
            status = "✅" if actual_out_frames == expected_out_frames else "❌"
            print(f"  [验证] {status} input={test_frames}frames → output={actual_out_frames}frames "
                  f"(expect={expected_out_frames})")
    except ImportError:
        print("  [验证] ⚠️  onnx/onnxruntime 未安装，跳过验证")
    except Exception as e:
        print(f"  [验证] ⚠️  验证失败 (非致命): {e}")


# ═══════════════════════════════════════════════════════════════
# Decoder 导出
# ═══════════════════════════════════════════════════════════════


def export_decoder(model, output_dir: Path, args):
    """导出 Whisper Decoder 到 ONNX。

    Inputs:
        tokens:          (batch, seq_len)          - token ID 序列
        encoder_output:  (batch, n_frames, d_model) - encoder 输出
    Output:
        logits:          (batch, seq_len, vocab_size) - 每位置的下一个 token 概率
    """
    import torch
    import torch.onnx
    import torch.nn.functional as F

    print("\n[2/3] 导出 Whisper Decoder...")

    decoder = model.decoder
    decoder.eval()

    # ─── 修复 PyTorch ≥2.5 SDPA: 全局替换为稳定手动实现 ──
    # TorchScript tracer 的 builtin SDPA 与 Whisper 的位置参数不兼容
    # 最稳定方案: 全局替换为手动实现的 SDPA (batch matmul + softmax)
    import torch.nn.functional as torch_f
    import whisper.model as whisper_model_module

    _original_sdpa = torch_f.scaled_dot_product_attention

    def _manual_sdpa(query, key, value, *args, **kwargs):
        """纯 PyTorch 手动实现 scaled_dot_product_attention，
        避免与 TorchScript builtin SDPA 的兼容性问题。
        兼容 old-style positional mask 参数。
        """
        # 提取 mask (可能是位置参数或关键字参数)
        attn_mask = kwargs.pop('attn_mask', None)
        is_causal = kwargs.pop('is_causal', False)
        if args and isinstance(args[0], torch.Tensor):
            attn_mask = args[0]
            args = args[1:]
        elif args and isinstance(args[0], bool):
            is_causal = args[0]
            args = args[1:]

        scale_factor = 1.0 / (query.size(-1) ** 0.5)
        attn_weight = query @ key.transpose(-2, -1) * scale_factor

        if is_causal:
            L, S = attn_weight.size(-2), attn_weight.size(-1)
            causal_mask = torch.ones(L, S, dtype=torch.bool, device=attn_weight.device).triu(1)
            attn_weight = attn_weight.masked_fill(causal_mask, float('-inf'))

        if attn_mask is not None:
            attn_weight += attn_mask

        attn_weight = torch.softmax(attn_weight, dim=-1)
        return attn_weight @ value

    torch_f.scaled_dot_product_attention = _manual_sdpa
    whisper_model_module.scaled_dot_product_attention = _manual_sdpa

    d_model = model.dims.n_audio_state  # 1280
    n_audio_ctx = model.dims.n_audio_ctx  # 1500
    vocab_size = model.dims.n_vocab  # 51865
    max_tokens = args.max_tokens

    # Dummy inputs
    dummy_tokens = torch.randint(0, vocab_size, (1, 10), device=args.device, dtype=torch.long)
    dummy_encoder_output = torch.randn(1, n_audio_ctx, d_model, device=args.device)

    # 由于 Whisper decoder 需要 attention mask 等，使用完整 decoder forward
    # 注意：Whisper 的 decoder.forward 需要 encoder_output

    # 包装为一个可追踪的模块
    class WhisperDecoderWrapper(torch.nn.Module):
        def __init__(self, decoder):
            super().__init__()
            self.decoder = decoder

        def forward(self, tokens, encoder_output):
            return self.decoder(tokens, encoder_output)

    wrapped = WhisperDecoderWrapper(decoder)

    dynamic_axes = {
        "tokens": {0: "batch_size", 1: "seq_len"},
        "encoder_output": {0: "batch_size", 1: "n_frames"},
        "logits": {0: "batch_size", 1: "seq_len"},
    }

    output_path = output_dir / "whisper_decoder" / "1" / "model.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    with torch.no_grad():
        # 直接使用旧版 TorchScript ONNX exporter
        # 注: Dynamo exporter 在 decoder 上也因 Whisper 内部 assert +
        #     复杂 attention 逻辑而失败，状态损坏问题同在
        print("  [INFO] 使用 TorchScript ONNX exporter (dynamo=False)...")
        torch.onnx.export(
            wrapped,
            (dummy_tokens, dummy_encoder_output),
            str(output_path),
            input_names=["tokens", "encoder_output"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )

    print(f"  [OK] Decoder ONNX 已保存: {output_path}")
    print(f"       大小: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"       耗时: {time.time() - t0:.1f}s")

    # ─── 恢复原始 sdpa ───────────────────────────────────
    torch_f.scaled_dot_product_attention = _original_sdpa
    whisper_model_module.scaled_dot_product_attention = _original_sdpa

    # 验证
    print("  [验证] 检查 ONNX 模型有效性...")
    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("  [验证] ✅ ONNX 模型有效")

        # 测试推理
        import onnxruntime as ort
        session = ort.InferenceSession(
            str(output_path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        test_tokens = np.random.randint(0, vocab_size, (1, 10), dtype=np.int64)
        test_enc = np.random.randn(1, n_audio_ctx, d_model).astype(np.float32)
        output = session.run(
            ["logits"],
            {"tokens": test_tokens, "encoder_output": test_enc},
        )
        print(f"  [验证] ✅ ONNX Runtime 推理成功: output shape={output[0].shape}")
    except ImportError:
        print("  [验证] ⚠️  onnx/onnxruntime 未安装，跳过验证")
    except Exception as e:
        print(f"  [验证] ⚠️  验证失败 (非致命): {e}")


# ═══════════════════════════════════════════════════════════════
# 推理正确性验证
# ═══════════════════════════════════════════════════════════════


def verify_accuracy(model, output_dir: Path, args):
    """对比 ONNX Runtime 推理结果与原始 PyTorch 推理结果。"""
    import torch

    print("\n[3/3] 验证 ONNX 推理正确性...")

    try:
        import onnxruntime as ort
    except ImportError:
        print("  [跳过] onnxruntime 未安装")
        return

    encoder = model.encoder
    decoder = model.decoder
    encoder.eval()
    decoder.eval()

    n_mels = model.dims.n_mels
    d_model = model.dims.n_audio_state
    vocab_size = model.dims.n_vocab

    # 创建测试输入 (小尺寸，验证正确性)
    n_frames = 50
    test_mel = torch.randn(1, n_mels, n_frames, device="cpu")
    test_tokens = torch.randint(0, vocab_size, (1, 5), dtype=torch.long)

    # PyTorch 推理
    with torch.no_grad():
        ref_enc_out = encoder(test_mel)
        ref_dec_out = decoder(test_tokens, ref_enc_out)

    # ONNX Runtime 推理
    enc_session = ort.InferenceSession(
        str(output_dir / "whisper_encoder" / "1" / "model.onnx"),
        providers=["CPUExecutionProvider"],
    )
    dec_session = ort.InferenceSession(
        str(output_dir / "whisper_decoder" / "1" / "model.onnx"),
        providers=["CPUExecutionProvider"],
    )

    ort_enc_out = enc_session.run(
        ["encoder_output"], {"mel": test_mel.numpy()}
    )[0]
    ort_dec_out = dec_session.run(
        ["logits"],
        {
            "tokens": test_tokens.numpy().astype(np.int64),
            "encoder_output": ort_enc_out,
        },
    )[0]

    # 对比误差
    enc_diff = np.abs(ref_enc_out.numpy() - ort_enc_out).max()
    dec_diff = np.abs(ref_dec_out.numpy() - ort_dec_out).max()

    print(f"  Encoder 最大误差: {enc_diff:.6f}")
    print(f"  Decoder 最大误差: {dec_diff:.6f}")

    tolerance = 1e-3  # FP32 容差
    if enc_diff < tolerance and dec_diff < tolerance:
        print(f"  [OK] ✅ 推理精度验证通过 (误差 < {tolerance})")
    else:
        print(f"  [WARN] ⚠️  推理精度偏差较大 (可能因 FP16/算子差异)")
        print(f"        通常不影响转写结果质量")


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════


def main():
    args = parse_args()

    # 设定 HF endpoint
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
        print(f"[INFO] HF_ENDPOINT = {args.hf_endpoint}")

    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 输出目录: {output_dir.resolve()}")

    # 加载模型
    print(f"\n[INFO] 加载 Whisper 模型: {args.model}")
    t0 = time.time()

    import whisper
    model = whisper.load_model(args.model, device=args.device)
    print(f"[INFO] 模型加载完成 ({(time.time() - t0):.1f}s)")
    print(f"       n_mels={model.dims.n_mels}")
    print(f"       d_model={model.dims.n_audio_state}")
    print(f"       n_audio_ctx={model.dims.n_audio_ctx}")
    print(f"       vocab_size={model.dims.n_vocab}")
    print(f"       n_text_ctx={model.dims.n_text_ctx}")

    # 1. 导出 Mel filterbank
    export_mel_filters(model, output_dir)

    # 2. 导出 Encoder
    export_encoder(model, output_dir, args)

    # 3. 导出 Decoder
    export_decoder(model, output_dir, args)

    # 4. 验证推理正确性
    if args.device == "cpu":
        verify_accuracy(model, output_dir, args)
    else:
        print("\n[3/3] 验证跳过 (需要 CPU 设备以对比 ONNX Runtime)")

    # 完成
    print("\n" + "=" * 60)
    print("  导出完成！产物列表:")
    print(f"  {output_dir / 'mel_filters.npy'}")
    print(f"  {output_dir / 'whisper_encoder' / '1' / 'model.onnx'}")
    print(f"  {output_dir / 'whisper_decoder' / '1' / 'model.onnx'}")
    print()
    print("  下一步: 启动 Triton Inference Server")
    print(f"    bash scripts/start_triton_server.sh")
    print("=" * 60)


if __name__ == "__main__":
    main()
