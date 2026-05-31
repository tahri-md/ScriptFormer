import json
from pathlib import Path

import cv2
import numpy as np
import torch


def _safe_token_label(tokenizer, token_id: int) -> str:
    token = tokenizer.id_to_char.get(token_id, tokenizer.unk_token)
    if token in getattr(tokenizer, "special_tokens", []):
        return token
    return token


def _register_cross_attention_hooks(model):
    layer_records = []
    originals = []

    decoder_layers = model.decoder.decoder.layers
    for layer_index, layer in enumerate(decoder_layers):
        attention_module = layer.multihead_attn
        original_forward = attention_module.forward
        originals.append((attention_module, original_forward))

        def wrapped_forward(query, key, value, _original_forward=original_forward, _layer_index=layer_index, **kwargs):
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = False
            output, weights = _original_forward(query, key, value, **kwargs)
            layer_records.append((_layer_index, weights.detach().cpu()))
            return output, weights

        attention_module.forward = wrapped_forward

    return layer_records, originals


def _restore_attention_hooks(originals):
    for attention_module, original_forward in originals:
        attention_module.forward = original_forward


def generate_with_attention(model, tokenizer, images: torch.Tensor, max_length: int = None):
    if images.size(0) != 1:
        raise ValueError("Attention export currently supports batch size 1 only.")

    if max_length is None:
        max_length = model.max_length

    was_training = model.training
    model.eval()

    layer_records, originals = _register_cross_attention_hooks(model)
    try:
        encoder_output = model.encoder(images)
        if model.encoder_transformer is not None:
            encoder_output = model.encoder_transformer(encoder_output)

        generated = torch.full(
            (1, 1),
            model.sos_id,
            dtype=torch.long,
            device=images.device,
        )

        steps = []
        decoder_layer_count = len(model.decoder.decoder.layers)

        for _ in range(max_length - 1):
            del layer_records[:]
            logits = model.decoder(encoder_output, generated)
            next_logits = logits[:, -1, :]
            next_token = next_logits.argmax(dim=-1)
            token_id = int(next_token.item())

            relevant_records = layer_records[-decoder_layer_count:] if decoder_layer_count else []
            if not relevant_records:
                raise RuntimeError("No attention weights were captured from the decoder.")

            layer_index, weights = max(relevant_records, key=lambda item: item[0])
            attention = weights.mean(dim=1)[0, -1, :].numpy()

            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

            if token_id == model.eos_id:
                break
            if token_id == model.pad_id:
                continue

            steps.append(
                {
                    "step": len(steps) + 1,
                    "token_id": token_id,
                    "token": _safe_token_label(tokenizer, token_id),
                    "attention": attention,
                    "layer_index": layer_index,
                }
            )

        decoded_text = tokenizer.decode(generated[0].tolist())
        return decoded_text, generated[0].tolist(), steps
    finally:
        _restore_attention_hooks(originals)
        if was_training:
            model.train()


def _attention_overlay(image: np.ndarray, attention: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    attention = np.asarray(attention, dtype=np.float32)
    attention = attention - attention.min()
    if attention.max() > 0:
        attention = attention / attention.max()

    height, width = image.shape[:2]
    if attention.size == 1:
        attention_map = np.full((height, width), float(attention[0]), dtype=np.float32)
    else:
        attention_map = cv2.resize(attention[None, :], (width, height), interpolation=cv2.INTER_CUBIC)

    attention_uint8 = np.clip(attention_map * 255.0, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(attention_uint8, cv2.COLORMAP_JET)

    if image.ndim == 2:
        base = cv2.cvtColor(np.clip(image * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        base = np.clip(image * 255.0, 0, 255).astype(np.uint8)

    overlay = cv2.addWeighted(base, 1.0 - alpha, heatmap, alpha, 0)
    return overlay


def export_attention_overlays(pipeline, image_path: str, export_root: str, max_length: int = None) -> dict:
    export_dir = Path(export_root) / Path(image_path).stem
    export_dir.mkdir(parents=True, exist_ok=True)

    raw_image = cv2.imread(image_path)
    if raw_image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    processed = pipeline.preprocessor(
        raw_image,
        target_height=pipeline.image_height,
        target_width=pipeline.image_width,
    )
    image_tensor = torch.from_numpy(processed).unsqueeze(0).unsqueeze(0).float().to(pipeline.device)

    text, token_ids, steps = generate_with_attention(
        pipeline.model,
        pipeline.tokenizer,
        image_tensor,
        max_length=max_length or pipeline.max_length,
    )

    base_path = export_dir / "preprocessed.png"
    cv2.imwrite(str(base_path), np.clip(processed * 255.0, 0, 255).astype(np.uint8))

    manifest_steps = []
    for step in steps:
        overlay = _attention_overlay(processed, step["attention"])
        token_slug = f"step_{step['step']:02d}_id_{step['token_id']:03d}.png"
        overlay_path = export_dir / token_slug
        cv2.imwrite(str(overlay_path), overlay)
        manifest_steps.append(
            {
                "step": step["step"],
                "token_id": step["token_id"],
                "token": step["token"],
                "layer_index": step["layer_index"],
                "overlay_file": overlay_path.name,
                "attention_max": float(np.max(step["attention"])) if len(step["attention"]) else 0.0,
                "attention_mean": float(np.mean(step["attention"])) if len(step["attention"]) else 0.0,
            }
        )

    manifest = {
        "image_path": image_path,
        "prediction": text,
        "token_ids": token_ids,
        "export_dir": str(export_dir),
        "preprocessed_file": base_path.name,
        "steps": manifest_steps,
    }

    with open(export_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest
