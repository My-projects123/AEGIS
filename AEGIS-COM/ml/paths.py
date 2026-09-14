from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "ml" / "models"
FP32_ONNX = MODELS / "tiny_mask.onnx"
INT8_ONNX = MODELS / "tiny_mask.int8.onnx"
META_JSON = MODELS / "tiny_mask.meta.json"
WEIGHTS = MODELS / "tiny_mask.pt"
