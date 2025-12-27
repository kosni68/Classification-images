#!/usr/bin/env python3
"""
Zero-shot image classification for a folder using CLIP.

Example:
  python classify_images.py --config config.json
  python classify_images.py --input-dir "C:\\images" --labels "enfant,famille,travail,autre"
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
import shutil
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
DEFAULT_CONFIG_NAME = "config.json"

# Built-in prompts for common French labels to improve results with English CLIP text.
BUILTIN_PROMPTS: Dict[str, List[str]] = {
    "enfant": [
        "a photo of a child",
        "a photo of a kid",
        "a photo of a boy",
        "a photo of a girl",
    ],
    "famille": [
        "a photo of a family",
        "a family portrait",
        "parents with children",
    ],
    "travail": [
        "people working",
        "a person at work",
        "office work",
        "construction workers",
    ],
}


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a JSON object.")
    return data


def resolve_config_path(cli_value: Optional[str]) -> Optional[Path]:
    if cli_value:
        return Path(cli_value)
    default_path = Path(DEFAULT_CONFIG_NAME)
    if default_path.exists():
        return default_path
    return None


def resolve_config_paths(config: Dict[str, Any], config_dir: Path) -> None:
    for key in ("input_dir", "output_csv", "copy_to", "move_to"):
        value = config.get(key)
        if isinstance(value, str):
            if not value.strip():
                config[key] = None
                continue
            path = Path(value)
            if not path.is_absolute():
                config[key] = str(config_dir / path)


def coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return default


def coerce_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def coerce_prompt_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw] if raw.strip() else []
    if isinstance(raw, list):
        cleaned: List[str] = []
        for item in raw:
            text = str(item).strip()
            if text:
                cleaned.append(text)
        return cleaned
    raise ValueError("label_prompts must be a string or list of strings.")


def merge_prompts(
    base: Dict[str, List[str]], overrides: Dict[str, Any]
) -> Dict[str, List[str]]:
    merged = {key: list(value) for key, value in base.items()}
    for label, prompts in overrides.items():
        cleaned = coerce_prompt_list(prompts)
        if cleaned:
            merged[label] = cleaned
    return merged


def parse_labels(raw: Any) -> List[str]:
    if raw is None:
        raise ValueError("No labels provided.")
    if isinstance(raw, list):
        labels = [str(item).strip() for item in raw if str(item).strip()]
    else:
        labels = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not labels:
        raise ValueError("No labels provided.")
    return labels


def iter_images(root: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        iterator = root.rglob("*")
    else:
        iterator = root.iterdir()
    for path in iterator:
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True)


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for i in range(1, 10000):
        candidate = path.with_name(f"{stem}_{i}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find a unique filename for {path}")


def build_label_features(
    model: CLIPModel,
    processor: CLIPProcessor,
    labels: List[str],
    prompt_template: str,
    prompts_by_label: Dict[str, List[str]],
    device: str,
) -> Tuple[torch.Tensor, List[str]]:
    if "{label}" not in prompt_template:
        prompt_template = prompt_template + " {label}"

    label_to_prompts: Dict[str, List[str]] = {}
    for label in labels:
        prompts = prompts_by_label.get(label)
        if not prompts:
            prompts = [prompt_template.format(label=label)]
        label_to_prompts[label] = prompts

    all_prompts: List[str] = []
    for prompts in label_to_prompts.values():
        all_prompts.extend(prompts)

    text_inputs = processor(text=all_prompts, return_tensors="pt", padding=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
    text_features = normalize(text_features)

    label_features = []
    index = 0
    for label in labels:
        prompts = label_to_prompts[label]
        count = len(prompts)
        features = text_features[index : index + count]
        index += count
        mean_feature = features.mean(dim=0)
        label_features.append(normalize(mean_feature))

    return torch.stack(label_features, dim=0), labels


def classify_image(
    model: CLIPModel,
    processor: CLIPProcessor,
    label_features: torch.Tensor,
    label_names: List[str],
    image_path: Path,
    device: str,
) -> Tuple[str, float]:
    with Image.open(image_path) as img:
        image = img.convert("RGB")

    image_inputs = processor(images=image, return_tensors="pt")
    image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
    with torch.no_grad():
        image_features = model.get_image_features(**image_inputs)
    image_features = normalize(image_features)

    logits = image_features @ label_features.T
    probs = logits.softmax(dim=-1).squeeze(0)
    best_index = int(torch.argmax(probs).item())
    best_score = float(probs[best_index].item())
    return label_names[best_index], best_score


def main() -> int:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None)
    pre_args, _ = pre_parser.parse_known_args()

    config_path = resolve_config_path(pre_args.config)
    config: Dict[str, Any] = {}
    if config_path:
        try:
            config = load_config(config_path)
        except Exception as exc:
            print(f"Failed to load config: {exc}", file=sys.stderr)
            return 2
        resolve_config_paths(config, config_path.parent)

    input_default = config.get("input_dir")
    if isinstance(input_default, str) and not input_default.strip():
        input_default = None
    input_required = input_default is None

    labels_default = config.get("labels", "enfant,famille,travail,autre")
    prompt_template_default = config.get("prompt_template", "a photo of {label}")
    model_default = config.get("model", "openai/clip-vit-base-patch32")
    threshold_default = coerce_float(config.get("threshold"), 0.25)
    fallback_default = config.get("fallback_label")
    if isinstance(fallback_default, str) and not fallback_default.strip():
        fallback_default = None
    output_csv_default = config.get("output_csv")
    if isinstance(output_csv_default, str) and not output_csv_default.strip():
        output_csv_default = None
    copy_to_default = config.get("copy_to")
    if isinstance(copy_to_default, str) and not copy_to_default.strip():
        copy_to_default = None
    move_to_default = config.get("move_to")
    if isinstance(move_to_default, str) and not move_to_default.strip():
        move_to_default = None
    recursive_default = coerce_bool(config.get("recursive"), False)

    config_prompts = config.get("label_prompts") or {}
    if not isinstance(config_prompts, dict):
        print("label_prompts must be a JSON object.", file=sys.stderr)
        return 2
    try:
        prompts_by_label = merge_prompts(BUILTIN_PROMPTS, config_prompts)
    except ValueError as exc:
        print(f"Invalid label_prompts: {exc}", file=sys.stderr)
        return 2

    parser = argparse.ArgumentParser(description="Classify images in a folder using CLIP.")
    parser.add_argument(
        "--config",
        default=str(config_path) if config_path else None,
        help="Path to config JSON. If omitted and config.json exists, it is used.",
    )
    parser.add_argument("--input-dir", required=input_required, default=input_default, help="Folder with images to classify.")
    parser.add_argument(
        "--labels",
        default=labels_default,
        help="Comma-separated labels.",
    )
    parser.add_argument(
        "--prompt-template",
        default=prompt_template_default,
        help="Template for labels not in built-in prompts.",
    )
    parser.add_argument(
        "--model",
        default=model_default,
        help="CLIP model id on Hugging Face.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=threshold_default,
        help="Below this score, use fallback label if set.",
    )
    parser.add_argument(
        "--fallback-label",
        default=fallback_default,
        help='Label to use when score is low. If omitted and "autre" is in labels, it is used.',
    )
    parser.add_argument("--output-csv", default=output_csv_default, help="CSV output path.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=recursive_default,
        help="Scan subfolders.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_false",
        dest="recursive",
        help="Do not scan subfolders.",
    )
    parser.add_argument(
        "--copy-to",
        default=copy_to_default,
        help="Copy images into label subfolders under this directory.",
    )
    parser.add_argument(
        "--move-to",
        default=move_to_default,
        help="Move images into label subfolders under this directory.",
    )
    args = parser.parse_args()

    if args.copy_to and args.move_to:
        print("Use only one of --copy-to or --move-to.", file=sys.stderr)
        return 2

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 2

    labels = parse_labels(args.labels)
    fallback_label = args.fallback_label
    if fallback_label is None and "autre" in labels:
        fallback_label = "autre"

    candidate_labels = [label for label in labels if label != fallback_label]
    if not candidate_labels:
        print("No candidate labels left after removing fallback label.", file=sys.stderr)
        return 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(args.model).to(device)
    processor = CLIPProcessor.from_pretrained(args.model)
    model.eval()

    label_features, label_names = build_label_features(
        model, processor, candidate_labels, args.prompt_template, prompts_by_label, device
    )

    output_rows: List[Tuple[str, str, float]] = []
    image_paths = list(iter_images(input_dir, args.recursive))
    if not image_paths:
        print("No images found.", file=sys.stderr)
        return 1

    output_dir = None
    if args.copy_to:
        output_dir = Path(args.copy_to)
    elif args.move_to:
        output_dir = Path(args.move_to)

    for image_path in image_paths:
        try:
            label, score = classify_image(
                model, processor, label_features, label_names, image_path, device
            )
        except Exception as exc:
            print(f"Skip {image_path}: {exc}", file=sys.stderr)
            continue

        if fallback_label and score < args.threshold:
            label = fallback_label

        output_rows.append((str(image_path), label, score))

        if output_dir is not None:
            target_dir = output_dir / label
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = unique_path(target_dir / image_path.name)
            if args.move_to:
                shutil.move(str(image_path), str(target_path))
            else:
                shutil.copy2(str(image_path), str(target_path))

    if args.output_csv:
        output_path = Path(args.output_csv)
    else:
        output_path = input_dir / "predictions.csv"

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "label", "score"])
        for row in output_rows:
            writer.writerow(row)

    print(f"Processed {len(output_rows)} images.")
    print(f"CSV: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
