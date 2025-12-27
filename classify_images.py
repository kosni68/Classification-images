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


def coerce_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
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


def coerce_label_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    if isinstance(raw, str):
        if not raw.strip():
            return []
        return [item.strip() for item in raw.split(",") if item.strip()]
    raise ValueError("Labels must be a list or comma-separated string.")


def coerce_label_mapping(raw: Any) -> Dict[str, List[str]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("coarse_to_fine must be a JSON object.")
    mapping: Dict[str, List[str]] = {}
    for key, value in raw.items():
        label = str(key).strip()
        if not label:
            continue
        mapping[label] = coerce_label_list(value)
    return mapping


def try_directml_device() -> Optional[Any]:
    try:
        import torch_directml  # type: ignore
    except Exception:
        return None
    try:
        return torch_directml.device()
    except Exception:
        return None


def resolve_device(requested: Optional[str]) -> Tuple[Any, str]:
    if requested is None:
        requested = "auto"
    if not isinstance(requested, str):
        requested = str(requested)
    value = requested.strip().lower()
    if value in {"", "auto"}:
        if torch.cuda.is_available():
            return torch.device("cuda"), "cuda"
        dml_device = try_directml_device()
        if dml_device is not None:
            return dml_device, "directml"
        return torch.device("cpu"), "cpu"
    if value in {"cuda", "gpu"}:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda"), "cuda"
    if value in {"directml", "dml"}:
        dml_device = try_directml_device()
        if dml_device is None:
            raise RuntimeError(
                "DirectML requested but unavailable. Install torch-directml and update GPU drivers."
            )
        return dml_device, "directml"
    if value == "cpu":
        return torch.device("cpu"), "cpu"
    raise ValueError(f"Unknown device: {requested}")


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


def iter_batches(items: List[Path], batch_size: int) -> Iterable[List[Path]]:
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


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


def classify_image_probs(
    model: CLIPModel,
    processor: CLIPProcessor,
    label_features: torch.Tensor,
    image_path: Path,
    device: str,
) -> torch.Tensor:
    with Image.open(image_path) as img:
        image = img.convert("RGB")

    image_inputs = processor(images=image, return_tensors="pt")
    image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
    with torch.no_grad():
        image_features = model.get_image_features(**image_inputs)
    image_features = normalize(image_features)

    logits = image_features @ label_features.T
    return logits.softmax(dim=-1).squeeze(0)


def classify_batch_probs(
    model: CLIPModel,
    processor: CLIPProcessor,
    label_features: torch.Tensor,
    image_paths: List[Path],
    device: str,
) -> List[Tuple[Path, torch.Tensor]]:
    images: List[Image.Image] = []
    valid_paths: List[Path] = []
    for image_path in image_paths:
        try:
            with Image.open(image_path) as img:
                images.append(img.convert("RGB"))
                valid_paths.append(image_path)
        except Exception as exc:
            print(f"Skip {image_path}: {exc}", file=sys.stderr)
    if not images:
        return []

    image_inputs = processor(images=images, return_tensors="pt")
    image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
    with torch.no_grad():
        image_features = model.get_image_features(**image_inputs)
    image_features = normalize(image_features)

    logits = image_features @ label_features.T
    probs = logits.softmax(dim=-1)
    return list(zip(valid_paths, probs))


def pick_best_label(label_names: List[str], probs: torch.Tensor) -> Tuple[str, float]:
    best_index = int(torch.argmax(probs).item())
    return label_names[best_index], float(probs[best_index].item())


def top_k_from_probs(
    label_names: List[str], probs: torch.Tensor, k: int
) -> List[Tuple[str, float]]:
    if k <= 0:
        return []
    k = min(k, probs.numel())
    values, indices = torch.topk(probs, k)
    top: List[Tuple[str, float]] = []
    for rank, index in enumerate(indices):
        top.append((label_names[int(index)], float(values[rank].item())))
    return top


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
    device_default = config.get("device", "auto")
    if not isinstance(device_default, str) or not device_default.strip():
        device_default = "auto"
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
    top_k_default = coerce_int(config.get("top_k"), 1)
    if top_k_default < 1:
        top_k_default = 1
    batch_size_default = coerce_int(config.get("batch_size"), 4)
    if batch_size_default < 1:
        batch_size_default = 1
    coarse_labels_default = config.get("coarse_labels")
    coarse_to_fine_default = config.get("coarse_to_fine")
    two_stage_default = coerce_bool(config.get("two_stage"), False)
    if "two_stage" not in config and coarse_labels_default is not None:
        two_stage_default = True

    config_prompts = config.get("label_prompts") or {}
    if not isinstance(config_prompts, dict):
        print("label_prompts must be a JSON object.", file=sys.stderr)
        return 2
    try:
        prompts_by_label = merge_prompts(BUILTIN_PROMPTS, config_prompts)
    except ValueError as exc:
        print(f"Invalid label_prompts: {exc}", file=sys.stderr)
        return 2

    coarse_prompts_config = config.get("coarse_label_prompts")
    if coarse_prompts_config is None:
        coarse_prompts_config = {}
    if not isinstance(coarse_prompts_config, dict):
        print("coarse_label_prompts must be a JSON object.", file=sys.stderr)
        return 2
    try:
        coarse_prompts_by_label = merge_prompts(BUILTIN_PROMPTS, coarse_prompts_config)
    except ValueError as exc:
        print(f"Invalid coarse_label_prompts: {exc}", file=sys.stderr)
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
        "--device",
        default=device_default,
        help="Device to use: auto, cpu, cuda, directml.",
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=batch_size_default,
        help="Number of images per batch.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=top_k_default,
        help="Include top-K predictions in CSV.",
    )
    parser.add_argument(
        "--two-stage",
        action="store_true",
        default=two_stage_default,
        help="Enable two-stage classification if coarse labels are configured.",
    )
    parser.add_argument(
        "--no-two-stage",
        action="store_false",
        dest="two_stage",
        help="Disable two-stage classification.",
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
    top_k = max(1, args.top_k)
    batch_size = max(1, args.batch_size)

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

    coarse_labels: List[str] = []
    coarse_to_fine: Dict[str, List[str]] = {}
    if args.two_stage:
        if coarse_labels_default is None:
            print("two-stage enabled but no coarse_labels provided.", file=sys.stderr)
            return 2
        try:
            coarse_labels = parse_labels(coarse_labels_default)
        except ValueError as exc:
            print(f"Invalid coarse_labels: {exc}", file=sys.stderr)
            return 2
        try:
            coarse_to_fine = coerce_label_mapping(coarse_to_fine_default)
        except ValueError as exc:
            print(f"Invalid coarse_to_fine: {exc}", file=sys.stderr)
            return 2
        if not coarse_labels:
            print("No coarse labels provided for two-stage classification.", file=sys.stderr)
            return 2

    try:
        device, device_name = resolve_device(args.device)
    except Exception as exc:
        print(f"Device error: {exc}", file=sys.stderr)
        return 2
    print(f"Device: {device_name}")
    model = CLIPModel.from_pretrained(args.model).to(device)
    processor = CLIPProcessor.from_pretrained(args.model)
    model.eval()

    fine_label_features, fine_label_names = build_label_features(
        model, processor, candidate_labels, args.prompt_template, prompts_by_label, device
    )
    fine_label_to_index = {label: idx for idx, label in enumerate(fine_label_names)}

    coarse_label_features: Optional[torch.Tensor] = None
    coarse_label_names: List[str] = []
    coarse_to_fine_features: Dict[str, Tuple[torch.Tensor, List[str]]] = {}
    if args.two_stage:
        coarse_label_features, coarse_label_names = build_label_features(
            model, processor, coarse_labels, args.prompt_template, coarse_prompts_by_label, device
        )
        missing_mappings = [label for label in coarse_labels if label not in coarse_to_fine]
        if missing_mappings:
            print(
                "Warning: no coarse_to_fine mapping for: "
                + ", ".join(missing_mappings),
                file=sys.stderr,
            )
        for coarse_label in coarse_labels:
            fine_labels = coarse_to_fine.get(coarse_label, [])
            if not fine_labels:
                continue
            indices = [
                fine_label_to_index[label]
                for label in fine_labels
                if label in fine_label_to_index
            ]
            if not indices:
                continue
            subset_names = [fine_label_names[index] for index in indices]
            coarse_to_fine_features[coarse_label] = (
                fine_label_features[indices],
                subset_names,
            )

    output_rows: List[Tuple[str, str, float, List[Tuple[str, float]]]] = []
    image_paths = list(iter_images(input_dir, args.recursive))
    if not image_paths:
        print("No images found.", file=sys.stderr)
        return 1

    output_dir = None
    if args.copy_to:
        output_dir = Path(args.copy_to)
    elif args.move_to:
        output_dir = Path(args.move_to)

    for batch_paths in iter_batches(image_paths, batch_size):
        if args.two_stage and coarse_label_features is not None:
            coarse_results = classify_batch_probs(
                model, processor, coarse_label_features, batch_paths, device
            )
            if not coarse_results:
                continue

            ordered_paths = [path for path, _ in coarse_results]
            coarse_info_by_path: Dict[Path, Tuple[str, float, List[Tuple[str, float]]]] = {}
            results_by_path: Dict[Path, Tuple[str, float, List[Tuple[str, float]]]] = {}
            fine_groups: Dict[str, List[Path]] = {}

            for path, coarse_probs in coarse_results:
                coarse_label, coarse_score = pick_best_label(
                    coarse_label_names, coarse_probs
                )
                coarse_top_pairs = top_k_from_probs(
                    coarse_label_names, coarse_probs, top_k
                )
                coarse_info_by_path[path] = (
                    coarse_label,
                    coarse_score,
                    coarse_top_pairs,
                )
                fine_subset = coarse_to_fine_features.get(coarse_label)
                if fine_subset:
                    fine_groups.setdefault(coarse_label, []).append(path)
                else:
                    results_by_path[path] = coarse_info_by_path[path]

            for coarse_label, group_paths in fine_groups.items():
                subset_features, subset_names = coarse_to_fine_features[coarse_label]
                fine_results = classify_batch_probs(
                    model, processor, subset_features, group_paths, device
                )
                for path, probs in fine_results:
                    label, score = pick_best_label(subset_names, probs)
                    top_pairs = top_k_from_probs(subset_names, probs, top_k)
                    results_by_path[path] = (label, score, top_pairs)

            for path in ordered_paths:
                result = results_by_path.get(path) or coarse_info_by_path.get(path)
                if result is None:
                    continue
                label, score, top_pairs = result

                if fallback_label and score < args.threshold:
                    label = fallback_label

                output_rows.append((str(path), label, score, top_pairs))

                if output_dir is not None:
                    target_dir = output_dir / label
                    target_dir.mkdir(parents=True, exist_ok=True)
                    target_path = unique_path(target_dir / path.name)
                    if args.move_to:
                        shutil.move(str(path), str(target_path))
                    else:
                        shutil.copy2(str(path), str(target_path))
        else:
            batch_results = classify_batch_probs(
                model, processor, fine_label_features, batch_paths, device
            )
            for path, probs in batch_results:
                label, score = pick_best_label(fine_label_names, probs)
                top_pairs = top_k_from_probs(fine_label_names, probs, top_k)

                if fallback_label and score < args.threshold:
                    label = fallback_label

                output_rows.append((str(path), label, score, top_pairs))

                if output_dir is not None:
                    target_dir = output_dir / label
                    target_dir.mkdir(parents=True, exist_ok=True)
                    target_path = unique_path(target_dir / path.name)
                    if args.move_to:
                        shutil.move(str(path), str(target_path))
                    else:
                        shutil.copy2(str(path), str(target_path))

    if args.output_csv:
        output_path = Path(args.output_csv)
    else:
        output_path = input_dir / "predictions.csv"

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = ["path", "label", "score"]
        if top_k > 1:
            for idx in range(1, top_k + 1):
                header.extend([f"top{idx}_label", f"top{idx}_score"])
        writer.writerow(header)
        for path_value, label_value, score_value, top_pairs in output_rows:
            row = [path_value, label_value, score_value]
            if top_k > 1:
                for idx in range(top_k):
                    if idx < len(top_pairs):
                        row.extend([top_pairs[idx][0], top_pairs[idx][1]])
                    else:
                        row.extend(["", ""])
            writer.writerow(row)

    print(f"Processed {len(output_rows)} images.")
    print(f"CSV: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
