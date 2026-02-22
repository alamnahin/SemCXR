import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def detect_text_column(df: pd.DataFrame) -> Optional[str]:
    if "Clean_Impression" in df.columns:
        return "Clean_Impression"
    if "Impression" in df.columns:
        return "Impression"
    return None


def analyze_text(df: pd.DataFrame, text_col: str) -> Dict:
    text = df[text_col].fillna("").astype(str)
    char_len = text.str.len()
    word_len = text.str.split().str.len()

    return {
        "text_column": text_col,
        "empty_text_rows": int((text.str.strip() == "").sum()),
        "char_length": {
            "mean": round(float(char_len.mean()), 3),
            "std": round(float(char_len.std(ddof=0)), 3),
            "min": int(char_len.min()) if len(char_len) else 0,
            "p25": round(float(char_len.quantile(0.25)), 3) if len(char_len) else 0,
            "median": round(float(char_len.median()), 3) if len(char_len) else 0,
            "p75": round(float(char_len.quantile(0.75)), 3) if len(char_len) else 0,
            "max": int(char_len.max()) if len(char_len) else 0,
        },
        "word_length": {
            "mean": round(float(word_len.mean()), 3),
            "std": round(float(word_len.std(ddof=0)), 3),
            "min": int(word_len.min()) if len(word_len) else 0,
            "p25": round(float(word_len.quantile(0.25)), 3) if len(word_len) else 0,
            "median": round(float(word_len.median()), 3) if len(word_len) else 0,
            "p75": round(float(word_len.quantile(0.75)), 3) if len(word_len) else 0,
            "max": int(word_len.max()) if len(word_len) else 0,
        },
    }


def analyze_images(df: pd.DataFrame, image_dir: Path, image_col: str = "Image", max_images: int = 0) -> Dict:
    if image_col not in df.columns:
        return {
            "image_column_found": False,
            "message": f"Column '{image_col}' not found in CSV",
        }

    subset = df[[image_col]].copy()
    if max_images and max_images > 0:
        subset = subset.iloc[:max_images].copy()

    records: List[Dict] = []
    missing = 0
    unreadable = 0
    widths = []
    heights = []
    formats = {}

    for img_name in subset[image_col].astype(str):
        img_path = image_dir / img_name
        row = {
            "image": img_name,
            "exists": img_path.exists(),
            "readable": False,
            "width": None,
            "height": None,
            "format": None,
        }

        if not img_path.exists():
            missing += 1
            records.append(row)
            continue

        try:
            with Image.open(img_path) as img:
                width, height = img.size
                fmt = img.format
            row.update(
                {
                    "readable": True,
                    "width": int(width),
                    "height": int(height),
                    "format": fmt,
                }
            )
            widths.append(width)
            heights.append(height)
            formats[fmt] = formats.get(fmt, 0) + 1
        except (UnidentifiedImageError, OSError):
            unreadable += 1

        records.append(row)

    report_df = pd.DataFrame(records)

    image_summary = {
        "image_column_found": True,
        "checked_images": int(len(subset)),
        "missing_files": int(missing),
        "unreadable_files": int(unreadable),
        "existing_readable_files": int((report_df["readable"] == True).sum()) if not report_df.empty else 0,
        "missing_pct": round(100.0 * safe_div(missing, len(subset)), 3),
        "unreadable_pct": round(100.0 * safe_div(unreadable, len(subset)), 3),
        "width_stats": {
            "min": int(np.min(widths)) if widths else None,
            "median": float(np.median(widths)) if widths else None,
            "max": int(np.max(widths)) if widths else None,
        },
        "height_stats": {
            "min": int(np.min(heights)) if heights else None,
            "median": float(np.median(heights)) if heights else None,
            "max": int(np.max(heights)) if heights else None,
        },
        "formats": formats,
        "detailed_rows": report_df,
    }
    return image_summary


def save_plots(df: pd.DataFrame, output_dir: Path, text_col: Optional[str]) -> List[str]:
    created = []
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return created

    if "Category" in df.columns:
        vc = df["Category"].value_counts(dropna=False)
        plt.figure(figsize=(8, 4))
        vc.plot(kind="bar")
        plt.title("Class Distribution")
        plt.ylabel("Count")
        plt.tight_layout()
        p = output_dir / "class_distribution.png"
        plt.savefig(p, dpi=150)
        plt.close()
        created.append(str(p))

    if "Category" in df.columns and "fold" in df.columns:
        ct = pd.crosstab(df["fold"], df["Category"])
        plt.figure(figsize=(9, 5))
        ct.plot(kind="bar", stacked=True, ax=plt.gca())
        plt.title("Class Distribution by Fold")
        plt.ylabel("Count")
        plt.tight_layout()
        p = output_dir / "class_by_fold.png"
        plt.savefig(p, dpi=150)
        plt.close()
        created.append(str(p))

    if text_col is not None:
        lengths = df[text_col].fillna("").astype(str).str.split().str.len()
        plt.figure(figsize=(8, 4))
        plt.hist(lengths, bins=40)
        plt.title(f"Word Count Distribution ({text_col})")
        plt.xlabel("Word count")
        plt.ylabel("Frequency")
        plt.tight_layout()
        p = output_dir / "text_word_count_hist.png"
        plt.savefig(p, dpi=150)
        plt.close()
        created.append(str(p))

    return created


def main():
    parser = argparse.ArgumentParser(description="Dataset EDA report generator for SemCXR-style datasets")
    parser.add_argument("--csv", type=str, default="data/data.csv", help="Path to dataset CSV")
    parser.add_argument("--image_dir", type=str, default="data/images", help="Path to image directory")
    parser.add_argument("--output_dir", type=str, default="eda_reports", help="Output directory for reports")
    parser.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="Max number of images to verify (0 = all)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Basic dataset profile
    summary = {
        "csv_path": str(csv_path),
        "image_dir": str(image_dir),
        "num_rows": int(len(df)),
        "num_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {k: str(v) for k, v in df.dtypes.to_dict().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": {k: int(v) for k, v in df.isna().sum().to_dict().items()},
    }

    # Class distribution
    if "Category" in df.columns:
        class_counts = df["Category"].value_counts(dropna=False)
        class_df = pd.DataFrame(
            {
                "Category": class_counts.index.astype(str),
                "Count": class_counts.values,
            }
        )
        class_df["Percent"] = (100.0 * class_df["Count"] / max(1, len(df))).round(3)
        class_df.to_csv(output_dir / "class_distribution.csv", index=False)
        summary["class_distribution"] = class_df.to_dict(orient="records")
    else:
        summary["class_distribution"] = "Category column not found"

    # Fold × class crosstab
    if "fold" in df.columns and "Category" in df.columns:
        fold_class = pd.crosstab(df["fold"], df["Category"])
        fold_class.to_csv(output_dir / "class_by_fold.csv")
        summary["folds"] = sorted(df["fold"].dropna().unique().tolist())

    # Patient-level quick checks
    if "PatientID" in df.columns:
        summary["num_unique_patients"] = int(df["PatientID"].nunique(dropna=True))
        summary["patients_with_multiple_images"] = int((df["PatientID"].value_counts() > 1).sum())

    # Text stats
    text_col = detect_text_column(df)
    if text_col:
        summary["text_stats"] = analyze_text(df, text_col)
    else:
        summary["text_stats"] = "No text column found (expected Clean_Impression or Impression)"

    # Image checks
    image_summary = analyze_images(df, image_dir=image_dir, image_col="Image", max_images=args.max_images)
    if image_summary.get("image_column_found"):
        image_details = image_summary.pop("detailed_rows")
        image_details.to_csv(output_dir / "image_integrity_report.csv", index=False)
    summary["image_stats"] = image_summary

    # Save missingness table separately
    missing_df = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": [int(df[c].isna().sum()) for c in df.columns],
            "missing_percent": [round(100.0 * float(df[c].isna().mean()), 3) for c in df.columns],
        }
    ).sort_values("missing_count", ascending=False)
    missing_df.to_csv(output_dir / "missing_values.csv", index=False)

    # Plots
    plot_files = save_plots(df, output_dir, text_col)
    summary["plot_files"] = plot_files

    # Save top-level summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("EDA completed.")
    print(f"Output directory: {output_dir.resolve()}")
    print("Generated files:")
    for p in sorted(output_dir.glob("*")):
        print(f"- {p.name}")


if __name__ == "__main__":
    main()
