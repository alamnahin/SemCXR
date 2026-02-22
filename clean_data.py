import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

CATEGORY_ALIASES = {
    "normal": "Normal",
    "pneumonia": "Pneumonia",
    "tb": "TB",
    "tuberculosis": "TB",
}


def normalize_category(category: object) -> Optional[str]:
    if category is None or pd.isna(category):
        return None
    key = str(category).strip().lower()
    return CATEGORY_ALIASES.get(key)


def is_lung_case(text: object) -> bool:
    if text is None or pd.isna(text):
        return False
    value = str(text).strip().lower()
    return bool(re.search(r"lung\s+(malignant|benign)\s+case", value))


def strip_slice_suffix(value: str) -> str:
    if not value:
        return value
    # Remove the (1), (2), etc. suffix
    cleaned = re.sub(r"\s*\(\d+\)$", "", value)
    cleaned = re.sub(r"[_-]\d+$", "", cleaned)
    return cleaned


def normalize_patient_id(
    row: pd.Series,
    patient_col: str,
    image_col: str,
    lung_case_only: bool = True,
) -> str:
    original = str(row.get(patient_col, "") or "")
    image_name = str(row.get(image_col, "") or "")
    image_stem = Path(image_name).stem if image_name else ""

    # FIX: Check if the PatientID or Image name contains "lung malignant/benign case"
    if lung_case_only and not (is_lung_case(original) or is_lung_case(image_name)):
        return original

    base = image_stem or original
    return strip_slice_suffix(base)


def mask_diagnosis_terms(text: str, pattern: re.Pattern) -> Tuple[str, bool]:
    if not text or pd.isna(text):
        return text, False
    masked = pattern.sub("[DIAGNOSIS HIDDEN]", text)
    return masked, masked != text


def recompute_folds(
    df: pd.DataFrame,
    category_col: str,
    patient_col: Optional[str],
    num_folds: int,
    seed: int,
) -> Tuple[pd.DataFrame, str]:
    if category_col not in df.columns:
        raise ValueError(f"Missing required column: {category_col}")

    y = df[category_col]
    if patient_col and patient_col in df.columns:
        splitter = StratifiedGroupKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        groups = df[patient_col]
        split_strategy = "stratified_group_kfold_patient"
        splits = splitter.split(df, y, groups=groups)
    else:
        splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        split_strategy = "stratified_kfold_rowwise"
        splits = splitter.split(df, y)

    fold_assignments = np.empty(len(df), dtype=int)
    for fold_idx, (_, val_idx) in enumerate(splits):
        fold_assignments[val_idx] = fold_idx

    df = df.copy()
    df["fold"] = fold_assignments
    return df, split_strategy


def main():
    parser = argparse.ArgumentParser(description="Clean SemCXR CSV to reduce leakage and label noise")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to cleaned CSV")
    parser.add_argument("--image_col", type=str, default="Image")
    parser.add_argument("--patient_col", type=str, default="PatientID")
    parser.add_argument("--category_col", type=str, default="Category")
    parser.add_argument("--category_original_col", type=str, default="Category_Original")
    parser.add_argument("--mask_text", action="store_true", help="Mask diagnosis terms in Clean_Impression")
    parser.add_argument("--drop_mismatched", action="store_true", help="Drop rows with Category != Category_Original")
    parser.add_argument("--normalize_patient", action="store_true", help="Normalize PatientID to remove slice suffixes")
    parser.add_argument("--recompute_folds", action="store_true", help="Recompute folds after cleaning")
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    report: Dict[str, object] = {
        "input_csv": str(input_path),
        "output_csv": str(output_path),
        "rows_before": int(len(df)),
    }

    # 1. Normalize PatientID for lung-case sequences.
    if args.normalize_patient and args.patient_col in df.columns:
        updated = []
        for _, row in df.iterrows():
            updated.append(
                normalize_patient_id(
                    row,
                    patient_col=args.patient_col,
                    image_col=args.image_col,
                    lung_case_only=True,
                )
            )
        updated_series = pd.Series(updated, index=df.index)
        changed = (df[args.patient_col].astype(str) != updated_series.astype(str)).sum()
        df[args.patient_col] = updated_series
        report["patient_id_normalized"] = True
        report["patient_id_changed_rows"] = int(changed)
    else:
        report["patient_id_normalized"] = False

    # 2. Drop rows with mismatched labels (Filters out the garbage CT slices)
    if args.drop_mismatched and args.category_original_col in df.columns:
        normalized_category = df[args.category_col].apply(normalize_category)
        normalized_original = df[args.category_original_col].apply(normalize_category)
        invalid_category = normalized_category.isna()
        invalid_original = normalized_original.isna()
        mismatch = normalized_category != normalized_original
        drop_mask = invalid_category | invalid_original | mismatch
        report["drop_mismatch_total"] = int(drop_mask.sum())
        df = df.loc[~drop_mask].copy()
    else:
        report["drop_mismatch_total"] = 0

    # 3. Mask diagnosis terms in text (FIXED Regex)
    if args.mask_text:
        text_col = "Clean_Impression" if "Clean_Impression" in df.columns else "Impression"
        if text_col in df.columns:
            if "Clean_Impression_Raw" not in df.columns:
                df["Clean_Impression_Raw"] = df[text_col]

            # EXPANDED REGEX: Catches words that imply the diagnosis
            pattern = re.compile(
                r"\b(tuberculosis|tb|pneumonia|pneumonic|pneumonitis|normal|bacterial|viral|infection|infiltrates?)\b",
                re.IGNORECASE,
            )
            masked_rows = 0
            masked_text = []
            for text in df[text_col].fillna("").astype(str):
                new_text, changed = mask_diagnosis_terms(text, pattern)
                masked_text.append(new_text)
                masked_rows += 1 if changed else 0

            df["Clean_Impression"] = masked_text
            report["masked_text_rows"] = int(masked_rows)
            report["masked_text_column"] = "Clean_Impression"
    else:
        report["masked_text_rows"] = 0

    # 4. Recompute fold assignments
    if args.recompute_folds:
        df, strategy = recompute_folds(
            df,
            category_col=args.category_col,
            patient_col=args.patient_col if args.patient_col in df.columns else None,
            num_folds=args.num_folds,
            seed=args.seed,
        )
        report["fold_strategy"] = strategy

    report["rows_after"] = int(len(df))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    report_path = output_path.with_suffix(".clean_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Data cleaning complete! Saved to {output_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
