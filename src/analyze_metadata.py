#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, Tuple
import itertools
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NA_VALUES = [
    "", " ", "NA", "N/A", "na", "n/a", "NaN", "nan", "None", "none", "NULL", "null", ".", "-", "?", "missing"
]

CORE_FIELDS_DEFAULT = ["taxonID_index", "Habitat", "Substrate", "Latitude", "Longitude", "eventDate"]


def read_metadata(csv_path: Path, date_col: str, sep: str = ",") -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        sep=sep,
        dtype={
            "filename_index": "string",
            "Habitat": "string",
            "Substrate": "string",
            "taxonID_index": "string",
        },
        parse_dates=[date_col],
        infer_datetime_format=True,
        keep_date_col=False,
        na_values=NA_VALUES,
        keep_default_na=True,
    )

    # Strip whitespace in string columns
    for c in df.select_dtypes(include="string").columns:
        df[c] = df[c].str.strip()

    # Coerce coords
    for c in ("Latitude", "Longitude"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def filter_split(df: pd.DataFrame, split: str, id_col: str = "filename_index") -> pd.DataFrame:
    if split == "all":
        return df
    prefix = f"fungi_{split}"
    keep = df[id_col].str.startswith(prefix, na=False)
    return df[keep].copy()


def basic_missingness(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    rows = []
    for c in df.columns:
        nn = df[c].notna().sum()
        na = n - nn
        uniq = df[c].nunique(dropna=True)
        example = None
        if nn > 0:
            example = df.loc[df[c].notna(), c].iloc[0]
            if isinstance(example, pd.Timestamp):
                example = str(example.date())
            else:
                example = str(example)[:60]
        rows.append((c, nn, na, (na / n * 100) if n else 0.0, uniq, example))
    out = pd.DataFrame(rows,
                       columns=["column", "non_null", "missing", "missing_pct", "unique_non_null", "example_non_null"])
    return out.sort_values("missing_pct", ascending=False)


def row_completeness(df: pd.DataFrame, exclude: list[str]) -> pd.Series:
    cols = [c for c in df.columns if c not in exclude]
    return df[cols].notna().mean(axis=1)  # fraction of non-null per row


def per_class_coverage(df: pd.DataFrame, class_col: str, key_fields: list[str]) -> pd.DataFrame:
    def cov(g: pd.DataFrame) -> dict:
        d = {"count": len(g)}
        for k in key_fields:
            if k in g.columns:
                d[f"{k}__non_null"] = g[k].notna().sum()
                d[f"{k}__coverage_pct"] = g[k].notna().mean() * 100
            else:
                d[f"{k}__non_null"] = np.nan
                d[f"{k}__coverage_pct"] = np.nan
        return d

    blocks = []
    for cls, g in df.groupby(class_col, dropna=False):
        info = cov(g)
        info[class_col] = cls if pd.notna(cls) else "⟂(missing)"
        blocks.append(info)
    out = pd.DataFrame(blocks).set_index(class_col).sort_values("count", ascending=False)
    return out


def coord_quality(df: pd.DataFrame) -> pd.DataFrame:
    if not {"Latitude", "Longitude"}.issubset(df.columns):
        return pd.DataFrame()
    lat_ok = df["Latitude"].between(-90, 90, inclusive="both")
    lon_ok = df["Longitude"].between(-180, 180, inclusive="both")
    status = np.where(df["Latitude"].isna() | df["Longitude"].isna(), "missing",
                      np.where(lat_ok & lon_ok, "valid", "invalid"))
    return pd.Series(status, name="coord_status").value_counts().rename_axis("status").to_frame("count")


def pairwise_missingness(df: pd.DataFrame, consider_cols: Optional[list[str]] = None) -> pd.DataFrame:
    if consider_cols is None:
        consider_cols = [c for c in df.columns if c != "filename_index"]
    n = len(df)
    pairs = []
    for a, b in itertools.combinations(consider_cols, 2):
        both_na = (df[a].isna() & df[b].isna()).sum()
        pairs.append((a, b, both_na, (both_na / n * 100) if n else 0.0))
    if not pairs:
        return pd.DataFrame()
    return pd.DataFrame(pairs, columns=["col_a", "col_b", "both_missing", "both_missing_pct"]).sort_values(
        "both_missing_pct", ascending=False)



def safe_bar(series: pd.Series, title: str, xlabel: str, ylabel: str, out: Path, wrap_width: int = 20):
    plt.figure(figsize=(16, 10))  # widen + dynamic height
    # Wrap index labels if they're too long
    labels = [textwrap.fill(str(lbl), wrap_width) for lbl in series.index]
    series.index = labels

    ax = series.plot(kind="bar")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")  # rotate for readability
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def safe_hist(values: pd.Series, title: str, xlabel: str, ylabel: str, out: Path, bins: int = 20):
    plt.figure()
    plt.hist(values.dropna(), bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


# --- Add alongside your other helpers ---

def safe_scatter(x: pd.Series, y: pd.Series, title: str, xlabel: str, ylabel: str, out: Path):
    plt.figure()
    plt.scatter(x, y, s=10)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def value_counts_nomiss(s: pd.Series) -> pd.Series:
    return (
        s.astype("string").str.strip().replace("", pd.NA).dropna().value_counts()
    )


def export_other_plots(df: pd.DataFrame, out_dir: Path, class_col: str, date_col: str):
    """
    Classic plots:
      - class distribution (top 30)
      - habitat counts
      - substrate counts
      - counts per year
      - month-of-year histogram
      - lon/lat scatter
    Handles sparse data gracefully.
    """
    # Class distribution
    if class_col in df.columns:
        class_counts = value_counts_nomiss(df["taxonID_index"])
        if len(class_counts) > 0:
            (out_dir / "class_distribution.csv").write_text(
                class_counts.to_csv(header=["count"]), encoding="utf-8"
            )
            safe_bar(
                class_counts.head(183),
                "Class distribution (taxonID_index) – top 30",
                "Class",
                "Count",
                out=out_dir / "class_distribution_top30.png",
            )

    # Habitat
    if "Habitat" in df.columns:
        habitat_counts = value_counts_nomiss(df["Habitat"])
        if len(habitat_counts) > 0:
            habitat_counts.to_csv(out_dir / "habitat_counts.csv", header=["count"])
            safe_bar(
                habitat_counts,
                "Habitat counts",
                "Habitat",
                "Count",
                out=out_dir / "habitat_counts.png",
            )

    # Substrate
    if "Substrate" in df.columns:
        substrate_counts = value_counts_nomiss(df["Substrate"])
        if len(substrate_counts) > 0:
            substrate_counts.to_csv(out_dir / "substrate_counts.csv", header=["count"])
            safe_bar(
                substrate_counts,
                "Substrate counts",
                "Substrate",
                "Count",
                out=out_dir / "substrate_counts.png",
            )

    # Dates
    if date_col in df.columns:
        years = df[date_col].dt.year.dropna().astype("Int64")
        if len(years) > 0:
            year_counts = years.value_counts().sort_index()
            year_counts.to_csv(out_dir / "year_counts.csv", header=["count"])
            safe_bar(
                year_counts,
                "Counts per year",
                "Year",
                "Count",
                out=out_dir / "year_counts.png",
            )

            months = df[date_col].dt.month.dropna().astype("Int64")
            if len(months) > 0:
                safe_hist(
                    months,
                    "Histogram of events by month (1–12)",
                    "Month",
                    "Count",
                    out=out_dir / "month_hist.png",
                    bins=12,
                )

    # Geo scatter
    if {"Latitude", "Longitude"}.issubset(df.columns):
        valid = df["Latitude"].between(-90, 90) & df["Longitude"].between(-180, 180)
        if valid.any():
            safe_scatter(
                df.loc[valid, "Longitude"],
                df.loc[valid, "Latitude"],
                "Longitude vs Latitude (TRAIN subset)",
                "Longitude",
                "Latitude",
                out=out_dir / "lon_lat_scatter.png",
            )

def plot_class_over_month(df: pd.DataFrame, class_col: str, date_col: str, out: Path, top_classes: int = 10):
    """
    Plot counts per month for the top N classes.
    Always shows months 1–12 on the x-axis.
    """
    # Filter for valid date + class
    valid = df[class_col].notna() & df[date_col].notna()
    data = df.loc[valid, [class_col, date_col]].copy()

    if data.empty:
        print("No valid date/class rows for month plot.")
        return

    data["month"] = data[date_col].dt.month

    # Limit to top N classes by overall count
    top_cls = data[class_col].value_counts().head(top_classes).index
    data = data[data[class_col].isin(top_cls)]

    # Group counts
    month_class_counts = data.groupby(["month", class_col]).size().unstack(fill_value=0)

    # Ensure all 12 months are present, even if missing in data
    month_class_counts = month_class_counts.reindex(range(1, 13), fill_value=0)

    # Plot
    plt.figure(figsize=(12, 6))
    month_class_counts.plot(kind="bar", stacked=False, ax=plt.gca())
    plt.title(f"Counts per month for top {top_classes} classes")
    plt.xlabel("Month")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()



def main():
    ap = argparse.ArgumentParser(description="Sparsity-aware analysis of fungi metadata (defaults to TRAIN split).")
    ap.add_argument("csv", type=Path, help="Path to metadata CSV")
    ap.add_argument("--out", type=Path, default=Path("metadata_report_train"), help="Output folder")
    ap.add_argument("--sep", type=str, default=",", help="CSV delimiter")
    ap.add_argument("--split", choices=["train", "test", "final", "all"], default="train",
                    help="Filter by filename prefix (default: train)")
    ap.add_argument("--id-col", default="filename_index")
    ap.add_argument("--class-col", default="taxonID_index")
    ap.add_argument("--date-col", default="eventDate")
    ap.add_argument("--key-fields", nargs="*", default=CORE_FIELDS_DEFAULT,
                    help="Key fields to audit per class")
    ap.add_argument("--min-nonnull", type=int, default=3,
                    help="Export usable_rows.csv with at least this many non-null fields (excl. filename)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    df = read_metadata(args.csv, date_col=args.date_col, sep=args.sep)
    df = filter_split(df, args.split, id_col=args.id_col)

    # === Missingness reports ===
    miss = basic_missingness(df)
    miss.to_csv(args.out / "missingness_by_column.csv", index=False)

    # Only keep the 2 columns of interest
    # Drop these two from the completeness plot
    cols_to_exclude = {"filename_index", "taxonID_index"}

    comp_series = (
        miss.loc[~miss["column"].isin(cols_to_exclude)]
        .set_index("column")["non_null"]
        .div(len(df))
        .sort_values(ascending=True)
    )

    safe_bar(
        comp_series,
        f"Column completeness (n={len(df)})",
        "Column",
        "Completeness (fraction non-null)",
        args.out / "column_completeness.png"
    )

    # Row completeness + histogram
    row_comp = row_completeness(df, exclude=[args.id_col])
    df.assign(_row_completeness=row_comp).to_csv(args.out / "rows_with_completeness.csv", index=False)
    safe_hist(row_comp, "Row completeness (fraction non-null)", "Fraction non-null", "Rows",
              args.out / "row_completeness_hist.png", bins=20)

    # Rows that are basically empty (except filename)
    empty_rows = df[row_comp == 0.0]
    if len(empty_rows) > 0:
        empty_rows.to_csv(args.out / "rows_all_missing_except_filename.csv", index=False)

    # Per-class coverage of key fields
    if args.class_col in df.columns:
        per_cls = per_class_coverage(df, class_col=args.class_col, key_fields=args.key_fields)
        per_cls.to_csv(args.out / "per_class_coverage.csv")

        # Also save a plain class count for quick eyeballing
        df[args.class_col].fillna("⟂(missing)").value_counts().to_csv(args.out / "class_counts.csv", header=["count"])

    # Coordinates sanity
    cq = coord_quality(df)
    if not cq.empty:
        cq.to_csv(args.out / "coordinate_quality.csv")

    # Invalid coordinates dump
    if {"Latitude", "Longitude"}.issubset(df.columns):
        bad = df[
            ~df["Latitude"].between(-90, 90, inclusive="both") | ~df["Longitude"].between(-180, 180, inclusive="both")]
        if len(bad) > 0:
            bad.to_csv(args.out / "invalid_coordinates_rows.csv", index=False)

    # Pairwise missingness (which fields tend to be missing together)
    consider_cols = [c for c in df.columns if c not in [args.id_col]]
    pm = pairwise_missingness(df, consider_cols=consider_cols)
    if not pm.empty:
        pm.to_csv(args.out / "pairwise_missingness.csv", index=False)

    # Usable subset export (at least N non-null fields)
    usable_mask = row_comp >= (args.min_nonnull / max(1, (df.shape[1] - 1)))
    usable = df[usable_mask]
    usable.to_csv(args.out / "usable_rows.csv", index=False)

    # Short console echo
    print("✅ Analysis complete")
    print(f"• Split: {args.split.upper()}  • Rows: {len(df):,}")
    print(f"• Output: {args.out.resolve()}")
    print(f"• Usable rows (≥{args.min_nonnull} non-null fields): {len(usable):,}")

    export_other_plots(df, args.out, class_col=args.class_col, date_col=args.date_col)

    plot_class_over_month(
        df,
        class_col=args.class_col,
        date_col=args.date_col,
        out=args.out / "class_over_month.png",
        top_classes=10  # adjust as needed
    )

if __name__ == "__main__":
    main()
