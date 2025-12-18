from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    return logging.getLogger("etl_sales")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simple sales ETL (clean, filter, enrich)")
    p.add_argument("--input", default="data/sales.csv", help="Input CSV path")
    p.add_argument("--out_csv", default="out/sales_clean.csv", help="Output CSV path")
    p.add_argument("--out_json", default="out/summary.json", help="Output summary JSON path")
    p.add_argument("--country", default="UK", help="Country to filter on")
    p.add_argument("--min_amount", type=float, default=0.0, help="Minimum amount to keep")
    p.add_argument("--vat_rate", type=float, default=0.2, help="VAT rate, e.g. 0.2 for 20%%")
    return p.parse_args()


def clean_amount(x) -> float:
    # handles values like "£120.50" or "120.50"
    if pd.isna(x):
        return Decimal("0.00")
    s = str(x).strip().replace("£","").replace(",","")
    d = Decimal(s)
    return d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

def q2(x: Decimal) -> Decimal:
    return x.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

def run_etl(
    input_path: Path,
    out_csv: Path,
    out_json: Path,
    country: str,
    min_amount: Decimal,
    vat_rate: Decimal,
    log: logging.Logger,
) -> dict:
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    df = pd.read_csv(input_path)
    log.info("Read %s rows from %s", len(df), input_path)

    required = {"order_id", "country", "amount"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["amount_dec"] = df["amount"].apply(clean_amount)

    df_f = df[(df["country"] == country) & (df["amount_dec"] >= min_amount)].copy()
    log.info("Filtered to %s rows for country=%s min_amount=%s", len(df_f), country, min_amount)

    # Decimal-safe money calculations
    df_f["vat_dec"] = df_f["amount_dec"].apply(lambda a: q2(a * vat_rate))
    df_f["total_with_vat_dec"] = df_f.apply(lambda r: q2(r["amount_dec"] + r["vat_dec"]), axis=1)

    # For CSV output, write numeric columns as 2dp strings (simple + stable)
    out_df = df_f[["order_id", "country", "amount_dec", "vat_dec", "total_with_vat_dec"]].copy()
    out_df.rename(
        columns={
            "amount_dec": "amount",
            "vat_dec": "vat",
            "total_with_vat_dec": "total_with_vat",
        },
        inplace=True,
    )
    for col in ["amount", "vat", "total_with_vat"]:
        out_df[col] = out_df[col].apply(lambda d: f"{d:.2f}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    log.info("Wrote cleaned CSV to %s", out_csv)

    amount_sum = q2(df_f["amount_dec"].sum())
    vat_sum = q2(df_f["vat_dec"].sum())
    total_sum = q2(df_f["total_with_vat_dec"].sum())

    summary = {
        "country": country,
        "min_amount": str(q2(min_amount)),
        "vat_rate": str(vat_rate),
        "rows_in": int(len(df)),
        "rows_out": int(len(df_f)),
        "amount_sum": str(amount_sum),
        "vat_sum": str(vat_sum),
        "total_with_vat_sum": str(total_sum),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Wrote summary JSON to %s", out_json)

    return summary


def main() -> None:
    log = setup_logger()
    args = parse_args()

    summary = run_etl(
        input_path=Path(args.input),
        out_csv=Path(args.out_csv),
        out_json=Path(args.out_json),
        country=args.country,
        min_amount=clean_amount(args.min_amount),
        vat_rate=clean_amount(args.vat_rate),
        log=log,
    )

    log.info("Done. Summary: %s", summary)


if __name__ == "__main__":
    main()
