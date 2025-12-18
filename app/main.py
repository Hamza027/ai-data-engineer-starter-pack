from __future__ import annotations

import os
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.pg_log import init_db, insert_run, list_runs
from jobs.etl_sales import clean_amount, run_etl, setup_logger


app = FastAPI(title="AI Data Engineer Starter Pack API", version="1.0.0")
log = setup_logger()

LAST_RUN: Optional[dict] = None

PIPELINE_NAME = "sales_etl"


def input_csv_path() -> Path:
    return Path(os.getenv("INPUT_CSV_PATH", "data/sales.csv"))


def out_dir() -> Path:
    return Path(os.getenv("OUT_DIR", "out"))


class RunETLRequest(BaseModel):
    country: str = Field(default="UK", min_length=2, max_length=32)
    min_amount: str = Field(default="0.00")
    vat_rate: str = Field(default="0.20")


class RunETLResponse(BaseModel):
    success: bool
    run_id: str
    summary: dict
    timestamp: str


@app.on_event("startup")
def startup() -> None:
    init_db()
    log.info("Postgres connectivity OK")
    log.info(
        "Config INPUT_CSV_PATH=%s OUT_DIR=%s PG_DSN=%s",
        input_csv_path(),
        out_dir(),
        os.getenv("PG_DSN", "<missing>"),
    )


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "ai-data-engineer-starter-pack", "time": datetime.now(timezone.utc).isoformat()}


@app.get("/config")
def config() -> dict:
    return {
        "INPUT_CSV_PATH": str(input_csv_path()),
        "OUT_DIR": str(out_dir()),
        "PG_DSN": "set" if os.getenv("PG_DSN") else "missing",
        "PIPELINE_NAME": PIPELINE_NAME,
    }


@app.get("/runs")
def runs(limit: int = 10) -> dict:
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100")
    return {"limit": limit, "runs": list_runs(limit=limit)}


@app.post("/run-etl", response_model=RunETLResponse)
def run_etl_endpoint(req: RunETLRequest) -> RunETLResponse:
    global LAST_RUN

    ts = datetime.now(timezone.utc).isoformat()
    run_id = str(uuid4())

    try:
        min_amount_dec = clean_amount(req.min_amount)
        vat_rate = Decimal(req.vat_rate)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid numeric input: {e}")

    base_out = out_dir()
    out_csv = base_out / "sales_clean.csv"
    out_json = base_out / "summary.json"

    status = "success"
    error_message = None
    summary = {}

    try:
        summary = run_etl(
            input_path=input_csv_path(),
            out_csv=out_csv,
            out_json=out_json,
            country=req.country,
            min_amount=min_amount_dec,
            vat_rate=vat_rate,
            log=log,
        )
    except FileNotFoundError as e:
        status = "failed"
        error_message = str(e)
        raise HTTPException(status_code=404, detail=error_message)
    except Exception as e:
        status = "failed"
        error_message = str(e)
        log.exception("ETL failed")
        raise HTTPException(status_code=500, detail=f"ETL failed: {e}")
    finally:
        run_row = {
            "run_id": run_id,
            "ts_utc": ts,
            "pipeline_name": PIPELINE_NAME,
            "status": status,
            "country": req.country,
            "min_amount": str(min_amount_dec),
            "vat_rate": str(vat_rate),
            "rows_in": int(summary.get("rows_in", 0)),
            "rows_out": int(summary.get("rows_out", 0)),
            "amount_sum": str(summary.get("amount_sum", "0.00")),
            "vat_sum": str(summary.get("vat_sum", "0.00")),
            "total_with_vat_sum": str(summary.get("total_with_vat_sum", "0.00")),
            "error_message": error_message,
        }
        try:
            insert_run(run_row)
        except Exception as e:
            log.error("Failed to write run log to Postgres: %s", e)

    LAST_RUN = {"timestamp": ts, "run_id": run_id, "request": req.model_dump(), "summary": summary}

    return RunETLResponse(success=True, run_id=run_id, summary=summary, timestamp=ts)
