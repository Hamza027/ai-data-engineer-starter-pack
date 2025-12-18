from __future__ import annotations

import os
from typing import Any

import psycopg


def pg_dsn() -> str:
    dsn = os.getenv("PG_DSN")
    if not dsn:
        raise RuntimeError("PG_DSN env var is required, e.g. postgresql://app:app@localhost:5432/foundations")
    return dsn


def connect():
    # Fail fast instead of hanging
    return psycopg.connect(pg_dsn(), connect_timeout=3)


def init_db() -> None:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE SCHEMA IF NOT EXISTS ai_pipeline_metadata;

                CREATE TABLE IF NOT EXISTS ai_pipeline_metadata.pipeline_runs (
                  run_id UUID PRIMARY KEY,
                  ts_utc TIMESTAMPTZ NOT NULL,
                  pipeline_name TEXT NOT NULL,
                  status TEXT NOT NULL,
                  country TEXT NOT NULL,
                  min_amount NUMERIC(12,2) NOT NULL,
                  vat_rate NUMERIC(6,4) NOT NULL,
                  rows_in INT NOT NULL,
                  rows_out INT NOT NULL,
                  amount_sum NUMERIC(12,2) NOT NULL,
                  vat_sum NUMERIC(12,2) NOT NULL,
                  total_with_vat_sum NUMERIC(12,2) NOT NULL,
                  error_message TEXT
                );

                CREATE TABLE IF NOT EXISTS ai_pipeline_metadata.pipeline_artifacts (
                  artifact_id UUID PRIMARY KEY,
                  run_id UUID NOT NULL REFERENCES ai_pipeline_metadata.pipeline_runs(run_id) ON DELETE CASCADE,
                  ts_utc TIMESTAMPTZ NOT NULL,
                  artifact_type TEXT NOT NULL,
                  path TEXT NOT NULL,
                  bytes BIGINT
                );
                """
            )
        conn.commit()


def insert_run(run: dict[str, Any]) -> None:
    sql = """
    INSERT INTO ai_pipeline_metadata.pipeline_runs
      (run_id, ts_utc, pipeline_name, status, country, min_amount, vat_rate,
       rows_in, rows_out, amount_sum, vat_sum, total_with_vat_sum, error_message)
    VALUES
      (%(run_id)s::uuid, %(ts_utc)s::timestamptz, %(pipeline_name)s, %(status)s, %(country)s,
       %(min_amount)s::numeric, %(vat_rate)s::numeric, %(rows_in)s::int, %(rows_out)s::int,
       %(amount_sum)s::numeric, %(vat_sum)s::numeric, %(total_with_vat_sum)s::numeric, %(error_message)s)
    """
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, run)
        conn.commit()


def list_runs(limit: int = 10) -> list[dict[str, Any]]:
    sql = """
    SELECT run_id::text, ts_utc, pipeline_name, status, country,
           min_amount, vat_rate, rows_in, rows_out, amount_sum, vat_sum, total_with_vat_sum, error_message
    FROM ai_pipeline_metadata.pipeline_runs
    ORDER BY ts_utc DESC
    LIMIT %s
    """
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (limit,))
            rows = cur.fetchall()
            cols = [d.name for d in cur.description]

    out = []
    for r in rows:
        item = dict(zip(cols, r))
        item["ts_utc"] = item["ts_utc"].isoformat()
        for k in ["min_amount", "vat_rate", "amount_sum", "vat_sum", "total_with_vat_sum"]:
            item[k] = str(item[k])
        out.append(item)
    return out
