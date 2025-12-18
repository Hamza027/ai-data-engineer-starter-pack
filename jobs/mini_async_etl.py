from __future__ import annotations

import asyncio
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter

import httpx


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    return logging.getLogger("mini_async_etl")


@dataclass(frozen=True)
class Job:
    base: str


async def fetch_one(client: httpx.AsyncClient, job: Job, sem: asyncio.Semaphore, log: logging.Logger) -> dict:
    url = f"https://api.frankfurter.app/latest?from={job.base}"
    async with sem:
        r = await client.get(url, timeout=8.0)
        r.raise_for_status()
        data = r.json()
        log.info("Fetched %s", job.base)
        return {"base": job.base, "payload": data}


def cpu_enrich(record: dict) -> dict:
    """
    Pretend CPU-heavy transform:
    - Take the payload
    - Create a stable fingerprint hash of it
    - Return a smaller cleaned structure
    """
    payload = record["payload"]
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")

    # do some extra hashing to simulate heavier CPU work
    h = raw
    for _ in range(150_000):
        h = sha256(h).digest()

    fingerprint = sha256(h).hexdigest()[:20]

    rates = payload.get("rates", {})
    pick = {k: rates.get(k) for k in ["EUR", "USD", "GBP"] if k in rates}

    return {
        "base": record["base"],
        "date": payload.get("date"),
        "picked_rates": pick,
        "fingerprint": fingerprint,
    }


async def main() -> None:
    log = setup_logger()

    jobs = [Job("GBP"), Job("EUR"), Job("USD"), Job("JPY"), Job("AUD"), Job("CAD"), Job("SEK"), Job("CHF")]

    t0 = perf_counter()

    # 1) async fetch with a concurrency limit
    sem = asyncio.Semaphore(4)
    async with httpx.AsyncClient() as client:
        fetched = await asyncio.gather(*[fetch_one(client, j, sem, log) for j in jobs])

    t_fetch = perf_counter() - t0
    log.info("Fetch phase done in %.2fs", t_fetch)

    # 2) CPU transform using multiprocessing
    t1 = perf_counter()
    workers = max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        enriched = list(ex.map(cpu_enrich, fetched))

    t_cpu = perf_counter() - t1
    log.info("CPU phase done in %.2fs using %s workers", t_cpu, workers)

    # 3) Write output
    out = {
        "rows": len(enriched),
        "timings_sec": {"fetch": round(t_fetch, 3), "cpu": round(t_cpu, 3)},
        "data": enriched,
    }

    with open("out/mini_async_etl.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    log.info("Wrote out/mini_async_etl.json")


if __name__ == "__main__":
    asyncio.run(main())
