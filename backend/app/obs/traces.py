"""SQLite-backed query trace store for observability & the Insights dashboard."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..schemas import StatsSummary

_SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    query_id       TEXT PRIMARY KEY,
    ts             TEXT NOT NULL,
    query          TEXT NOT NULL,
    refused        INTEGER NOT NULL,
    answer         TEXT NOT NULL,
    latency_ms     REAL NOT NULL,
    first_token_ms REAL,
    grounding_score REAL NOT NULL,
    cited_sections TEXT NOT NULL,   -- JSON list
    retrieval      TEXT NOT NULL,   -- JSON list of hit dicts
    timings        TEXT NOT NULL    -- JSON list of {stage, ms}
);
CREATE INDEX IF NOT EXISTS idx_traces_ts ON traces(ts);
"""


class TraceStore:
    def __init__(self, db_path: Path) -> None:
        self._local = threading.local()
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn = conn
        return conn

    def new_query_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def record(
        self,
        *,
        query_id: str,
        query: str,
        refused: bool,
        answer: str,
        latency_ms: float,
        first_token_ms: float | None,
        grounding_score: float,
        cited_sections: list[str],
        retrieval: list[dict[str, Any]],
        timings: list[dict[str, Any]],
    ) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO traces VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    query_id,
                    datetime.now(UTC).isoformat(timespec="seconds"),
                    query,
                    int(refused),
                    answer,
                    latency_ms,
                    first_token_ms,
                    grounding_score,
                    json.dumps(cited_sections),
                    json.dumps(retrieval),
                    json.dumps(timings),
                ),
            )

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT * FROM traces ORDER BY ts DESC LIMIT ?", (limit,)
            )
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
        out = []
        for r in rows:
            d = dict(zip(cols, r, strict=True))
            d["cited_sections"] = json.loads(d["cited_sections"])
            d["retrieval"] = json.loads(d["retrieval"])
            d["timings"] = json.loads(d["timings"])
            out.append(d)
        return out

    def stats(self) -> StatsSummary:
        with self._conn() as c:
            row = c.execute(
                """SELECT COUNT(*),
                          AVG(refused),
                          AVG(latency_ms)
                   FROM traces"""
            ).fetchone()
            total, refusal_rate, avg_latency = row
            if not total:
                return StatsSummary(total_traces=0, refusal_rate=0.0, p50_latency_ms=0,
                                    p95_latency_ms=0)
            lat = [r[0] for r in c.execute("SELECT latency_ms FROM traces").fetchall()]
            ft = [r[0] for r in c.execute(
                "SELECT first_token_ms FROM traces WHERE first_token_ms IS NOT NULL"
            ).fetchall()]
            lat.sort()
            top_cited: dict[str, int] = {}
            for (js,) in c.execute("SELECT cited_sections FROM traces"):
                for s in json.loads(js):
                    top_cited[s] = top_cited.get(s, 0) + 1
            top = sorted(top_cited.items(), key=lambda kv: -kv[1])[:10]

        def pct(p: float) -> float:
            return lat[min(len(lat) - 1, int(p * len(lat)))]

        return StatsSummary(
            total_traces=total,
            refusal_rate=round(refusal_rate or 0.0, 4),
            p50_latency_ms=round(pct(0.50), 1),
            p95_latency_ms=round(pct(0.95), 1),
            avg_first_token_ms=round(sum(ft) / len(ft), 1) if ft else None,
            top_cited=[{"section_id": s, "count": n} for s, n in top],
        )


class Timer:
    """Context manager that appends {stage, ms} timings to a list."""

    def __init__(self, stage: str, timings: list[dict[str, Any]]) -> None:
        self.stage = stage
        self.timings = timings
        self.t0 = 0.0

    def __enter__(self) -> Timer:
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        ms = (time.perf_counter() - self.t0) * 1000
        self.timings.append({"stage": self.stage, "ms": round(ms, 1)})
