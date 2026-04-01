from __future__ import annotations

import json
import os
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

DEFAULT_K = 5
MAX_K = 20
MAX_HISTORY = 50
ONLINE_PREFIX_LIMIT = 2


class ListenEvent(BaseModel):
    user_id: int = Field(..., ge=1)
    track_id: int = Field(..., ge=1)


class EventResponse(BaseModel):
    status: str
    user_id: int
    track_id: int
    history_size: int


class RecommendationItem(BaseModel):
    track_id: int
    score: float
    rank: int
    source: str
    source_track_id: int | None = None
    track_name: str
    artist_names: list[str]
    genre_names: list[str]


class RecommendationsResponse(BaseModel):
    user_id: int
    k: int
    online_history_size: int
    recommendations: list[RecommendationItem]


class HealthResponse(BaseModel):
    status: str
    artifact_dir: str
    item_count: int
    ranked_user_count: int
    similar_track_count: int
    metrics_available: list[str]


@dataclass
class Candidate:
    track_id: int
    score: float
    source: str
    source_track_id: int | None = None


@dataclass
class RecommendationStore:
    artifact_dir: Path
    items: pd.DataFrame
    ranked: pd.DataFrame
    personal: pd.DataFrame
    top_popular: pd.DataFrame
    similar: pd.DataFrame
    global_top_popular: pd.DataFrame
    metrics: dict[str, Any]
    online_history: dict[int, deque[int]] = field(
        default_factory=lambda: defaultdict(lambda: deque(maxlen=MAX_HISTORY))
    )

    @classmethod
    def load(cls, artifact_dir: str | Path) -> "RecommendationStore":
        artifact_dir = Path(artifact_dir)
        required_files = [
            "items.parquet",
            "top_popular.parquet",
            "personal_als.parquet",
            "similar.parquet",
            "recommendations.parquet",
            "metrics.json",
        ]
        missing = [name for name in required_files if not (artifact_dir / name).exists()]
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise FileNotFoundError(
                f"Missing Part 1 artifacts in {artifact_dir}: {missing_str}. "
                "Generate them before starting the service."
            )

        items = (
            pd.read_parquet(artifact_dir / "items.parquet")
            .sort_values("track_id")
            .set_index("track_id")
        )

        ranked = (
            pd.read_parquet(artifact_dir / "recommendations.parquet")
            .sort_values(["user_id", "rank"])
            .set_index("user_id")
        )
        personal = (
            pd.read_parquet(artifact_dir / "personal_als.parquet")
            .sort_values(["user_id", "rank"])
            .set_index("user_id")
        )

        top_popular_raw = pd.read_parquet(artifact_dir / "top_popular.parquet").sort_values(
            ["user_id", "rank"]
        )
        top_popular = top_popular_raw.set_index("user_id")
        global_top_popular = (
            top_popular_raw[["track_id", "score"]]
            .groupby("track_id", as_index=False)["score"]
            .max()
            .sort_values(["score", "track_id"], ascending=[False, True])
            .reset_index(drop=True)
        )
        global_top_popular["rank"] = global_top_popular.index + 1

        similar = (
            pd.read_parquet(artifact_dir / "similar.parquet")
            .sort_values(["track_id", "rank"])
            .set_index("track_id")
        )

        metrics = json.loads((artifact_dir / "metrics.json").read_text(encoding="utf-8"))

        return cls(
            artifact_dir=artifact_dir,
            items=items,
            ranked=ranked,
            personal=personal,
            top_popular=top_popular,
            similar=similar,
            global_top_popular=global_top_popular,
            metrics=metrics,
        )

    def health(self) -> HealthResponse:
        return HealthResponse(
            status="ok",
            artifact_dir=str(self.artifact_dir),
            item_count=int(len(self.items)),
            ranked_user_count=int(self.ranked.index.nunique()),
            similar_track_count=int(self.similar.index.nunique()),
            metrics_available=sorted(self.metrics.keys()),
        )

    def register_event(self, user_id: int, track_id: int) -> EventResponse:
        if track_id not in self.items.index:
            raise HTTPException(status_code=404, detail=f"Unknown track_id: {track_id}")

        history = self.online_history[user_id]
        history.append(track_id)
        return EventResponse(
            status="accepted",
            user_id=user_id,
            track_id=track_id,
            history_size=len(history),
        )

    def build_recommendations(self, user_id: int, k: int) -> RecommendationsResponse:
        history = self.online_history.get(user_id, deque())
        consumed_track_ids = set(history)

        online_candidates = self._collect_online_candidates(user_id, consumed_track_ids)
        offline_candidates = self._collect_offline_candidates(user_id, consumed_track_ids)

        results: list[RecommendationItem] = []
        seen_track_ids = set(consumed_track_ids)
        online_idx = 0
        offline_idx = 0
        online_used = 0
        online_quota = min(ONLINE_PREFIX_LIMIT, len(online_candidates))

        while len(results) < k:
            take_online = (
                online_idx < len(online_candidates)
                and online_used < online_quota
                and (len(results) % 2 == 0 or offline_idx >= len(offline_candidates))
            )

            candidate: Candidate | None = None
            if take_online:
                candidate = online_candidates[online_idx]
                online_idx += 1
                online_used += 1
            elif offline_idx < len(offline_candidates):
                candidate = offline_candidates[offline_idx]
                offline_idx += 1
            elif online_idx < len(online_candidates):
                candidate = online_candidates[online_idx]
                online_idx += 1
            else:
                break

            if candidate.track_id in seen_track_ids:
                continue

            seen_track_ids.add(candidate.track_id)
            results.append(self._candidate_to_response(candidate, rank=len(results) + 1))

        return RecommendationsResponse(
            user_id=user_id,
            k=k,
            online_history_size=len(history),
            recommendations=results,
        )

    def _collect_online_candidates(
        self,
        user_id: int,
        consumed_track_ids: set[int],
    ) -> list[Candidate]:
        history = self.online_history.get(user_id, deque())
        if not history:
            return []

        candidates: list[Candidate] = []
        seen_track_ids = set(consumed_track_ids)

        for source_track_id in reversed(history):
            similar_rows = self._rows_for_index(self.similar, source_track_id)
            for row in similar_rows.itertuples(index=False):
                track_id = int(row.similar_track_id)
                if track_id in seen_track_ids:
                    continue

                seen_track_ids.add(track_id)
                candidates.append(
                    Candidate(
                        track_id=track_id,
                        score=float(row.score),
                        source="online_similar",
                        source_track_id=int(source_track_id),
                    )
                )
        return candidates

    def _collect_offline_candidates(
        self,
        user_id: int,
        consumed_track_ids: set[int],
    ) -> list[Candidate]:
        candidates: list[Candidate] = []
        seen_track_ids = set(consumed_track_ids)

        for source_name, rows in (
            ("offline_ranked", self._rows_for_index(self.ranked, user_id)),
            ("offline_personal", self._rows_for_index(self.personal, user_id)),
            ("top_popular", self._top_popular_rows_for_user(user_id)),
        ):
            for row in rows.itertuples(index=False):
                track_id = int(row.track_id)
                if track_id in seen_track_ids:
                    continue

                seen_track_ids.add(track_id)
                candidates.append(
                    Candidate(
                        track_id=track_id,
                        score=float(row.score),
                        source=source_name,
                    )
                )
        return candidates

    def _top_popular_rows_for_user(self, user_id: int) -> pd.DataFrame:
        user_rows = self._rows_for_index(self.top_popular, user_id)
        if not user_rows.empty:
            return user_rows
        return self.global_top_popular

    @staticmethod
    def _rows_for_index(frame: pd.DataFrame, index_value: int) -> pd.DataFrame:
        if index_value not in frame.index:
            return frame.iloc[0:0].reset_index(drop=True)

        rows = frame.loc[index_value]
        if isinstance(rows, pd.Series):
            rows = rows.to_frame().T
        return rows.reset_index(drop=True)

    def _candidate_to_response(self, candidate: Candidate, rank: int) -> RecommendationItem:
        track_id = int(candidate.track_id)
        track_name = ""
        artist_names: list[str] = []
        genre_names: list[str] = []

        if track_id in self.items.index:
            item_row = self.items.loc[track_id]
            track_name = str(item_row.get("track_name", ""))
            artist_names = self._list_value(item_row.get("artist_names"))
            genre_names = self._list_value(item_row.get("genre_names"))

        return RecommendationItem(
            track_id=track_id,
            score=float(candidate.score),
            rank=rank,
            source=candidate.source,
            source_track_id=candidate.source_track_id,
            track_name=track_name,
            artist_names=artist_names,
            genre_names=genre_names,
        )

    @staticmethod
    def _list_value(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value]
        if hasattr(value, "tolist"):
            return [str(item) for item in value.tolist()]
        return []


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifact_dir = Path(os.getenv("RECSYS_DATA_DIR", Path(__file__).resolve().parent))
    app.state.store = RecommendationStore.load(artifact_dir)
    yield


app = FastAPI(
    title="Music Recommendations Service",
    version="0.1.0",
    lifespan=lifespan,
)


def get_store(request: Request) -> RecommendationStore:
    return request.app.state.store


@app.get("/health", response_model=HealthResponse)
def healthcheck(request: Request) -> HealthResponse:
    return get_store(request).health()


@app.post("/events", response_model=EventResponse)
def record_event(event: ListenEvent, request: Request) -> EventResponse:
    return get_store(request).register_event(event.user_id, event.track_id)


@app.get("/recommendations", response_model=RecommendationsResponse)
def get_recommendations(
    request: Request,
    user_id: int = Query(..., ge=1),
    k: int = Query(DEFAULT_K, ge=1, le=MAX_K),
) -> RecommendationsResponse:
    return get_store(request).build_recommendations(user_id=user_id, k=k)
