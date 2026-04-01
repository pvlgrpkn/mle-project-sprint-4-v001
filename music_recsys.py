from __future__ import annotations

import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import boto3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from implicit.als import AlternatingLeastSquares
from scipy import sparse


DEFAULT_TOP_K = 5
DEFAULT_CANDIDATE_K = 5
TRAIN_CUTOFF = pd.Timestamp("2022-12-16")
RANKER_VALID_CUTOFF = pd.Timestamp("2022-12-09")
ALS_FACTORS = 24
ALS_ITERATIONS = 4
ALS_REGULARIZATION = 0.08
ALS_RANDOM_STATE = 42
RANKER_TRAIN_MAX_USERS = 150_000


@dataclass
class S3Config:
    bucket: str | None = None
    endpoint_url: str | None = None
    access_key_id: str | None = None
    secret_access_key: str | None = None

    @classmethod
    def from_env(cls) -> "S3Config":
        return cls(
            bucket=os.getenv("RECSYS_S3_BUCKET"),
            endpoint_url=os.getenv("AWS_ENDPOINT_URL"),
            access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )

    @property
    def enabled(self) -> bool:
        return bool(self.bucket and self.access_key_id and self.secret_access_key)


@dataclass
class MatrixArtifacts:
    user_item: sparse.csr_matrix
    user_ids: np.ndarray
    user_to_row: pd.Series
    track_ids: np.ndarray
    item_to_col: pd.Series
    track_popularity: pd.Series
    track_user_popularity: pd.Series
    user_history_len: pd.Series


def chunked(values: Iterable[Any], chunk_size: int) -> Iterable[list[Any]]:
    batch: list[Any] = []

    for value in values:
        batch.append(value)

        if len(batch) >= chunk_size:
            yield batch
            batch = []

    if batch:
        yield batch


def normalize_id_list(value: Any) -> list[int]:
    if value is None:
        return []

    if isinstance(value, np.ndarray):
        return value.astype("int64").tolist()

    if isinstance(value, list):
        return [int(v) for v in value]

    return [int(v) for v in value]


def map_names(values: list[int], mapping: dict[int, str]) -> list[str]:
    return [mapping.get(value, f"unknown_{value}") for value in values]


def load_raw_data(
    data_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    tracks = pd.read_parquet(data_dir / "tracks.parquet")
    catalog_names = pd.read_parquet(data_dir / "catalog_names.parquet")
    interactions = pd.read_parquet(data_dir / "interactions.parquet")
    return tracks, catalog_names, interactions


def inspect_raw_data(
    tracks: pd.DataFrame,
    catalog_names: pd.DataFrame,
    interactions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    catalog_by_type = {
        name_type: set(frame["id"].astype("int64").tolist())
        for name_type, frame in catalog_names.groupby("type", sort=False)
    }

    unknown_counts: dict[str, int] = {}
    missing_examples: dict[str, list[int]] = {}

    for column, name_type in (("albums", "album"), ("artists", "artist"), ("genres", "genre")):
        exploded = tracks[["track_id", column]].explode(column)
        exploded = exploded.dropna(subset=[column]).copy()
        exploded[column] = exploded[column].astype("int64")
        missing = exploded.loc[~exploded[column].isin(catalog_by_type[name_type]), column]
        unknown_counts[f"unknown_{column}_rows"] = int(len(missing))
        unknown_counts[f"unknown_{column}_ids"] = int(missing.nunique())
        missing_examples[f"unknown_{column}_examples"] = sorted(missing.unique().tolist())[:10]

    summary = pd.DataFrame(
        [
            {
                "table": "tracks",
                "rows": int(len(tracks)),
                "unique_track_id": int(tracks["track_id"].nunique()),
                "date_min": pd.NaT,
                "date_max": pd.NaT,
            },
            {
                "table": "catalog_names",
                "rows": int(len(catalog_names)),
                "unique_track_id": pd.NA,
                "date_min": pd.NaT,
                "date_max": pd.NaT,
            },
            {
                "table": "interactions",
                "rows": int(len(interactions)),
                "unique_track_id": int(interactions["track_id"].nunique()),
                "date_min": interactions["started_at"].min(),
                "date_max": interactions["started_at"].max(),
            },
        ]
    )

    details = {
        "track_id_dtype": str(tracks["track_id"].dtype),
        "catalog_id_dtype": str(catalog_names["id"].dtype),
        "user_id_dtype": str(interactions["user_id"].dtype),
        "track_seq_dtype": str(interactions["track_seq"].dtype),
        "started_at_dtype": str(interactions["started_at"].dtype),
        "track_name_missing": int((~tracks["track_id"].isin(catalog_by_type["track"])).sum()),
        "empty_albums": int(tracks["albums"].map(len).eq(0).sum()),
        "empty_artists": int(tracks["artists"].map(len).eq(0).sum()),
        "empty_genres": int(tracks["genres"].map(len).eq(0).sum()),
        "non_positive_track_seq": int((interactions["track_seq"] <= 0).sum()),
        "null_started_at": int(interactions["started_at"].isna().sum()),
        "user_count": int(interactions["user_id"].nunique()),
        "track_count_in_events": int(interactions["track_id"].nunique()),
    }
    details.update(unknown_counts)
    details.update(missing_examples)

    return summary, details


def build_items(
    tracks: pd.DataFrame,
    catalog_names: pd.DataFrame,
) -> tuple[pd.DataFrame, list[int]]:
    tracks = tracks.copy()
    catalog_names = catalog_names.copy()

    for column in ("albums", "artists", "genres"):
        tracks[column] = tracks[column].map(normalize_id_list)

    known_genres = set(catalog_names.loc[catalog_names["type"] == "genre", "id"].astype("int64"))
    all_genres = {
        genre_id
        for values in tracks["genres"]
        for genre_id in values
    }
    missing_genres = sorted(all_genres - known_genres)

    if missing_genres:
        catalog_names = pd.concat(
            [
                catalog_names,
                pd.DataFrame(
                    {
                        "id": missing_genres,
                        "type": "genre",
                        "name": [f"unknown_genre_{genre_id}" for genre_id in missing_genres],
                    }
                ),
            ],
            ignore_index=True,
        )

    lookup = {
        name_type: frame.set_index("id")["name"].to_dict()
        for name_type, frame in catalog_names.groupby("type", sort=False)
    }

    items = tracks.copy()
    items["track_name"] = items["track_id"].map(lookup["track"])
    items["album_names"] = items["albums"].map(lambda values: map_names(values, lookup["album"]))
    items["artist_names"] = items["artists"].map(lambda values: map_names(values, lookup["artist"]))
    items["genre_names"] = items["genres"].map(lambda values: map_names(values, lookup["genre"]))
    items["album_count"] = items["albums"].map(len).astype("int16")
    items["artist_count"] = items["artists"].map(len).astype("int16")
    items["genre_count"] = items["genres"].map(len).astype("int16")

    columns = [
        "track_id",
        "track_name",
        "albums",
        "artists",
        "genres",
        "album_names",
        "artist_names",
        "genre_names",
        "album_count",
        "artist_count",
        "genre_count",
    ]
    return items[columns], missing_genres


def plot_user_listen_distribution(
    events: pd.DataFrame,
    max_users: int = 300_000,
) -> pd.DataFrame:
    distribution = (
        events.groupby("user_id", sort=False)
        .size()
        .rename("listen_count")
        .reset_index()
    )

    sample = distribution
    if len(distribution) > max_users:
        sample = distribution.sample(max_users, random_state=ALS_RANDOM_STATE)

    plt.figure(figsize=(10, 5))
    sns.histplot(sample["listen_count"], bins=50, log_scale=(True, True))
    plt.title("Распределение количества прослушиваний на пользователя")
    plt.xlabel("Количество прослушиваний")
    plt.ylabel("Количество пользователей")
    plt.show()
    return distribution


def get_top_tracks(
    events: pd.DataFrame,
    items: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    popularity = (
        events.groupby("track_id", sort=False)
        .size()
        .rename("listen_count")
        .reset_index()
        .sort_values("listen_count", ascending=False)
        .head(top_n)
    )

    return popularity.merge(
        items[["track_id", "track_name", "artist_names", "genre_names"]],
        on="track_id",
        how="left",
    )


def get_top_genres(
    events: pd.DataFrame,
    items: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    track_popularity = (
        events.groupby("track_id", sort=False)
        .size()
        .rename("listen_count")
        .reset_index()
    )

    exploded = (
        track_popularity.merge(items[["track_id", "genre_names"]], on="track_id", how="left")
        .explode("genre_names")
        .dropna(subset=["genre_names"])
    )

    return (
        exploded.groupby("genre_names", sort=False)["listen_count"]
        .sum()
        .rename("listen_count")
        .reset_index()
        .sort_values("listen_count", ascending=False)
        .head(top_n)
    )


def get_unheard_tracks(
    events: pd.DataFrame,
    items: pd.DataFrame,
    sample_size: int = 20,
) -> tuple[int, pd.DataFrame]:
    heard_tracks = pd.Index(events["track_id"].unique())
    unheard = items.loc[~items["track_id"].isin(heard_tracks), ["track_id", "track_name", "artist_names"]]
    return int(len(unheard)), unheard.head(sample_size)


def save_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def maybe_upload_to_s3(local_path: str | Path, object_key: str, s3_config: S3Config | None = None) -> str | None:
    s3_config = s3_config or S3Config.from_env()

    if not s3_config.enabled:
        return None

    session = boto3.session.Session()
    client = session.client(
        "s3",
        endpoint_url=s3_config.endpoint_url,
        aws_access_key_id=s3_config.access_key_id,
        aws_secret_access_key=s3_config.secret_access_key,
    )
    client.upload_file(str(local_path), s3_config.bucket, object_key)
    return f"s3://{s3_config.bucket}/{object_key}"


def save_stage2_outputs(
    items: pd.DataFrame,
    events: pd.DataFrame,
    output_dir: str | Path,
    s3_config: S3Config | None = None,
) -> dict[str, str | None]:
    output_dir = Path(output_dir)
    items_path = save_dataframe(items, output_dir / "items.parquet")
    events_path = save_dataframe(events, output_dir / "events.parquet")

    return {
        "items_local": str(items_path),
        "events_local": str(events_path),
        "items_s3": maybe_upload_to_s3(items_path, "recsys/data/items.parquet", s3_config),
        "events_s3": maybe_upload_to_s3(events_path, "recsys/data/events.parquet", s3_config),
    }


def build_matrix_artifacts(
    events: pd.DataFrame,
    items: pd.DataFrame,
) -> MatrixArtifacts:
    if list(events.columns) == ["user_id", "track_id"]:
        train_events = events
    else:
        train_events = events[["user_id", "track_id"]]
    user_codes, user_ids = pd.factorize(train_events["user_id"], sort=False)
    item_to_col = pd.Series(
        np.arange(len(items), dtype=np.int32),
        index=items["track_id"].to_numpy(),
    )
    col_idx = item_to_col.loc[train_events["track_id"]].to_numpy(dtype=np.int32, copy=False)
    row_idx = user_codes.astype(np.int32, copy=False)
    data = np.ones(len(train_events), dtype=np.float32)

    user_item = sparse.csr_matrix(
        (data, (row_idx, col_idx)),
        shape=(len(user_ids), len(items)),
        dtype=np.float32,
    )

    track_popularity = (
        train_events.groupby("track_id", sort=False)
        .size()
        .astype("int32")
        .rename("listen_count")
    )

    track_user_popularity = track_popularity.rename("user_count")
    user_history_len = (
        train_events.groupby("user_id", sort=False)
        .size()
        .astype("int32")
        .rename("user_history_len")
    )

    user_id_values = np.asarray(user_ids, dtype=np.int32)
    user_to_row = pd.Series(
        np.arange(len(user_id_values), dtype=np.int32),
        index=user_id_values,
    )

    del user_codes, row_idx, col_idx, data
    gc.collect()

    return MatrixArtifacts(
        user_item=user_item,
        user_ids=user_id_values,
        user_to_row=user_to_row,
        track_ids=items["track_id"].to_numpy(dtype=np.int32, copy=False),
        item_to_col=item_to_col,
        track_popularity=track_popularity,
        track_user_popularity=track_user_popularity,
        user_history_len=user_history_len,
    )


def fit_als_model(matrix_artifacts: MatrixArtifacts) -> AlternatingLeastSquares:
    model = AlternatingLeastSquares(
        factors=ALS_FACTORS,
        iterations=ALS_ITERATIONS,
        regularization=ALS_REGULARIZATION,
        random_state=ALS_RANDOM_STATE,
        num_threads=max(1, min(4, os.cpu_count() or 1)),
    )
    model.fit(matrix_artifacts.user_item)
    return model


def generate_top_popular_recommendations(
    matrix_artifacts: MatrixArtifacts,
    user_ids: np.ndarray,
    top_k: int = DEFAULT_TOP_K,
    max_scan: int = 500,
) -> pd.DataFrame:
    popular = matrix_artifacts.track_popularity.sort_values(ascending=False)
    popular_track_ids = popular.index.to_numpy(dtype=np.int32, copy=False)[:max_scan]
    popular_scores = np.log1p(popular.to_numpy(dtype=np.float32, copy=False)[:max_scan])
    popular_item_idx = matrix_artifacts.item_to_col.loc[popular_track_ids].to_numpy(dtype=np.int32, copy=False)

    indptr = matrix_artifacts.user_item.indptr
    indices = matrix_artifacts.user_item.indices
    rows: list[tuple[int, int, float, int]] = []

    for user_id in user_ids.tolist():
        row_id = matrix_artifacts.user_to_row.get(user_id)
        if pd.isna(row_id):
            seen_items: set[int] = set()
        else:
            row_id = int(row_id)
            seen_items = set(indices[indptr[row_id] : indptr[row_id + 1]].tolist())

        rank = 1
        for item_idx, track_id, score in zip(popular_item_idx, popular_track_ids, popular_scores):
            if item_idx in seen_items:
                continue

            rows.append((int(user_id), int(track_id), float(score), rank))
            rank += 1

            if rank > top_k:
                break

    return pd.DataFrame(rows, columns=["user_id", "track_id", "score", "rank"])


def generate_als_recommendations(
    model: AlternatingLeastSquares,
    matrix_artifacts: MatrixArtifacts,
    user_ids: np.ndarray,
    top_k: int = DEFAULT_TOP_K,
    batch_size: int = 50_000,
) -> pd.DataFrame:
    known_users = pd.Index(user_ids).intersection(matrix_artifacts.user_to_row.index)
    if known_users.empty:
        return pd.DataFrame(columns=["user_id", "track_id", "score", "rank"])

    row_ids = matrix_artifacts.user_to_row.loc[known_users].to_numpy(dtype=np.int32, copy=False)
    track_lookup = matrix_artifacts.track_ids
    max_rows = len(row_ids) * top_k
    user_buffer = np.empty(max_rows, dtype=np.int32)
    track_buffer = np.empty(max_rows, dtype=np.int32)
    score_buffer = np.empty(max_rows, dtype=np.float32)
    rank_buffer = np.empty(max_rows, dtype=np.int16)
    offset = 0

    processed_users = 0

    for batch_row_ids in chunked(row_ids.tolist(), batch_size):
        batch_row_ids_arr = np.asarray(batch_row_ids, dtype=np.int32)
        rec_ids, rec_scores = model.recommend(
            batch_row_ids_arr,
            matrix_artifacts.user_item[batch_row_ids_arr],
            N=top_k,
            filter_already_liked_items=True,
        )

        rec_ids_flat = rec_ids.reshape(-1)
        valid_mask = rec_ids_flat >= 0
        valid_count = int(valid_mask.sum())

        if valid_count == 0:
            continue

        batch_user_ids = matrix_artifacts.user_ids[batch_row_ids_arr]
        batch_scores = rec_scores.reshape(-1)
        batch_ranks = np.tile(np.arange(1, top_k + 1, dtype=np.int16), len(batch_row_ids_arr))
        end = offset + valid_count

        user_buffer[offset:end] = np.repeat(batch_user_ids, top_k)[valid_mask]
        track_buffer[offset:end] = track_lookup[rec_ids_flat[valid_mask]]
        score_buffer[offset:end] = batch_scores[valid_mask]
        rank_buffer[offset:end] = batch_ranks[valid_mask]
        offset = end
        processed_users += len(batch_row_ids_arr)
        print(f"ALS recommend progress: {processed_users:,}/{len(row_ids):,} users")

    if offset == 0:
        return pd.DataFrame(columns=["user_id", "track_id", "score", "rank"])

    return pd.DataFrame(
        {
            "user_id": user_buffer[:offset],
            "track_id": track_buffer[:offset],
            "score": score_buffer[:offset],
            "rank": rank_buffer[:offset],
        }
    )


def generate_similar_tracks(
    model: AlternatingLeastSquares,
    matrix_artifacts: MatrixArtifacts,
    top_k: int = DEFAULT_TOP_K,
    batch_size: int = 50_000,
) -> pd.DataFrame:
    track_lookup = matrix_artifacts.track_ids
    item_ids = np.arange(len(track_lookup), dtype=np.int32)
    max_rows = len(track_lookup) * (top_k + 1)
    track_buffer = np.empty(max_rows, dtype=np.int32)
    similar_buffer = np.empty(max_rows, dtype=np.int32)
    score_buffer = np.empty(max_rows, dtype=np.float32)
    rank_buffer = np.empty(max_rows, dtype=np.int16)
    offset = 0

    processed_items = 0

    for batch_item_ids in chunked(item_ids.tolist(), batch_size):
        batch_item_ids_arr = np.asarray(batch_item_ids, dtype=np.int32)
        sim_ids, sim_scores = model.similar_items(batch_item_ids_arr, N=top_k + 1)

        sim_ids_flat = sim_ids.reshape(-1)
        valid_mask = sim_ids_flat >= 0
        source_track_ids = np.repeat(track_lookup[batch_item_ids_arr], top_k + 1)[valid_mask]
        similar_track_ids = track_lookup[sim_ids_flat[valid_mask]]
        scores = sim_scores.reshape(-1)[valid_mask]
        ranks = np.tile(np.arange(0, top_k + 1, dtype=np.int16), len(batch_item_ids_arr))[valid_mask]
        valid_count = len(source_track_ids)
        end = offset + valid_count
        track_buffer[offset:end] = source_track_ids
        similar_buffer[offset:end] = similar_track_ids
        score_buffer[offset:end] = scores
        rank_buffer[offset:end] = ranks
        offset = end
        processed_items += len(batch_item_ids_arr)
        print(f"ALS similar progress: {processed_items:,}/{len(item_ids):,} items")

    frame = pd.DataFrame(
        {
            "track_id": track_buffer[:offset],
            "similar_track_id": similar_buffer[:offset],
            "score": score_buffer[:offset],
            "rank": rank_buffer[:offset],
        }
    )
    frame = frame.loc[frame["track_id"] != frame["similar_track_id"]].copy()
    frame["rank"] = frame.groupby("track_id", sort=False).cumcount() + 1
    return frame.loc[frame["rank"] <= top_k].reset_index(drop=True)


def build_candidate_frame(
    als_recs: pd.DataFrame,
    items: pd.DataFrame,
    matrix_artifacts: MatrixArtifacts,
) -> pd.DataFrame:
    als_features = als_recs.rename(columns={"score": "als_score", "rank": "als_rank"})
    candidates = als_features.copy()
    item_features = items.set_index("track_id")[["artist_count", "genre_count", "album_count"]]
    track_popularity_log = np.log1p(matrix_artifacts.track_popularity.astype("float32")).rename(
        "track_popularity_log"
    )
    user_history_len = matrix_artifacts.user_history_len

    candidates["source_count"] = np.int8(1)
    candidates["track_popularity_log"] = (
        candidates["track_id"].map(track_popularity_log).fillna(0).astype("float32")
    )
    candidates["artist_count"] = candidates["track_id"].map(item_features["artist_count"]).astype("int16")
    candidates["genre_count"] = candidates["track_id"].map(item_features["genre_count"]).astype("int16")
    candidates["album_count"] = candidates["track_id"].map(item_features["album_count"]).astype("int16")
    candidates["user_history_len"] = (
        candidates["user_id"].map(user_history_len).fillna(0).astype("int32")
    )
    return candidates


def build_ranker_train_frame(
    events_fit: pd.DataFrame,
    events_label: pd.DataFrame,
    items: pd.DataFrame,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    negatives_per_user: int = 5,
    max_users: int = RANKER_TRAIN_MAX_USERS,
) -> tuple[pd.DataFrame, list[str]]:
    print("Building ranker matrix and first ALS model...")
    fit_artifacts = build_matrix_artifacts(events_fit, items)
    als_model = fit_als_model(fit_artifacts)
    label_users_series = events_label["user_id"].drop_duplicates()

    if len(label_users_series) > max_users:
        label_users_series = label_users_series.sample(max_users, random_state=ALS_RANDOM_STATE)

    label_users = label_users_series.to_numpy(dtype=np.int32, copy=False)
    events_label = events_label.loc[events_label["user_id"].isin(label_users)].copy()

    print("Generating ALS candidates for ranker...")
    als_recs = generate_als_recommendations(
        als_model,
        fit_artifacts,
        label_users,
        top_k=candidate_k,
    )

    print("Merging ALS candidates with ranking features and sampling negatives...")
    candidates = build_candidate_frame(als_recs, items, fit_artifacts)
    label_keys = (
        (events_label["user_id"].astype("int64").to_numpy(copy=False) << 32)
        | events_label["track_id"].astype("int64").to_numpy(copy=False)
    )
    label_keys = np.unique(label_keys)
    candidate_keys = (
        (candidates["user_id"].astype("int64").to_numpy(copy=False) << 32)
        | candidates["track_id"].astype("int64").to_numpy(copy=False)
    )
    candidates["target"] = np.isin(candidate_keys, label_keys).astype("int8")

    positives = candidates.loc[candidates["target"] == 1].copy()
    negatives = (
        candidates.loc[candidates["target"] == 0]
        .sort_values(["user_id", "als_score"], ascending=[True, False])
        .groupby("user_id", group_keys=False)
        .head(negatives_per_user)
        .copy()
    )
    train_frame = pd.concat([positives, negatives], ignore_index=True)

    feature_columns = [
        "als_score",
        "als_rank",
        "track_popularity_log",
        "source_count",
        "user_history_len",
        "artist_count",
        "genre_count",
        "album_count",
    ]

    del fit_artifacts, als_model, als_recs, candidates, positives, negatives, label_keys, candidate_keys
    gc.collect()

    return train_frame, feature_columns


def fit_ranker(train_frame: pd.DataFrame, feature_columns: list[str]) -> CatBoostClassifier:
    ranker = CatBoostClassifier(
        iterations=100,
        depth=6,
        learning_rate=0.08,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=ALS_RANDOM_STATE,
        verbose=False,
    )
    ranker.fit(train_frame[feature_columns], train_frame["target"])
    return ranker


def generate_ranked_recommendations(
    events_train: pd.DataFrame,
    events_target_users: pd.DataFrame,
    items: pd.DataFrame,
    ranker: CatBoostClassifier,
    feature_columns: list[str],
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, MatrixArtifacts, pd.DataFrame]:
    print("Building final train matrix and second ALS model...")
    train_artifacts = build_matrix_artifacts(events_train, items)
    als_model = fit_als_model(train_artifacts)

    target_users = events_target_users["user_id"].drop_duplicates().to_numpy(dtype=np.int32, copy=False)
    print("Generating final ALS candidates and top-popular baseline...")
    als_recs = generate_als_recommendations(
        als_model,
        train_artifacts,
        target_users,
        top_k=candidate_k,
    )
    top_recs = generate_top_popular_recommendations(
        train_artifacts,
        target_users,
        top_k=top_k,
        max_scan=max(120, top_k * 8),
    )
    print("Scoring final candidates with ranker...")
    candidates = build_candidate_frame(als_recs, items, train_artifacts)
    candidates["ranker_score"] = ranker.predict_proba(candidates[feature_columns])[:, 1]
    final_recs = (
        candidates.sort_values(["user_id", "ranker_score"], ascending=[True, False])
        .groupby("user_id", group_keys=False)
        .head(top_k)
        .copy()
    )
    final_recs["rank"] = final_recs.groupby("user_id", sort=False).cumcount() + 1
    final_recs = final_recs[["user_id", "track_id", "ranker_score", "rank"]].rename(
        columns={"ranker_score": "score"}
    )

    similar_tracks = generate_similar_tracks(als_model, train_artifacts, top_k=top_k)
    top_recs_eval = (
        top_recs.sort_values(["user_id", "rank"], ascending=[True, True])
        .groupby("user_id", group_keys=False)
        .head(top_k)
        .copy()
    )
    als_recs_eval = (
        als_recs.sort_values(["user_id", "rank"], ascending=[True, True])
        .groupby("user_id", group_keys=False)
        .head(top_k)
        .copy()
    )

    del als_model, candidates
    gc.collect()

    return top_recs_eval, als_recs_eval, final_recs, train_artifacts, similar_tracks


def evaluate_recommendations(
    recommendations: pd.DataFrame,
    events_test: pd.DataFrame,
    train_artifacts: MatrixArtifacts,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, float]:
    recs = (
        recommendations.sort_values(["user_id", "rank"], ascending=[True, True])
        .groupby("user_id", group_keys=False)
        .head(top_k)[["user_id", "track_id"]]
        .drop_duplicates()
    )
    truth = events_test[["user_id", "track_id"]].drop_duplicates()
    common_users = pd.Index(truth["user_id"].unique()).intersection(recs["user_id"].unique())

    hit_counts = (
        truth.merge(recs, on=["user_id", "track_id"], how="inner")
        .groupby("user_id", sort=False)
        .size()
    )
    truth_counts = truth.groupby("user_id", sort=False).size().reindex(common_users, fill_value=0)
    hits = hit_counts.reindex(common_users, fill_value=0)

    precision = float((hits / top_k).mean())
    recall = float((hits / truth_counts.clip(lower=1)).mean())

    active_items = int(train_artifacts.track_popularity.index.nunique())
    coverage = float(recs["track_id"].nunique() / active_items)

    item_probability = (
        train_artifacts.track_user_popularity / max(1, len(train_artifacts.user_ids))
    ).clip(lower=1 / max(1, len(train_artifacts.user_ids)))
    item_self_information = -np.log2(item_probability)
    novelty = float(
        recs.merge(
            item_self_information.rename("self_information").reset_index(),
            on="track_id",
            how="left",
        )["self_information"].fillna(0).mean()
    )

    return {
        "precision": precision,
        "recall": recall,
        "coverage": coverage,
        "novelty": novelty,
    }


def metrics_to_frame(metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "model"})
    return frame[["model", "precision", "recall", "coverage", "novelty"]]


def save_offline_outputs(
    output_dir: str | Path,
    top_popular: pd.DataFrame,
    personal_als: pd.DataFrame,
    similar_tracks: pd.DataFrame,
    final_recommendations: pd.DataFrame,
    metrics: dict[str, dict[str, float]],
    s3_config: S3Config | None = None,
) -> dict[str, str | None]:
    output_dir = Path(output_dir)
    paths = {
        "top_popular": save_dataframe(top_popular, output_dir / "top_popular.parquet"),
        "personal_als": save_dataframe(personal_als, output_dir / "personal_als.parquet"),
        "similar": save_dataframe(similar_tracks, output_dir / "similar.parquet"),
        "recommendations": save_dataframe(final_recommendations, output_dir / "recommendations.parquet"),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    s3_paths = {
        "top_popular_s3": maybe_upload_to_s3(paths["top_popular"], "recsys/recommendations/top_popular.parquet", s3_config),
        "personal_als_s3": maybe_upload_to_s3(paths["personal_als"], "recsys/recommendations/personal_als.parquet", s3_config),
        "similar_s3": maybe_upload_to_s3(paths["similar"], "recsys/recommendations/similar.parquet", s3_config),
        "recommendations_s3": maybe_upload_to_s3(paths["recommendations"], "recsys/recommendations/recommendations.parquet", s3_config),
    }

    return {
        **{f"{name}_local": str(path) for name, path in paths.items()},
        "metrics_local": str(metrics_path),
        **s3_paths,
    }


def load_cached_offline_outputs(output_dir: str | Path) -> dict[str, Any] | None:
    output_dir = Path(output_dir)
    cached_paths = {
        "top_popular": output_dir / "top_popular.parquet",
        "personal_als": output_dir / "personal_als.parquet",
        "similar": output_dir / "similar.parquet",
        "recommendations": output_dir / "recommendations.parquet",
        "metrics": output_dir / "metrics.json",
    }

    if not all(path.exists() for path in cached_paths.values()):
        return None

    with open(cached_paths["metrics"], encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)

    return {
        "top_popular": pd.read_parquet(cached_paths["top_popular"]),
        "personal_als": pd.read_parquet(cached_paths["personal_als"]),
        "similar": pd.read_parquet(cached_paths["similar"]),
        "recommendations": pd.read_parquet(cached_paths["recommendations"]),
        "metrics": metrics,
        "saved_paths": {
            "top_popular_local": str(cached_paths["top_popular"]),
            "personal_als_local": str(cached_paths["personal_als"]),
            "similar_local": str(cached_paths["similar"]),
            "recommendations_local": str(cached_paths["recommendations"]),
            "metrics_local": str(cached_paths["metrics"]),
        },
    }


def load_event_slice(
    events_path: str | Path,
    start_at: pd.Timestamp | None = None,
    end_at: pd.Timestamp | None = None,
) -> pd.DataFrame:
    filters: list[tuple[str, str, pd.Timestamp]] = []

    if start_at is not None:
        filters.append(("started_at", ">=", start_at))

    if end_at is not None:
        filters.append(("started_at", "<", end_at))

    return pd.read_parquet(
        events_path,
        columns=["user_id", "track_id"],
        filters=filters or None,
    )


def run_offline_pipeline(
    items: pd.DataFrame,
    events: pd.DataFrame,
    output_dir: str | Path,
    s3_config: S3Config | None = None,
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    cached_outputs = load_cached_offline_outputs(output_dir)
    if cached_outputs is not None:
        return cached_outputs

    events_fit = events.loc[events["started_at"] < RANKER_VALID_CUTOFF, ["user_id", "track_id"]]
    events_label = events.loc[
        (events["started_at"] >= RANKER_VALID_CUTOFF) & (events["started_at"] < TRAIN_CUTOFF),
        ["user_id", "track_id"],
    ]

    print("Preparing training data for ranker...")
    ranker_train_frame, feature_columns = build_ranker_train_frame(
        events_fit,
        events_label,
        items,
        candidate_k=candidate_k,
    )
    print("Fitting CatBoost ranker...")
    ranker = fit_ranker(ranker_train_frame, feature_columns)
    del ranker_train_frame, events_fit, events_label
    gc.collect()

    events_train = events.loc[events["started_at"] < TRAIN_CUTOFF, ["user_id", "track_id"]]
    events_test = events.loc[events["started_at"] >= TRAIN_CUTOFF, ["user_id", "track_id"]]
    del events
    gc.collect()

    print("Generating final recommendation files...")
    top_popular, personal_als, final_recommendations, train_artifacts, similar_tracks = generate_ranked_recommendations(
        events_train,
        events_test,
        items,
        ranker,
        feature_columns,
        top_k=top_k,
        candidate_k=candidate_k,
    )

    metrics = {
        "top_popular": evaluate_recommendations(top_popular, events_test, train_artifacts, top_k=top_k),
        "personal_als": evaluate_recommendations(personal_als, events_test, train_artifacts, top_k=top_k),
        "ranked": evaluate_recommendations(final_recommendations, events_test, train_artifacts, top_k=top_k),
    }

    print("Saving offline artifacts...")
    saved_paths = save_offline_outputs(
        output_dir,
        top_popular,
        personal_als,
        similar_tracks,
        final_recommendations,
        metrics,
        s3_config,
    )

    return {
        "top_popular": top_popular,
        "personal_als": personal_als,
        "similar": similar_tracks,
        "recommendations": final_recommendations,
        "metrics": metrics,
        "saved_paths": saved_paths,
    }


def run_offline_pipeline_from_files(
    items_path: str | Path,
    events_path: str | Path,
    output_dir: str | Path,
    s3_config: S3Config | None = None,
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    cached_outputs = load_cached_offline_outputs(output_dir)
    if cached_outputs is not None:
        return cached_outputs

    items = pd.read_parquet(items_path)

    print("Loading ranker fit and label slices from events.parquet...")
    events_fit = load_event_slice(events_path, end_at=RANKER_VALID_CUTOFF)
    events_label = load_event_slice(events_path, start_at=RANKER_VALID_CUTOFF, end_at=TRAIN_CUTOFF)

    print("Preparing training data for ranker...")
    ranker_train_frame, feature_columns = build_ranker_train_frame(
        events_fit,
        events_label,
        items,
        candidate_k=candidate_k,
    )
    print("Fitting CatBoost ranker...")
    ranker = fit_ranker(ranker_train_frame, feature_columns)
    del ranker_train_frame, events_fit, events_label
    gc.collect()

    print("Loading final train and test slices from events.parquet...")
    events_train = load_event_slice(events_path, end_at=TRAIN_CUTOFF)
    events_test = load_event_slice(events_path, start_at=TRAIN_CUTOFF)

    print("Generating final recommendation files...")
    top_popular, personal_als, final_recommendations, train_artifacts, similar_tracks = generate_ranked_recommendations(
        events_train,
        events_test,
        items,
        ranker,
        feature_columns,
        top_k=top_k,
        candidate_k=candidate_k,
    )

    metrics = {
        "top_popular": evaluate_recommendations(top_popular, events_test, train_artifacts, top_k=top_k),
        "personal_als": evaluate_recommendations(personal_als, events_test, train_artifacts, top_k=top_k),
        "ranked": evaluate_recommendations(final_recommendations, events_test, train_artifacts, top_k=top_k),
    }

    print("Saving offline artifacts...")
    saved_paths = save_offline_outputs(
        output_dir,
        top_popular,
        personal_als,
        similar_tracks,
        final_recommendations,
        metrics,
        s3_config,
    )

    return {
        "top_popular": top_popular,
        "personal_als": personal_als,
        "similar": similar_tracks,
        "recommendations": final_recommendations,
        "metrics": metrics,
        "saved_paths": saved_paths,
    }
