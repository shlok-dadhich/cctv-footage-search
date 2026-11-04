"""In-memory embedding index for image-to-image "find similar" search.

Loads SigLIP vectors from the embeddings table into a numpy matrix (populated
per (db_path, entity_type, model) and cached by callers e.g. via
functools.lru_cache / Streamlit st.cache_resource) and answers cosine-similarity
queries. Supports incremental refresh: rows with id > last seen max are appended
without rebuilding the whole matrix.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from common.db import CCTVDatabase
from common.embed import get_model_tag


@dataclass
class EmbeddingEntry:
    id: int
    video_id: str
    entity_type: str
    entity_id: int          # persons/vehicles.id (upsert row id)
    track_id: int | None
    camera_id: str | None
    ts: float | None
    timestamp_s: float | None
    crop_path: str | None


class EmbeddingIndex:
    """Cosine-similarity search over embeddings loaded from the database."""

    def __init__(
        self,
        db: CCTVDatabase,
        entity_type: str,
        model: str | None = None,
    ) -> None:
        self.db = db
        self.entity_type = entity_type
        self.model = model or get_model_tag()
        self.entries: list[EmbeddingEntry] = []
        self.matrix: np.ndarray | None = None   # (N, 768) float32, L2-normalized rows
        self.last_max_id = 0

    # ── Loading / refresh ────────────────────────────────────────────
    def refresh(self) -> int:
        """Load new embeddings since the last refresh. Returns count added."""
        rows = self.db.get_all_embeddings(
            entity_type=self.entity_type, model=self.model,
            min_id=self.last_max_id + 1 if self.entries else None,
        )
        new_vecs: list[np.ndarray] = []
        for r in rows:
            vec = np.frombuffer(r["embedding"], dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(vec))
            if norm <= 0.0 or vec.shape[0] != self._dim_hint():
                continue
            new_vecs.append(vec / (norm + 1e-8))
            self.entries.append(EmbeddingEntry(
                id=int(r["id"]),
                video_id=r["video_id"],
                entity_type=r["entity_type"],
                entity_id=int(r["entity_id"]),
                track_id=int(r["track_id"]) if r.get("track_id") is not None else None,
                camera_id=r.get("camera_id"),
                ts=r.get("ts"),
                timestamp_s=r.get("timestamp_s"),
                crop_path=r.get("crop_path"),
            ))
        if new_vecs:
            block = np.stack(new_vecs, axis=0).astype(np.float32)
            if self.matrix is None:
                self.matrix = block
            else:
                self.matrix = np.vstack([self.matrix, block])
        self.last_max_id = self.db.get_embedding_max_id(self.entity_type, self.model)
        return len(new_vecs)

    def _dim_hint(self) -> int:
        from common.embed import get_embedding_dim
        return get_embedding_dim()

    def __len__(self) -> int:
        return len(self.entries)

    # ── Search ───────────────────────────────────────────────────────
    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 10,
        min_sim: float = 0.0,
        video_id: str | None = None,
        camera_id: str | None = None,
        ts_from: float | None = None,
        ts_to: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return top_k embedding hits as dicts sorted by similarity desc."""
        if self.matrix is None or self.matrix.shape[0] == 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        sims = self.matrix @ q
        if min_sim > -2.0:
            sims_filtered = sims >= min_sim
        else:
            sims_filtered = np.ones_like(sims, dtype=bool)
        order = np.argsort(-sims)
        hits: list[dict[str, Any]] = []
        for idx in order:
            if not sims_filtered[idx]:
                continue
            e = self.entries[int(idx)]
            if video_id and e.video_id != video_id:
                continue
            if camera_id and e.camera_id != camera_id:
                continue
            # Rows without an absolute timestamp cannot be proven outside the
            # window, so keep them instead of silently emptying timed searches.
            if ts_from is not None and e.ts is not None and e.ts < ts_from:
                continue
            if ts_to is not None and e.ts is not None and e.ts > ts_to:
                continue
            hits.append({
                "similarity": float(sims[idx]),
                "embedding_id": e.id,
                "video_id": e.video_id,
                "entity_type": self.entity_type,
                "entity_id": e.entity_id,
                "track_id": e.track_id,
                "camera_id": e.camera_id,
                "ts": e.ts,
                "timestamp_s": e.timestamp_s,
                "crop_path": e.crop_path,
            })
            if len(hits) >= top_k:
                break
        return hits

    def search_tracks(
        self,
        query_vec: np.ndarray,
        top_k: int = 10,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        """Search embeddings but collapse to the best hit per (video, track)."""
        emb_hits = self.search(query_vec, top_k=max(top_k * 8, 50), **filters)
        best: dict[tuple[str, str, int], dict[str, Any]] = {}
        for h in emb_hits:
            key = (h["video_id"], self.entity_type, h["track_id"] or h["entity_id"])
            if key not in best or h["similarity"] > best[key]["similarity"]:
                best[key] = h
        ranked = sorted(best.values(), key=lambda h: -h["similarity"])
        return ranked[:top_k]
