"""search_core.py — shared search plumbing for CLI, Streamlit app, and API use.

Consolidates the vector operations that used to live inside app.py so all
front-ends agree on results and defaults:
- text_search: SigLIP text-embedding similarity over DB vectors
- image_search: image-to-image similarity (delegates to image_query.find_similar)
- find_similar_image_and_text: combined query vector (image alpha=0.7 + text 0.3)

All searches respect absolute-time and camera/video filters when provided.
Query images are processed in memory only.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common.emb_index import EmbeddingIndex
from common.embed import embed_text, is_available
from common.db import CCTVDatabase

DEFAULT_IMAGE_TEXT_ALPHA = 0.7


@dataclass
class SearchRequest:
    """A unified vector-search request (text, image, or blended)."""
    query_text: str | None = None
    query_images: list[Any] = field(default_factory=list)  # paths/bytes/BGR arrays
    entity: str = "auto"          # person | vehicle | auto
    top_k: int = 20
    min_sim: float = 0.0
    video_id: str | None = None
    camera_id: str | None = None
    ts_from: float | None = None
    ts_to: float | None = None
    alpha: float = DEFAULT_IMAGE_TEXT_ALPHA   # image weight when blending with text
    per_track: bool = True                    # collapse frame-level hits per track
    device: str | None = None


@dataclass
class SearchResponse:
    entity: str | None
    hits: list[dict[str, Any]]
    method: str                    # "text" | "image" | "image+text"
    error: str | None = None


def resolve_device(requested: str | None = None) -> str:
    if requested:
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def text_search(db: CCTVDatabase, request: SearchRequest) -> SearchResponse:
    """SigLIP text-embedding similarity over all indexed embeddings."""
    if not request.query_text:
        return SearchResponse(entity=None, hits=[], method="text",
                              error="no query text")
    if not is_available():
        return SearchResponse(entity=None, hits=[], method="text",
                              error="SigLIP stack unavailable")
    device = resolve_device(request.device)
    qvec = embed_text(request.query_text, device=device)
    if qvec is None:
        return SearchResponse(entity=None, hits=[], method="text",
                              error="text embedding failed")

    pooling = ["person", "vehicle"] if request.entity == "auto" else [request.entity]
    best_hits: list[dict[str, Any]] = []
    resolved = None
    for etype in pooling:
        idx = EmbeddingIndex(db, entity_type=etype)
        idx.refresh()
        if len(idx) == 0:
            continue
        fn = idx.search_tracks if request.per_track else idx.search
        hits = fn(qvec, top_k=request.top_k, min_sim=request.min_sim,
                  video_id=request.video_id, camera_id=request.camera_id,
                  ts_from=request.ts_from, ts_to=request.ts_to)
        if hits and (resolved is None or hits[0]["similarity"] > (best_hits[0]["similarity"] if best_hits else -1)):
            best_hits, resolved = hits, etype
        elif hits and resolved is not None:
            # keep merge: append non-overlapping hits up to top_k
            seen = {(h["video_id"], h["entity_id"]) for h in best_hits}
            extra = [h for h in hits if (h["video_id"], h["entity_id"]) not in seen]
            best_hits.extend(extra[: max(0, request.top_k - len(best_hits))])
    return SearchResponse(entity=resolved, hits=best_hits, method="text")


def image_search(db: CCTVDatabase, request: SearchRequest) -> SearchResponse:
    """Image-to-image similarity (QueryCrop + SigLIP image embeddings)."""
    from image_query import find_similar
    resolved, hits = find_similar(
        db,
        request.query_images,
        entity=request.entity,
        top_k=request.top_k,
        min_sim=request.min_sim,
        video_id=request.video_id,
        camera_id=request.camera_id,
        ts_from=request.ts_from,
        ts_to=request.ts_to,
        device=resolve_device(request.device),
        per_track=request.per_track,
    )
    err = None
    if hits and isinstance(hits[0], dict) and hits[0].get("error"):
        err = hits[0]["error"]
        hits = []
    return SearchResponse(entity=resolved or None, hits=hits, method="image", error=err)


def _blend(q_img: np.ndarray, q_text: np.ndarray, alpha: float) -> np.ndarray:
    q = alpha * q_img + (1.0 - alpha) * q_text
    return q / (np.linalg.norm(q) + 1e-8)


def find_similar_image_and_text(
    db: CCTVDatabase, request: SearchRequest
) -> SearchResponse:
    """Blend image + text query vectors (default alpha=0.7 image).

    The images resolve the entity type (QueryCrop), then the blend searches
    that single index so text nudges ranking without diluting appearance.
    """
    if not request.query_images:
        return text_search(db, request)
    if not is_available():
        return SearchResponse(entity=None, hits=[], method="image+text",
                              error="SigLIP stack unavailable")
    from image_query import _resolve_query_crops, load_image_bgr

    device = resolve_device(request.device)
    imgs: list[np.ndarray] = []
    for item in request.query_images:
        img = load_image_bgr(item)
        if img is not None:
            imgs.append(img)
    if not imgs:
        return SearchResponse(entity=None, hits=[], method="image+text",
                              error="no decodable query image")

    resolved, crops = _resolve_query_crops(imgs, request.entity, device)

    from common.embed import embed_images_mean
    q_img = embed_images_mean(crops if crops else imgs, device=device)
    q_txt = embed_text(request.query_text or "", device=device)
    if q_img is None:
        return SearchResponse(entity=None, hits=[], method="image+text",
                              error="failed to embed query image(s)")
    blended = _blend(q_img, q_txt, request.alpha) if q_txt is not None else q_img

    if resolved is None:
        return SearchResponse(entity=None, hits=[], method="image+text",
                              error="could not resolve entity for query")
    idx = EmbeddingIndex(db, entity_type=resolved)
    idx.refresh()
    if len(idx) == 0:
        return SearchResponse(entity=resolved, hits=[], method="image+text")
    fn = idx.search_tracks if request.per_track else idx.search
    hits = fn(blended, top_k=request.top_k, min_sim=request.min_sim,
              video_id=request.video_id, camera_id=request.camera_id,
              ts_from=request.ts_from, ts_to=request.ts_to)
    for h in hits:
        h["query_entity"] = resolved
    return SearchResponse(entity=resolved, hits=hits, method="image+text")


def similar_to_track(
    db: CCTVDatabase,
    entity_type: str,
    video_id: str,
    track_id: int,
    request: SearchRequest,
    exclude_self: bool = True,
    model: str | None = None,
) -> SearchResponse:
    """Find tracks similar to an already-indexed track via its stored embeddings.

    The source track's mean embedding is the query; itself is excluded by default.
    """
    rows = db.get_track_embedding(entity_type, video_id, track_id, model=model)
    if not rows:
        return SearchResponse(entity=entity_type, hits=[], method="similar",
                              error=f"no embeddings for {entity_type}/{video_id}/{track_id}")
    vecs = []
    for r in rows:
        v = np.frombuffer(r["embedding"], dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(v))
        if n > 0.0:
            vecs.append(v / (n + 1e-8))
    if not vecs:
        return SearchResponse(entity=entity_type, hits=[], method="similar",
                              error="all source embeddings unreadable")
    q = np.mean(np.stack(vecs, axis=0), axis=0)
    q = q / (np.linalg.norm(q) + 1e-8)

    idx = EmbeddingIndex(db, entity_type=entity_type)
    idx.refresh()
    hits = idx.search_tracks(
        q, top_k=request.top_k + 1, video_id=request.video_id,
        camera_id=request.camera_id, ts_from=request.ts_from, ts_to=request.ts_to,
    )
    src_entity_id = rows[0].get("entity_id")
    if exclude_self:
        hits = [h for h in hits
                if not (h["video_id"] == video_id and h["track_id"] == track_id)]
        if src_entity_id is not None:
            hits = [h for h in hits if not (h["video_id"] == video_id and h["entity_id"] == src_entity_id)]
    hits = hits[:request.top_k]
    for h in hits:
        h["query_entity"] = entity_type
        h["similar_to"] = f"{entity_type}:{video_id}:track{track_id}"
    return SearchResponse(entity=entity_type, hits=hits, method="similar")


def execute(request: SearchRequest, db: CCTVDatabase) -> SearchResponse:
    """Route a request to the right vector search implementation."""
    if request.query_images and request.query_text:
        return find_similar_image_and_text(db, request)
    if request.query_images:
        return image_search(db, request)
    return text_search(db, request)
