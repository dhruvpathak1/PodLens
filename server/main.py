"""Whisper transcription API for the transcript-ui React app."""

from __future__ import annotations

import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env", encoding="utf-8-sig")

# Keep model weights out of ~/.cache — macOS / sandbox often returns "Operation not permitted"
# for that path. Must run before `import whisper` (which imports torch).
_server_dir = Path(__file__).resolve().parent
WHISPER_DOWNLOAD_ROOT = Path(
    os.environ.get("WHISPER_DOWNLOAD_ROOT", str(_server_dir / "whisper_models"))
)
os.environ.setdefault("TORCH_HOME", str(_server_dir / "torch_home"))

import whisper
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from enrichment import enrich_entities_payload
from entity_pipeline import build_document, resolve_entity_backend, run_extraction, save_document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("WHISPER_MODEL", "base")
TRANSCRIPTS_DIR = Path(
    os.environ.get("TRANSCRIPTS_DIR", str(Path(__file__).resolve().parent / "transcripts"))
)
ENTITY_JSON_DIR = Path(
    os.environ.get("ENTITY_JSON_DIR", str(Path(__file__).resolve().parent / "entity_exports"))
)
_model = None


class ChunkIn(BaseModel):
    id: int = 0
    start: float
    end: float
    text: str = ""


class ExtractEntitiesRequest(BaseModel):
    chunks: list[ChunkIn]
    source_label: str | None = None
    persist: bool = True
    backend: str | None = None  # "spacy" | "claude"


class EntityRefIn(BaseModel):
    type: str
    text: str
    start_sec: float = 0.0
    end_sec: float = 0.0
    chunk_id: int = 0


class EnrichEntitiesRequest(BaseModel):
    entities: list[EntityRefIn]


def _safe_audio_stem(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"[^\w\-.]", "_", stem, flags=re.UNICODE)[:80]
    return stem or "audio"


def _format_ts(sec: float) -> str:
    sec = max(0.0, float(sec))
    ms = int(round((sec % 1) * 1000))
    total = int(sec)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _segments_from_result(result: dict) -> list[dict]:
    raw = result.get("segments") or []
    out: list[dict] = []
    for i, seg in enumerate(raw):
        if not isinstance(seg, dict):
            continue
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        out.append(
            {
                "id": i,
                "start": float(seg.get("start") or 0.0),
                "end": float(seg.get("end") or 0.0),
                "text": text,
            }
        )
    return out


def _transcript_file_body(text: str, segments: list[dict]) -> str:
    if segments:
        lines = [f"[{_format_ts(s['start'])} → {_format_ts(s['end'])}] {s['text']}" for s in segments]
        return "\n".join(lines) + "\n"
    return text + ("\n" if text else "")

app = FastAPI(title="Whisper transcribe")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_model():
    global _model
    if _model is None:
        logger.info("Loading Whisper model %r (first request may take a while)", MODEL_NAME)
        WHISPER_DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
        _model = whisper.load_model(MODEL_NAME, download_root=str(WHISPER_DOWNLOAD_ROOT))
    return _model


@app.post("/api/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    language: str | None = Form(None),
    extract_entities: bool = Form(True),
    entity_backend: str | None = Form(None),
):
    if not audio.filename:
        raise HTTPException(status_code=400, detail="Missing file name")

    suffix = Path(audio.filename).suffix.lower()
    if suffix not in {".mp3", ".wav", ".m4a", ".webm", ".ogg", ".flac", ".mp4", ".mpeg", ".mpga"}:
        suffix = ".wav"

    tmp_path: str | None = None
    try:
        contents = await audio.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(contents)

        model = get_model()
        opts: dict = {}
        if language and language.strip():
            opts["language"] = language.strip()

        result = model.transcribe(tmp_path, **opts)
        text = (result.get("text") or "").strip()
        segments = _segments_from_result(result)

        TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_name = f"{_safe_audio_stem(audio.filename)}_{ts}.txt"
        out_path = TRANSCRIPTS_DIR / out_name
        out_path.write_text(_transcript_file_body(text, segments), encoding="utf-8")

        document: dict | None = None
        entity_saved_path: str | None = None
        entity_error: str | None = None
        if extract_entities and segments:
            backend = resolve_entity_backend(entity_backend)
            chunks_dict = [
                {"id": s["id"], "start": s["start"], "end": s["end"], "text": s["text"]} for s in segments
            ]
            try:
                normalized, entities = run_extraction(chunks_dict, backend=backend)
                doc = build_document(
                    chunks=normalized,
                    entities=entities,
                    source_label=audio.filename,
                    backend=backend,
                )
                document = doc
                base = audio.filename or "entities"
                path = save_document(doc, ENTITY_JSON_DIR, _safe_audio_stem(base))
                entity_saved_path = str(path)
            except Exception as e:
                logger.exception("entity extraction after transcribe failed")
                entity_error = str(e)

        return {
            "transcript": text,
            "segments": segments,
            "saved_path": str(out_path.resolve()),
            "document": document,
            "entity_saved_path": entity_saved_path,
            "entity_error": entity_error,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("transcribe failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        if tmp_path and os.path.isfile(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@app.post("/api/extract-entities")
async def extract_entities(body: ExtractEntitiesRequest):
    if not body.chunks:
        raise HTTPException(status_code=400, detail="chunks must be non-empty")

    backend = resolve_entity_backend(body.backend)

    chunks_dict = [c.model_dump() for c in body.chunks]
    try:
        normalized, entities = run_extraction(chunks_dict, backend=backend)
    except Exception as e:
        logger.exception("entity extraction failed")
        raise HTTPException(status_code=500, detail=str(e)) from e

    doc = build_document(
        chunks=normalized,
        entities=entities,
        source_label=body.source_label,
        backend=backend,
    )
    saved_path: str | None = None
    if body.persist:
        base = body.source_label or "entities"
        path = save_document(doc, ENTITY_JSON_DIR, _safe_audio_stem(base) if base else "entities")
        saved_path = str(path)

    return {"document": doc, "saved_path": saved_path}


@app.post("/api/enrich-entities")
async def enrich_entities_route(body: EnrichEntitiesRequest):
    if not body.entities:
        raise HTTPException(status_code=400, detail="entities must be non-empty")
    rows = [e.model_dump() for e in body.entities]
    try:
        result = await enrich_entities_payload(rows)
    except Exception as e:
        logger.exception("enrich-entities failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
    return result


@app.get("/api/health")
def health():
    uk = bool(os.environ.get("UNSPLASH_ACCESS_KEY", "").strip())
    return {
        "ok": True,
        "model": MODEL_NAME,
        "entity_backend": resolve_entity_backend(None),
        "unsplash_configured": uk,
    }
