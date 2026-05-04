"""Whisper transcription API for the transcript-ui React app."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
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
UNKNOWN_TRANSCRIPT_SENTENCE = "Unknown Sentence"
LIVE_CHUNK_WINDOW_SEC = float(os.environ.get("LIVE_CHUNK_WINDOW_SEC", "10"))
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

def _cors_allow_origins() -> list[str]:
    defaults = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ]
    extra = os.environ.get("CORS_EXTRA_ORIGINS", "")
    more = [o.strip() for o in extra.split(",") if o.strip()]
    return defaults + more


app = FastAPI(title="Whisper transcribe")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allow_origins(),
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


def _ffmpeg_binary() -> str:
    return (os.environ.get("FFMPEG_PATH") or "").strip() or shutil.which("ffmpeg") or "ffmpeg"


def _ffmpeg_normalize_for_whisper(src_path: str) -> str:
    """
    Decode arbitrary browser / container input to mono 16 kHz PCM WAV.

    Whisper's built-in ``load_audio`` pipes ffmpeg stdout; fragmented WebM from
    MediaRecorder often fails that path. A two-step file decode is more reliable.
    """
    fd, wav_path = tempfile.mkstemp(suffix=".whisper16k.wav")
    os.close(fd)
    ffmpeg = _ffmpeg_binary()
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-nostdin",
        "-threads",
        "0",
        "-fflags",
        "+genpts+discardcorrupt",
        "-err_detect",
        "ignore_err",
        "-i",
        src_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        "-f",
        "wav",
        wav_path,
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        err = (r.stderr or b"").decode(errors="replace").strip()
        try:
            os.unlink(wav_path)
        except OSError:
            pass
        tail = err[-3500:] if err else "(no stderr)"
        raise RuntimeError(f"ffmpeg could not decode audio (exit {r.returncode}): {tail}") from None
    return wav_path


def _whisper_transcribe_file(tmp_path: str, language: str | None) -> tuple[str, list[dict]]:
    decoded = _ffmpeg_normalize_for_whisper(tmp_path)
    try:
        model = get_model()
        opts: dict = {}
        if language and language.strip():
            opts["language"] = language.strip()
        result = model.transcribe(decoded, **opts)
        text = (result.get("text") or "").strip()
        segments = _segments_from_result(result)
        return text, segments
    finally:
        if decoded != tmp_path:
            try:
                os.unlink(decoded)
            except OSError:
                pass


def _offset_segment_ids_and_times(segments: list[dict], *, time_offset_sec: float, chunk_seq: int) -> list[dict]:
    """Stable IDs across a live session: chunk_seq * 1000 + index; shift times into session timeline."""
    off = float(time_offset_sec)
    base = max(0, int(chunk_seq)) * 1000
    out: list[dict] = []
    for i, seg in enumerate(segments):
        out.append(
            {
                "id": base + i,
                "start": float(seg["start"]) + off,
                "end": float(seg["end"]) + off,
                "text": seg["text"],
            }
        )
    return out


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

        decode_placeholder = False
        try:
            text, segments = _whisper_transcribe_file(tmp_path, language)
        except Exception as e:
            logger.warning("transcribe/decode failed; using %r: %s", UNKNOWN_TRANSCRIPT_SENTENCE, e)
            text = UNKNOWN_TRANSCRIPT_SENTENCE
            segments = [
                {"id": 0, "start": 0.0, "end": 1.0, "text": UNKNOWN_TRANSCRIPT_SENTENCE},
            ]
            decode_placeholder = True

        TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_name = f"{_safe_audio_stem(audio.filename)}_{ts}.txt"
        out_path = TRANSCRIPTS_DIR / out_name
        out_path.write_text(_transcript_file_body(text, segments), encoding="utf-8")

        document: dict | None = None
        entity_saved_path: str | None = None
        entity_error: str | None = None
        if extract_entities and segments and not decode_placeholder:
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


@app.post("/api/transcribe-chunk")
async def transcribe_chunk(
    audio: UploadFile = File(...),
    time_offset_sec: float = Form(0.0),
    chunk_seq: int = Form(0),
    language: str | None = Form(None),
    extract_entities: bool = Form(True),
    entity_backend: str | None = Form(None),
    persist_transcript: bool = Form(False),
):
    """Transcribe one timed slice (e.g. 10s of live mic). Times are shifted into a session timeline."""
    if not audio.filename:
        raise HTTPException(status_code=400, detail="Missing file name")

    suffix = Path(audio.filename).suffix.lower()
    if suffix not in {".mp3", ".wav", ".m4a", ".webm", ".ogg", ".flac", ".mp4", ".mpeg", ".mpga"}:
        suffix = ".webm"

    tmp_path: str | None = None
    try:
        contents = await audio.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(contents)

        decode_placeholder = False
        try:
            text, raw_segments = _whisper_transcribe_file(tmp_path, language)
            segments = _offset_segment_ids_and_times(
                raw_segments, time_offset_sec=time_offset_sec, chunk_seq=chunk_seq
            )
        except Exception as e:
            logger.warning(
                "transcribe-chunk %s decode/transcribe failed; using %r: %s",
                int(chunk_seq),
                UNKNOWN_TRANSCRIPT_SENTENCE,
                e,
            )
            text = UNKNOWN_TRANSCRIPT_SENTENCE
            raw_placeholder = [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": LIVE_CHUNK_WINDOW_SEC,
                    "text": UNKNOWN_TRANSCRIPT_SENTENCE,
                }
            ]
            segments = _offset_segment_ids_and_times(
                raw_placeholder, time_offset_sec=time_offset_sec, chunk_seq=chunk_seq
            )
            decode_placeholder = True

        saved_path: str | None = None
        if persist_transcript and segments:
            TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            stem = _safe_audio_stem(audio.filename or "live_chunk")
            out_name = f"{stem}_chunk{chunk_seq}_{ts}.txt"
            out_path = TRANSCRIPTS_DIR / out_name
            out_path.write_text(_transcript_file_body(text, segments), encoding="utf-8")
            saved_path = str(out_path.resolve())

        document: dict | None = None
        entity_saved_path: str | None = None
        entity_error: str | None = None
        if extract_entities and segments and not decode_placeholder:
            backend = resolve_entity_backend(entity_backend)
            chunks_dict = [
                {"id": s["id"], "start": s["start"], "end": s["end"], "text": s["text"]} for s in segments
            ]
            try:
                normalized, entities = run_extraction(chunks_dict, backend=backend)
                label = f"{audio.filename or 'live'}#chunk{chunk_seq}"
                doc = build_document(
                    chunks=normalized,
                    entities=entities,
                    source_label=label,
                    backend=backend,
                )
                document = doc
                if persist_transcript:
                    base = audio.filename or "live_chunk"
                    path = save_document(doc, ENTITY_JSON_DIR, _safe_audio_stem(base))
                    entity_saved_path = str(path)
            except Exception as e:
                logger.exception("entity extraction after transcribe-chunk failed")
                entity_error = str(e)

        return {
            "transcript": text,
            "segments": segments,
            "saved_path": saved_path,
            "document": document,
            "entity_saved_path": entity_saved_path,
            "entity_error": entity_error,
            "chunk_seq": int(chunk_seq),
            "time_offset_sec": float(time_offset_sec),
            "decode_placeholder": decode_placeholder,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("transcribe-chunk failed")
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
