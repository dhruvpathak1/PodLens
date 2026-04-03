"""Entity extraction from timestamped transcript chunks (spaCy or Claude classic NER)."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Repo root (parent of server/). Ensures .env loads even if the process cwd differs.
_ENTITY_ENV_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ENTITY_ENV_ROOT / ".env", encoding="utf-8-sig")

# --- Allowed output types (user-facing). MISC = classic CoNLL miscellaneous (non PER/ORG/LOC). ---
ENTITY_TYPES = frozenset(
    {"PLACE", "PERSON", "TECHNOLOGY", "EVENT", "COMPANY", "MISC"}
)


def resolve_entity_backend(explicit: str | None = None) -> str:
    """Prefer explicit (API / form), then ENTITY_BACKEND env, then Claude if key is set, else spaCy."""
    if explicit and str(explicit).strip():
        b = str(explicit).strip().lower()
    else:
        env = os.environ.get("ENTITY_BACKEND", "").strip().lower()
        if env in ("spacy", "claude"):
            b = env
        elif os.environ.get("ANTHROPIC_API_KEY", "").strip():
            b = "claude"
        else:
            b = "spacy"
    if b not in ("spacy", "claude"):
        b = "spacy"
    return b

# Filler / disfluency phrases (removed before NER)
_FILLER_PATTERN = re.compile(
    r"\b("
    r"um|uh|er|ah|hmm|hm|mm|uhm|erm|"
    r"like(?=\s+(?:i|you|the|a|an|this|that|it)\b)|"
    r"you know|i mean|i guess|sort of|kind of|"
    r"actually|basically|literally|obviously|honestly|"
    r"right\?|okay\?|ok\?"
    r")\b",
    re.IGNORECASE,
)

# Whole-line or clause noise
_NOISE_PHRASES = re.compile(
    r"\b(thanks for watching|subscribe|like and subscribe|see you next time)\b",
    re.IGNORECASE,
)

# Post-filter: drop entity text matching these (too generic)
_GENERIC_ENTITIES = frozenset(
    {
        "today",
        "tomorrow",
        "here",
        "there",
        "now",
        "something",
        "someone",
        "anything",
        "everything",
        "nothing",
        "people",
        "things",
        "stuff",
        "way",
        "thing",
        "lot",
        "bit",
    }
)


def filter_noise(text: str) -> str:
    """Strip filler words and common throwaway phrases."""
    if not text or not text.strip():
        return ""
    t = _NOISE_PHRASES.sub("", text)
    t = _FILLER_PATTERN.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _entity_time_span(
    chunk_start: float, chunk_end: float, chunk_text: str, ent_start_char: int, ent_end_char: int
) -> tuple[float, float]:
    """Map character span inside chunk to approximate audio times."""
    n = max(len(chunk_text), 1)
    dur = max(chunk_end - chunk_start, 0.0)
    rel0 = max(0, min(ent_start_char, n)) / n
    rel1 = max(0, min(ent_end_char, n)) / n
    return chunk_start + rel0 * dur, chunk_start + rel1 * dur


def _clean_entity_text(text: str) -> str | None:
    t = text.strip()
    if len(t) < 2:
        return None
    low = t.lower()
    if low in _GENERIC_ENTITIES:
        return None
    return t


# Context cues for homonyms (company vs common noun, etc.). Chunk text is lowercased.
_APPLE_TECH = re.compile(
    r"\b(iphone|ipad|ipod|macbook|imac|ios|ipados|mac\s*os|app\s+store|cupertino|"
    r"tim\s+cook|steve\s+jobs|wwdc|airpods|apple\s+watch|icloud|m1\b|m2\b|m3\b|m4\b|"
    r"a\d+\s+bionic|vision\s+pro|apple\s+park|nasdaq|aapl|ecosystem|siri|facetime|"
    r"apple\s+music|apple\s+tv|mac\s+studio|ipad\s+pro|developer\s+conference|"
    r"silicon|osx|watchos|testflight|"
    r"apple\s+(announced|reported|reports|unveiled|released|launched|introduced|said|posted))\b",
    re.I,
)
_APPLE_FRUIT = re.compile(
    r"\b(fruit|apple\s+pie|apples\s+and|orchard|apple\s+juice|recipe|granny\s+smith|"
    r"cider|rotten\s+apple|red\s+delicious|gala\s+apple|honeycrisp|peel|crisp\s+apple|"
    r"\ban apple\b|\bthe apple\s+was\b|\bapples\s+(are|were|taste))\b",
    re.I,
)
_AMAZON_CORP = re.compile(
    r"\b(aws|amazon\s+prime|prime\s+video|kindle|alexa|bezos|e-?commerce|marketplace|"
    r"amazon\.com|fulfillment|amazon\s+web)\b",
    re.I,
)
_AMAZON_RIVER = re.compile(
    r"\b(rainforest|amazon\s+river|amazon\s+basin|manaus|peru|amazonia)\b",
    re.I,
)
_ORACLE_TECH = re.compile(
    r"\b(database|sql|java\b|oci\b|oracle\s+corp|larry\s+ellison|oracle\s+cloud|"
    r"enterprise\s+software|erp)\b",
    re.I,
)
_ORACLE_MYTH = re.compile(
    r"\b(myth|prophecy|greek|delphi|ancient|pythia|apollo)\b",
    re.I,
)
_META_CORP = re.compile(
    r"\b(facebook|instagram|whatsapp|zuckerberg|meta\s+quest|threads|reality\s+labs|"
    r"oculus|meta\s+platforms)\b",
    re.I,
)


def _apply_context_disambiguation(
    entity: dict[str, Any], chunk_lower: str
) -> dict[str, Any] | None:
    """Adjust or drop entities when chunk context resolves homonyms."""
    e = dict(entity)
    text = (e.get("text") or "").strip()
    low = text.lower()
    typ = str(e.get("type", ""))

    if low == "apple":
        tech = _APPLE_TECH.search(chunk_lower)
        fruit = _APPLE_FRUIT.search(chunk_lower)
        if tech and not fruit:
            e["type"] = "COMPANY"
            return e
        if fruit and not tech:
            return None
        if tech and fruit:
            e["type"] = "COMPANY"
            return e
        # Bare \"Apple\" / PRODUCT label in business podcasts: treat as the company, not the fruit.
        if typ == "TECHNOLOGY":
            e["type"] = "COMPANY"
            return e
        if typ == "COMPANY":
            return e
        if typ == "PERSON":
            e["type"] = "COMPANY"
            return e
        e["type"] = "COMPANY"
        return e

    if low == "amazon":
        corp = _AMAZON_CORP.search(chunk_lower)
        river = _AMAZON_RIVER.search(chunk_lower)
        if corp and not river:
            e["type"] = "COMPANY"
            return e
        if river and not corp and typ == "COMPANY":
            e["type"] = "PLACE"
            return e
        return e

    if low == "oracle":
        sw = _ORACLE_TECH.search(chunk_lower)
        myth = _ORACLE_MYTH.search(chunk_lower)
        if sw and not myth:
            e["type"] = "COMPANY"
            return e
        if myth and not sw:
            e["type"] = "EVENT"
            return e
        return e

    if low == "meta":
        if _META_CORP.search(chunk_lower):
            e["type"] = "COMPANY"
        return e

    return e


def _refine_entities_with_chunk_context(
    chunks: list[dict[str, Any]], entities: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_id = {int(c["id"]): c for c in chunks if c.get("id") is not None}
    out: list[dict[str, Any]] = []
    for ent in entities:
        try:
            cid = int(ent.get("chunk_id", -1))
        except (TypeError, ValueError):
            out.append(ent)
            continue
        ch = by_id.get(cid)
        chunk_lower = ""
        if ch is not None:
            chunk_lower = (ch.get("text_clean") or ch.get("text") or "").lower()
        refined = _apply_context_disambiguation(ent, chunk_lower)
        if refined is not None:
            out.append(refined)
    return out


# --- spaCy ---
_spacy_nlp = None


def _get_spacy():
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy

        model = os.environ.get("SPACY_MODEL", "en_core_web_sm")
        try:
            _spacy_nlp = spacy.load(model)
        except OSError as e:
            logger.error("spaCy model %r not installed.", model)
            raise RuntimeError(
                f"spaCy model {model!r} is missing. From the project venv run one of:\n"
                f"  python -m spacy download {model}\n"
                "  pip install -r server/requirements.txt\n"
                "(requirements.txt includes the en_core_web_sm wheel.)"
            ) from e
    return _spacy_nlp


# spaCy label -> our tag (None = drop). Tune here if your domain needs more labels.
_SPACY_MAP: dict[str, str] = {
    "PERSON": "PERSON",
    "PER": "PERSON",
    "GPE": "PLACE",
    "LOC": "PLACE",
    "FAC": "PLACE",
    "ORG": "COMPANY",
    "PRODUCT": "TECHNOLOGY",
    "EVENT": "EVENT",
    "WORK_OF_ART": "EVENT",
    "LANGUAGE": "TECHNOLOGY",  # spoken languages, stacks mentioned as languages
    "LAW": "EVENT",  # named laws, cases — often discussed like events in podcasts
    "NORP": "MISC",  # nationalities, religions, political groups (CoNLL-style misc)
}


def extract_with_spacy(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    nlp = _get_spacy()
    entities: list[dict[str, Any]] = []
    for ch in chunks:
        cid = int(ch.get("id", 0))
        t0, t1 = float(ch["start"]), float(ch["end"])
        raw = ch.get("text") or ""
        cleaned = filter_noise(raw)
        if not cleaned:
            continue
        doc = nlp(cleaned)
        for ent in doc.ents:
            tag = _SPACY_MAP.get(ent.label_)
            if not tag or tag not in ENTITY_TYPES:
                continue
            text = _clean_entity_text(ent.text)
            if not text:
                continue
            es, ee = _entity_time_span(t0, t1, cleaned, ent.start_char, ent.end_char)
            entities.append(
                {
                    "type": tag,
                    "text": text,
                    "start_sec": round(es, 3),
                    "end_sec": round(ee, 3),
                    "chunk_id": cid,
                    "source": "spacy",
                    "original_label": ent.label_,
                }
            )
    return entities


# --- Claude (classic NER: CoNLL-style PER / ORG / LOC / MISC) ---
_CLAUDE_SYSTEM = """You perform classic Named Entity Recognition on podcast transcript chunks.
Use the standard CoNLL-2003 style taxonomy (4 types). Return ONLY valid JSON (no markdown). Schema:
{"entities":[{"type":"PERSON|ORG|LOC|MISC","text":"exact surface span from the chunk","chunk_id":number}]}

Definitions:
- PERSON: people, including titles if part of the name span (e.g. "Dr. Smith"). No generic roles alone ("the host").
- ORG: companies, agencies, institutions, political parties, sports teams when named as organizations.
- LOC: locations — cities, countries, mountains, rivers, named venues/regions, addresses. Include the Amazon River here, not under ORG.
- MISC: miscellaneous named entities that are not PERSON/ORG/LOC: languages, nationalities/ethnic groups, events, laws, works of art,
  religions, named products/technologies when they are not clearly an ORG (e.g. "Python" the language → MISC), wars, holidays.

Rules:
- type must be exactly one of: PERSON, ORG, LOC, MISC
- Copy "text" as the verbatim substring from that chunk (for alignment). One row per distinct span; do not duplicate the same span+type in a chunk.
- Use chunk wording to disambiguate homonyms: "Apple" + iPhone/Mac/Cupertino → ORG; "apple" + pie/orchard/fruit → omit as a company.
  "Amazon" + AWS/Prime/Bezos → ORG; Amazon as the river/rainforest → LOC. "Oracle" software company → ORG; Greek oracle/myth → MISC.
  "Meta" + Facebook/Instagram → ORG.
- Omit filler (um, uh), pronouns, generic words, and non-entity mentions
"""

# Map classic NER labels (and common aliases) to our app types (ORG→COMPANY, LOC→PLACE).
_CLASSIC_NER_ALIASES: dict[str, str] = {
    "PER": "PERSON",
    "PERSON": "PERSON",
    "ORG": "ORG",
    "ORGANIZATION": "ORG",
    "LOC": "LOC",
    "LOCATION": "LOC",
    "GPE": "LOC",
    "FAC": "LOC",
    "MISC": "MISC",
    "MISCELLANEOUS": "MISC",
}

_CLASSIC_NER_TO_APP: dict[str, str] = {
    "PERSON": "PERSON",
    "ORG": "COMPANY",
    "LOC": "PLACE",
    "MISC": "MISC",
}


def _normalize_claude_ner_type(raw: str) -> str | None:
    """Accept classic CoNLL-style labels or legacy app labels; return ENTITY_TYPES member or None."""
    t = str(raw or "").strip().upper()
    if not t:
        return None
    if t in ENTITY_TYPES:
        return t
    bucket = _CLASSIC_NER_ALIASES.get(t)
    if bucket and bucket in _CLASSIC_NER_TO_APP:
        return _CLASSIC_NER_TO_APP[bucket]
    return None


def extract_with_claude(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import anthropic

    load_dotenv(_ENTITY_ENV_ROOT / ".env", encoding="utf-8-sig")
    api_key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if not api_key:
        env_file = _ENTITY_ENV_ROOT / ".env"
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set for Claude entity extraction. "
            f"Set ANTHROPIC_API_KEY in {env_file} (no quotes) and restart the API server."
        )

    model = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
    client = anthropic.Anthropic(api_key=api_key)

    # Batch chunks to control tokens (~15 per call)
    batch_size = int(os.environ.get("CLAUDE_ENTITY_BATCH", "12"))
    all_entities: list[dict[str, Any]] = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        lines = []
        for ch in batch:
            cid = int(ch.get("id", 0))
            cleaned = filter_noise(ch.get("text") or "")
            if not cleaned:
                continue
            lines.append(f"[chunk_id={cid} start={ch['start']:.2f} end={ch['end']:.2f}]\n{cleaned}")
        if not lines:
            continue
        user_block = "Chunks:\n\n" + "\n\n---\n\n".join(lines)

        msg = client.messages.create(
            model=model,
            max_tokens=4096,
            system=_CLAUDE_SYSTEM,
            messages=[{"role": "user", "content": user_block}],
        )
        raw_text = ""
        for block in msg.content:
            if block.type == "text":
                raw_text += block.text
        raw_text = raw_text.strip()
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as e:
            logger.warning("Claude JSON parse failed: %s | snippet=%r", e, raw_text[:400])
            continue

        ents = data.get("entities") if isinstance(data, dict) else None
        if not isinstance(ents, list):
            continue

        by_id = {int(c["id"]): c for c in batch if "id" in c}
        for item in ents:
            if not isinstance(item, dict):
                continue
            raw_label = str(item.get("type", "")).strip().upper()
            typ = _normalize_claude_ner_type(raw_label)
            if not typ:
                continue
            text = _clean_entity_text(str(item.get("text", "")))
            if not text:
                continue
            try:
                cid = int(item.get("chunk_id", -1))
            except (TypeError, ValueError):
                continue
            ch = by_id.get(cid)
            if not ch:
                continue
            cleaned = filter_noise(ch.get("text") or "")
            if not cleaned:
                continue
            # Approximate char span: find first occurrence
            idx = cleaned.lower().find(text.lower())
            if idx < 0:
                es = float(ch["start"])
                ee = float(ch["end"])
            else:
                es, ee = _entity_time_span(
                    float(ch["start"]), float(ch["end"]), cleaned, idx, idx + len(text)
                )
            row: dict[str, Any] = {
                "type": typ,
                "text": text,
                "start_sec": round(es, 3),
                "end_sec": round(ee, 3),
                "chunk_id": cid,
                "source": "claude",
            }
            if raw_label and raw_label != typ:
                row["original_label"] = raw_label
            all_entities.append(row)

    return all_entities


def run_extraction(
    chunks: list[dict[str, Any]],
    backend: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Returns (chunks_with_cleaned_text, entities).
    Each chunk dict should have id, start, end, text.
    """
    b = resolve_entity_backend(backend)

    normalized: list[dict[str, Any]] = []
    for i, ch in enumerate(chunks):
        cid = ch.get("id")
        if cid is None:
            cid = i
        cleaned = filter_noise(str(ch.get("text", "")))
        normalized.append(
            {
                "id": int(cid),
                "start": float(ch["start"]),
                "end": float(ch["end"]),
                "text": str(ch.get("text") or ""),
                "text_clean": cleaned,
            }
        )

    if b == "claude":
        entities = extract_with_claude([{**c, "text": c["text_clean"] or c["text"]} for c in normalized])
    else:
        entities = extract_with_spacy([{**c, "text": c["text_clean"] or c["text"]} for c in normalized])

    entities = _refine_entities_with_chunk_context(normalized, entities)
    return normalized, entities


def build_document(
    *,
    chunks: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    source_label: str | None,
    backend: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "source_label": source_label,
        "chunks": [
            {
                "id": c["id"],
                "start_sec": round(float(c["start"]), 3),
                "end_sec": round(float(c["end"]), 3),
                "text_raw": c.get("text", ""),
                "text_clean": c.get("text_clean", ""),
            }
            for c in chunks
        ],
        "entities": entities,
    }


def save_document(doc: dict[str, Any], directory: Path, basename: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w\-.]", "_", basename, flags=re.UNICODE)[:80] or "entities"
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = directory / f"{safe}_{ts}.json"
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return path.resolve()
