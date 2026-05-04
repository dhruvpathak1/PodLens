import type { EntityDocument } from './extractEntities'

const DEFAULT_URL = '/api/transcribe'

function viteTranscribeUrl(): string | undefined {
  const v = import.meta.env.VITE_TRANSCRIBE_URL
  return typeof v === 'string' && v.trim() ? v.trim() : undefined
}

export type TranscribeOptions = {
  endpoint?: string
  /** Sent as optional form field `language` if set */
  language?: string
  /** When true (default), server runs entity extraction after Whisper. */
  extractEntities?: boolean
  /** Optional: `spacy` | `claude` — forwarded as `entity_backend`. */
  backend?: 'spacy' | 'claude'
}

export type TranscriptSegment = {
  id: number
  start: number
  end: number
  text: string
}

export type TranscribeResult = {
  transcript: string
  segments: TranscriptSegment[]
  savedPath?: string
  document?: EntityDocument | null
  entitySavedPath?: string | null
  entityError?: string | null
}

function resolveEndpoint(explicit?: string): string {
  return (explicit ?? viteTranscribeUrl() ?? DEFAULT_URL).replace(/\/$/, '')
}

function pickTranscriptFromJson(data: unknown): string | null {
  if (!data || typeof data !== 'object') return null
  const o = data as Record<string, unknown>
  if (typeof o.transcript === 'string') return o.transcript
  if (typeof o.text === 'string') return o.text
  if (Array.isArray(o.segments)) {
    const parts = o.segments
      .map((s) => {
        if (s && typeof s === 'object' && typeof (s as { text?: string }).text === 'string') {
          return (s as { text: string }).text
        }
        return ''
      })
      .filter(Boolean)
    if (parts.length) return parts.join(' ')
  }
  return null
}

function pickSegmentsFromJson(data: unknown): TranscriptSegment[] {
  if (!data || typeof data !== 'object') return []
  const o = data as Record<string, unknown>
  const raw = o.segments
  if (!Array.isArray(raw)) return []
  const out: TranscriptSegment[] = []
  for (let i = 0; i < raw.length; i++) {
    const item = raw[i]
    if (!item || typeof item !== 'object') continue
    const s = item as Record<string, unknown>
    const text = typeof s.text === 'string' ? s.text.trim() : ''
    if (!text) continue
    const start = typeof s.start === 'number' ? s.start : Number(s.start)
    const end = typeof s.end === 'number' ? s.end : Number(s.end)
    if (!Number.isFinite(start) || !Number.isFinite(end)) continue
    const id = typeof s.id === 'number' && Number.isFinite(s.id) ? s.id : i
    out.push({ id, start, end, text })
  }
  return out
}

function pickSavedPathFromJson(data: unknown): string | undefined {
  if (!data || typeof data !== 'object') return undefined
  const o = data as Record<string, unknown>
  if (typeof o.saved_path === 'string' && o.saved_path.trim()) return o.saved_path.trim()
  return undefined
}

function pickDocumentFromJson(data: unknown): EntityDocument | null {
  if (!data || typeof data !== 'object') return null
  const o = data as Record<string, unknown>
  const doc = o.document
  if (!doc || typeof doc !== 'object') return null
  const d = doc as Record<string, unknown>
  if (!Array.isArray(d.entities)) return null
  return doc as EntityDocument
}

function pickEntitySavedPathFromJson(data: unknown): string | null {
  if (!data || typeof data !== 'object') return null
  const p = (data as Record<string, unknown>).entity_saved_path
  if (typeof p === 'string' && p.trim()) return p.trim()
  return null
}

function pickEntityErrorFromJson(data: unknown): string | null {
  if (!data || typeof data !== 'object') return null
  const e = (data as Record<string, unknown>).entity_error
  if (typeof e === 'string' && e.trim()) return e.trim()
  return null
}

function formatErrorDetail(j: unknown): string | null {
  if (!j || typeof j !== 'object') return null
  const o = j as Record<string, unknown>
  const detail = o.detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    const parts = detail
      .map((item) => {
        if (item && typeof item === 'object' && 'msg' in item) {
          const msg = (item as { msg?: unknown }).msg
          if (typeof msg === 'string') return msg
        }
        return typeof item === 'string' ? item : null
      })
      .filter((x): x is string => typeof x === 'string')
    if (parts.length) return parts.join('; ')
  }
  return null
}

export type LiveChunkTranscribeOptions = TranscribeOptions & {
  /** Monotonic index for stable segment IDs server-side (chunk_seq). */
  chunkSeq: number
  /** Seconds to add to Whisper segment times so chunks align on a session timeline. */
  timeOffsetSec: number
  /** When true, server writes transcript / entity JSON to disk (default false). */
  persistTranscript?: boolean
}

function resolveChunkEndpoint(explicit?: string): string {
  const transcribeRoot = resolveEndpoint(explicit)
  return transcribeRoot.replace(/\/transcribe$/i, '/transcribe-chunk')
}

/**
 * Transcribe one live microphone slice (e.g. 10s WebM). Same response shape as {@link transcribeAudio}.
 */
export async function transcribeLiveAudioChunk(
  blob: Blob,
  options: LiveChunkTranscribeOptions
): Promise<TranscribeResult> {
  const endpoint = resolveChunkEndpoint(options.endpoint)
  const body = new FormData()
  body.append('audio', blob, `live-chunk-${options.chunkSeq}.webm`)
  body.append('chunk_seq', String(options.chunkSeq))
  body.append('time_offset_sec', String(options.timeOffsetSec))
  body.append('extract_entities', options.extractEntities === false ? 'false' : 'true')
  body.append('persist_transcript', options.persistTranscript === true ? 'true' : 'false')
  if (options.language) body.append('language', options.language)
  if (options.backend) body.append('entity_backend', options.backend)

  const res = await fetch(endpoint, {
    method: 'POST',
    body,
  })

  const ct = res.headers.get('content-type') ?? ''

  if (!res.ok) {
    let detail = res.statusText
    try {
      if (ct.includes('application/json')) {
        const j = await res.json()
        if (j && typeof j === 'object' && typeof (j as { error?: string }).error === 'string') {
          detail = (j as { error: string }).error
        } else if (j && typeof j === 'object' && typeof (j as { message?: string }).message === 'string') {
          detail = (j as { message: string }).message
        } else {
          const fromDetail = formatErrorDetail(j)
          if (fromDetail) detail = fromDetail
        }
      } else {
        const t = await res.text()
        if (t) detail = t.slice(0, 500)
      }
    } catch {
      /* keep */
    }
    throw new Error(detail || `Request failed (${res.status})`)
  }

  if (ct.includes('application/json')) {
    const data = await res.json()
    const text = pickTranscriptFromJson(data)
    if (text != null) {
      return {
        transcript: text.trim(),
        segments: pickSegmentsFromJson(data),
        savedPath: pickSavedPathFromJson(data),
        document: pickDocumentFromJson(data),
        entitySavedPath: pickEntitySavedPathFromJson(data),
        entityError: pickEntityErrorFromJson(data),
      }
    }
    throw new Error('Response JSON did not include transcript text')
  }

  const text = await res.text()
  if (!text.trim()) throw new Error('Empty transcript response')
  return { transcript: text.trim(), segments: [] }
}

export async function transcribeAudio(
  file: File,
  options: TranscribeOptions = {}
): Promise<TranscribeResult> {
  const endpoint = resolveEndpoint(options.endpoint)
  const body = new FormData()
  body.append('audio', file, file.name)
  if (options.language) body.append('language', options.language)
  body.append('extract_entities', options.extractEntities === false ? 'false' : 'true')
  if (options.backend) body.append('entity_backend', options.backend)

  const res = await fetch(endpoint, {
    method: 'POST',
    body,
  })

  const ct = res.headers.get('content-type') ?? ''

  if (!res.ok) {
    let detail = res.statusText
    try {
      if (ct.includes('application/json')) {
        const j = await res.json()
        if (j && typeof j === 'object' && typeof (j as { error?: string }).error === 'string') {
          detail = (j as { error: string }).error
        } else if (j && typeof j === 'object' && typeof (j as { message?: string }).message === 'string') {
          detail = (j as { message: string }).message
        } else {
          const fromDetail = formatErrorDetail(j)
          if (fromDetail) detail = fromDetail
        }
      } else {
        const t = await res.text()
        if (t) detail = t.slice(0, 500)
      }
    } catch {
      /* use statusText */
    }
    throw new Error(detail || `Request failed (${res.status})`)
  }

  if (ct.includes('application/json')) {
    const data = await res.json()
    const text = pickTranscriptFromJson(data)
    if (text != null) {
      return {
        transcript: text.trim(),
        segments: pickSegmentsFromJson(data),
        savedPath: pickSavedPathFromJson(data),
        document: pickDocumentFromJson(data),
        entitySavedPath: pickEntitySavedPathFromJson(data),
        entityError: pickEntityErrorFromJson(data),
      }
    }
    throw new Error('Response JSON did not include transcript text')
  }

  const text = await res.text()
  if (!text.trim()) throw new Error('Empty transcript response')
  return { transcript: text.trim(), segments: [] }
}

export function getTranscribeEndpointDisplay(): string {
  return resolveEndpoint()
}
