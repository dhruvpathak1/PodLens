import type { EntityRecord } from './extractEntities'

const DEFAULT_URL = '/api/enrich-entities'

export type WikipediaInfo = {
  title: string
  extract: string
  url: string
  thumbnail?: string | null
}

export type LocationInfo = {
  lat: number
  lon: number
  display_name: string
  map_embed_url: string
  openstreetmap_url: string
}

export type UnsplashPhotoInfo = {
  image_url: string
  thumb_url?: string | null
  alt?: string
  photographer_name: string
  photographer_url: string
  unsplash_url: string
}

export type EnrichedEntityCard = {
  id: string
  type: string
  text: string
  start_sec: number
  end_sec: number
  chunk_id: number
  wikipedia: WikipediaInfo | null
  location: LocationInfo | null
  unsplash: UnsplashPhotoInfo | null
}

export type EnrichEntitiesResponse = {
  cards: EnrichedEntityCard[]
  count: number
  /** Present when server supports Unsplash enrichment. */
  unsplash_enabled?: boolean
}

function viteTranscribeUrl(): string | undefined {
  const v = import.meta.env.VITE_TRANSCRIBE_URL
  return typeof v === 'string' && v.trim() ? v.trim() : undefined
}

function resolveEndpoint(explicit?: string): string {
  if (explicit) return explicit.replace(/\/$/, '')
  const base = viteTranscribeUrl()
  if (!base) return DEFAULT_URL
  const root = base.replace(/\/$/, '').replace(/\/api\/transcribe$/i, '')
  return `${root}/api/enrich-entities`
}

function formatDetail(j: unknown): string | null {
  if (!j || typeof j !== 'object') return null
  const d = (j as Record<string, unknown>).detail
  return typeof d === 'string' ? d : null
}

export async function enrichEntityCards(
  entities: EntityRecord[],
  options: { endpoint?: string } = {}
): Promise<EnrichEntitiesResponse> {
  const endpoint = resolveEndpoint(options.endpoint)
  const res = await fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      entities: entities.map((e) => ({
        type: e.type,
        text: e.text,
        start_sec: e.start_sec,
        end_sec: e.end_sec,
        chunk_id: e.chunk_id,
      })),
    }),
  })
  if (!res.ok) {
    let msg = res.statusText
    try {
      const j = await res.json()
      const d = formatDetail(j)
      if (d) msg = d
    } catch {
      /* keep */
    }
    throw new Error(msg || `Enrich failed (${res.status})`)
  }
  return res.json() as Promise<EnrichEntitiesResponse>
}
