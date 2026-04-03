import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AudioDropZone } from './components/AudioDropZone'
import { MiniAudioPlayer, type MiniAudioPlayerHandle } from './components/MiniAudioPlayer'
import { transcribeAudio, type TranscriptSegment } from './lib/transcribeAudio'
import { EntitySourceCard } from './components/EntitySourceCard'
import { enrichEntityCards, type EnrichedEntityCard } from './lib/enrichEntities'
import type { EntityDocument, EntityRecord } from './lib/extractEntities'
import { getEntitiesActiveAtPlayback } from './lib/liveEntities'
import {
  buildSentencesFromTranscript,
  getActiveSentenceIdAtTime,
  type TimestampedSentence,
} from './lib/sentences'
import './App.css'

const PODLENS_TITLE = 'PodLens'

type JobState = 'idle' | 'transcribing' | 'enriching'

const ENTITY_TYPE_FILTER_OPTIONS = [
  'ALL',
  'PLACE',
  'PERSON',
  'TECHNOLOGY',
  'EVENT',
  'COMPANY',
  'MISC',
] as const

type EntityTypeFilterOption = (typeof ENTITY_TYPE_FILTER_OPTIONS)[number]

function formatTimestampSeconds(sec: number): string {
  const s = Math.max(0, sec)
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const secPart = Math.floor(s % 60)
  return [h, m, secPart].map((n) => String(n).padStart(2, '0')).join(':')
}

function formatTimeRange(start: number, end: number): string {
  return `${formatTimestampSeconds(start)} – ${formatTimestampSeconds(end)}`
}

const LIVE_QUEUE_MAX = 3

function entityMatchKey(type: string, text: string): string {
  return `${type}\0${text.trim().toLowerCase()}`
}

function findEnrichedForEntity(
  cards: EnrichedEntityCard[],
  e: { type: string; text: string }
): EnrichedEntityCard | undefined {
  const t = e.text.trim().toLowerCase()
  return cards.find((c) => c.type === e.type && c.text.trim().toLowerCase() === t)
}

/** Placeholder until enrichment matches this mention; swapped for full card when available. */
function minimalEnrichedCard(e: EntityRecord): EnrichedEntityCard {
  return {
    id: `pending-${entityMatchKey(e.type, e.text)}`,
    type: e.type,
    text: e.text,
    start_sec: e.start_sec,
    end_sec: e.end_sec,
    chunk_id: e.chunk_id,
    wikipedia: null,
    location: null,
    unsplash: null,
  }
}

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [transcript, setTranscript] = useState<string | null>(null)
  const [segments, setSegments] = useState<TranscriptSegment[]>([])
  const [savedPath, setSavedPath] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [job, setJob] = useState<JobState>('idle')
  const [searchOpen, setSearchOpen] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedSentence, setSelectedSentence] = useState<TimestampedSentence | null>(null)
  const [entityDoc, setEntityDoc] = useState<EntityDocument | null>(null)
  const [entitySavedPath, setEntitySavedPath] = useState<string | null>(null)
  const [entityError, setEntityError] = useState<string | null>(null)
  const [enrichedCards, setEnrichedCards] = useState<EnrichedEntityCard[]>([])
  const [enrichError, setEnrichError] = useState<string | null>(null)
  const [unsplashHint, setUnsplashHint] = useState<boolean | null>(null)
  const [entityTypeFilter, setEntityTypeFilter] = useState<EntityTypeFilterOption>('ALL')
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [playbackTime, setPlaybackTime] = useState(0)
  const [playbackDuration, setPlaybackDuration] = useState(0)
  const [liveRollingCards, setLiveRollingCards] = useState<EnrichedEntityCard[]>([])
  const playerRef = useRef<MiniAudioPlayerHandle>(null)
  const feedRef = useRef<HTMLDivElement>(null)
  const prevActivePlayIdRef = useRef<string | null>(null)
  const prevLiveActiveKeysRef = useRef<Set<string>>(new Set())

  useEffect(() => {
    document.title = PODLENS_TITLE
  }, [])

  const busy = job !== 'idle'

  const viteEntityBackend = import.meta.env.VITE_ENTITY_BACKEND
  const entityBackendOption =
    viteEntityBackend === 'spacy' || viteEntityBackend === 'claude' ? viteEntityBackend : undefined

  const entityFilterCounts = useMemo(() => {
    if (!entityDoc?.entities.length) return null
    const byType = new Map<string, number>()
    for (const e of entityDoc.entities) {
      byType.set(e.type, (byType.get(e.type) ?? 0) + 1)
    }
    const counts: Record<EntityTypeFilterOption, number> = {
      ALL: entityDoc.entities.length,
      PLACE: byType.get('PLACE') ?? 0,
      PERSON: byType.get('PERSON') ?? 0,
      TECHNOLOGY: byType.get('TECHNOLOGY') ?? 0,
      EVENT: byType.get('EVENT') ?? 0,
      COMPANY: byType.get('COMPANY') ?? 0,
      MISC: byType.get('MISC') ?? 0,
    }
    return counts
  }, [entityDoc])

  const visibleEnrichedCards = useMemo(() => {
    if (entityTypeFilter === 'ALL') return enrichedCards
    return enrichedCards.filter((c) => c.type === entityTypeFilter)
  }, [enrichedCards, entityTypeFilter])

  const liveEntitySource = useMemo(() => {
    if (!entityDoc?.entities.length) return []
    if (entityTypeFilter === 'ALL') return entityDoc.entities
    return entityDoc.entities.filter((e) => e.type === entityTypeFilter)
  }, [entityDoc, entityTypeFilter])

  const liveActiveEntities = useMemo(
    () => getEntitiesActiveAtPlayback(liveEntitySource, playbackTime),
    [liveEntitySource, playbackTime]
  )

  useEffect(() => {
    setLiveRollingCards([])
    prevLiveActiveKeysRef.current = new Set()
  }, [entityTypeFilter])

  useEffect(() => {
    if (!enrichedCards.length) return
    setLiveRollingCards((prev) =>
      prev.map((c) => {
        const full = findEnrichedForEntity(enrichedCards, c)
        return full ?? c
      })
    )
  }, [enrichedCards])

  useEffect(() => {
    if (!audioUrl) return
    if (liveEntitySource.length === 0) {
      prevLiveActiveKeysRef.current = new Set()
      return
    }
    const currentKeys = new Set(
      liveActiveEntities.map((e) => entityMatchKey(e.type, e.text))
    )
    const prevKeys = prevLiveActiveKeysRef.current
    const newlyActive = liveActiveEntities.filter(
      (e) => !prevKeys.has(entityMatchKey(e.type, e.text))
    )
    prevLiveActiveKeysRef.current = currentKeys

    if (!newlyActive.length) return
    setLiveRollingCards((prevQ) => {
      const inQueue = new Set(prevQ.map((c) => entityMatchKey(c.type, c.text)))
      const next = [...prevQ]
      for (const e of newlyActive) {
        const k = entityMatchKey(e.type, e.text)
        if (inQueue.has(k)) continue
        const full = findEnrichedForEntity(enrichedCards, e)
        next.push(full ?? minimalEnrichedCard(e))
        inQueue.add(k)
      }
      while (next.length > LIVE_QUEUE_MAX) next.shift()
      return next
    })
  }, [audioUrl, liveActiveEntities, liveEntitySource.length, enrichedCards])

  useEffect(() => {
    if (!file) {
      setAudioUrl(null)
      return
    }
    const url = URL.createObjectURL(file)
    setAudioUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  useEffect(() => {
    setPlaybackTime(0)
    setPlaybackDuration(0)
    setLiveRollingCards([])
    prevLiveActiveKeysRef.current = new Set()
  }, [audioUrl])

  useEffect(() => {
    prevActivePlayIdRef.current = null
  }, [transcript, audioUrl])

  const allSentences = useMemo(
    () => buildSentencesFromTranscript(transcript, segments),
    [transcript, segments]
  )

  const visibleSentences = useMemo(() => {
    const q = searchQuery.trim().toLowerCase()
    if (!q) return allSentences
    return allSentences.filter((s) => s.text.toLowerCase().includes(q))
  }, [allSentences, searchQuery])

  const runPodLens = useCallback(async () => {
    if (!file || busy) return
    setError(null)
    setTranscript(null)
    setSegments([])
    setSavedPath(null)
    setSelectedSentence(null)
    setEntityDoc(null)
    setEntitySavedPath(null)
    setEntityError(null)
    setEnrichedCards([])
    setEnrichError(null)
    setUnsplashHint(null)
    setEntityTypeFilter('ALL')
    setLiveRollingCards([])
    prevLiveActiveKeysRef.current = new Set()
    setJob('transcribing')
    try {
      const {
        transcript: text,
        segments: segs,
        savedPath: path,
        document,
        entitySavedPath: entPath,
        entityError: entErr,
      } = await transcribeAudio(file, { backend: entityBackendOption })
      setTranscript(text)
      setSegments(segs)
      setSavedPath(path ?? null)
      setEntityDoc(document ?? null)
      setEntitySavedPath(entPath ?? null)
      setEntityError(entErr ?? null)

      const doc = document ?? null
      if (doc?.entities.length) {
        setJob('enriching')
        setEnrichError(null)
        try {
          const res = await enrichEntityCards(doc.entities)
          setEnrichedCards(res.cards)
          setUnsplashHint(res.unsplash_enabled ?? null)
        } catch (e) {
          setEnrichError(e instanceof Error ? e.message : 'Enrichment failed')
          setUnsplashHint(null)
        }
      } else {
        setEnrichedCards([])
        setUnsplashHint(null)
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Transcription failed')
    } finally {
      setJob('idle')
    }
  }, [file, busy, entityBackendOption])

  const statusLabel =
    job === 'transcribing'
      ? 'Transcribing…'
      : job === 'enriching'
        ? 'Fetching sources…'
        : transcript
          ? 'Synced'
          : 'Ready'
  const hasTimedSentences = allSentences.some((s) => s.start > 0 || s.end > 0)

  const activePlaySentenceId = useMemo(() => {
    if (!hasTimedSentences || !allSentences.length) return null
    return getActiveSentenceIdAtTime(allSentences, playbackTime, playbackDuration)
  }, [hasTimedSentences, allSentences, playbackTime, playbackDuration])

  const handlePlaybackTick = useCallback((t: number, d: number) => {
    setPlaybackTime(t)
    setPlaybackDuration(d)
  }, [])

  useEffect(() => {
    if (!hasTimedSentences || !activePlaySentenceId) return
    if (prevActivePlayIdRef.current === activePlaySentenceId) return
    prevActivePlayIdRef.current = activePlaySentenceId
    const root = feedRef.current
    if (!root) return
    const el = root.querySelector<HTMLElement>(
      `[data-sentence-id="${CSS.escape(activePlaySentenceId)}"]`
    )
    el?.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
  }, [activePlaySentenceId, hasTimedSentences])

  const selectSentence = useCallback((s: TimestampedSentence) => {
    setSelectedSentence(s)
    playerRef.current?.seekTo(s.start)
  }, [])

  const onSentenceClick = useCallback(
    (s: TimestampedSentence) => {
      selectSentence(s)
    },
    [selectSentence]
  )

  const onSentenceKeyDown = useCallback(
    (e: React.KeyboardEvent, s: TimestampedSentence) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault()
        selectSentence(s)
      }
    },
    [selectSentence]
  )

  return (
    <div className="dashboard">
      <div className="dashboard__grid">
        <aside className="transcript-sidebar" aria-label="Live transcript">
          <div className="transcript-sidebar__head">
            <div className="transcript-sidebar__titles">
              <h2 className="transcript-sidebar__label">Live transcript</h2>
              <p className={`transcript-sidebar__status${busy ? ' transcript-sidebar__status--busy' : ''}`}>
                <span className="transcript-sidebar__status-dot" aria-hidden />
                {statusLabel}
              </p>
            </div>
            <button
              type="button"
              className="icon-btn"
              aria-expanded={searchOpen}
              aria-label={searchOpen ? 'Close search' : 'Search transcript'}
              onClick={() => {
                setSearchOpen((v) => !v)
                if (searchOpen) setSearchQuery('')
              }}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="11" cy="11" r="7" />
                <path d="M21 21l-4.35-4.35" strokeLinecap="round" />
              </svg>
            </button>
          </div>

          {searchOpen && (
            <label className="transcript-search">
              <span className="sr-only">Filter transcript</span>
              <input
                type="search"
                className="transcript-search__input"
                placeholder="Filter sentences…"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                autoComplete="off"
              />
            </label>
          )}

          <div className="transcript-sidebar__ingest">
            <AudioDropZone file={file} onFileChange={setFile} disabled={busy} compact />

            <button
              type="button"
              className="btn btn--primary btn--block btn--podlens"
              disabled={!file || busy}
              onClick={runPodLens}
            >
              {job === 'transcribing'
                ? 'Transcribing & tagging…'
                : job === 'enriching'
                  ? 'Fetching Wikipedia, maps & photos…'
                  : 'PodLens'}
            </button>
          </div>

          <MiniAudioPlayer
            key={audioUrl ?? 'no-audio'}
            ref={playerRef}
            src={audioUrl}
            onPlaybackTick={handlePlaybackTick}
          />

          {error && (
            <div className="alert alert--error" role="alert">
              {error}
            </div>
          )}

          {entityError && (
            <div className="alert alert--error" role="alert">
              {entityError}
            </div>
          )}

          {enrichError && (
            <div className="alert alert--error" role="alert">
              {enrichError}
            </div>
          )}

          <div
            ref={feedRef}
            className="transcript-sidebar__feed"
            role="region"
            aria-label="Transcript sentences"
          >
            {busy && (
              <div className="skeleton skeleton--in-feed" aria-live="polite">
                <div className="skeleton__line" />
                <div className="skeleton__line skeleton__line--short" />
                <div className="skeleton__line" />
              </div>
            )}

            {!busy &&
              visibleSentences.map((s) => {
                const isSelected = selectedSentence?.id === s.id
                const isPlayingHere = hasTimedSentences && activePlaySentenceId === s.id
                return (
                  <button
                    key={s.id}
                    type="button"
                    data-sentence-id={s.id}
                    className={`transcript-sentence${isSelected ? ' transcript-sentence--selected' : ''}${
                      isPlayingHere ? ' transcript-sentence--playing' : ''
                    }`}
                    onClick={() => onSentenceClick(s)}
                    onKeyDown={(e) => onSentenceKeyDown(e, s)}
                    aria-pressed={isSelected}
                    aria-current={isPlayingHere ? 'location' : undefined}
                  >
                    <span className="transcript-sentence__meta">
                      <span className="transcript-sentence__speaker">Sentence</span>
                      <time
                        className="transcript-sentence__time"
                        dateTime={hasTimedSentences ? `PT${Math.floor(s.start)}S` : undefined}
                      >
                        {hasTimedSentences ? formatTimeRange(s.start, s.end) : '—'}
                      </time>
                    </span>
                    <span className="transcript-sentence__text">{s.text}</span>
                  </button>
                )
              })}

            {!busy && !transcript && !error && (
              <p className="transcript-sidebar__empty">Run <strong>PodLens</strong> to transcribe and load tags here.</p>
            )}

            {!busy && transcript && visibleSentences.length === 0 && searchQuery && (
              <p className="transcript-sidebar__empty">No sentences match this filter.</p>
            )}
          </div>

          {savedPath && (
            <p className="transcript-sidebar__saved" title={savedPath}>
              Saved: <code>{savedPath.split('/').pop()}</code>
            </p>
          )}
        </aside>

        <main className="main-stage">
          <div
            className="main-stage__canvas"
            role="region"
            aria-label="Workspace, entities, and source cards"
          >
            {entityDoc && entityDoc.entities.length > 0 && (
              <section className="live-workspace" aria-labelledby="live-workspace-heading">
                <div className="live-workspace__head">
                  <h2 id="live-workspace-heading" className="live-workspace__title">
                    <span className="live-workspace__pulse" aria-hidden />
                    Live
                  </h2>
                  {audioUrl ? (
                    <time
                      className="live-workspace__clock"
                      dateTime={`PT${Math.floor(playbackTime)}S`}
                    >
                      {formatTimestampSeconds(playbackTime)}
                      {playbackDuration > 0
                        ? ` / ${formatTimestampSeconds(playbackDuration)}`
                        : ''}
                    </time>
                  ) : null}
                </div>
                <div className="live-workspace__body" aria-live="polite">
                  {!audioUrl ? (
                    <p className="live-workspace__hint">
                      Load audio and press play — up to three source cards queue here as each new mention{' '}
                      <strong>starts</strong>; the oldest drops off when a fourth begins.
                    </p>
                  ) : liveEntitySource.length === 0 ? (
                    <p className="live-workspace__hint">
                      No tags match the current <strong>type filter</strong> for live follow-along.
                    </p>
                  ) : liveRollingCards.length === 0 ? (
                    <p className="live-workspace__hint">
                      Play the audio — full cards appear when a tagged mention begins. Up to {LIVE_QUEUE_MAX} stay on
                      screen; older ones roll off automatically.
                    </p>
                  ) : (
                    <div className="live-workspace__cards" aria-label="Live rolling source cards">
                      {liveRollingCards.map((c) => (
                        <div key={entityMatchKey(c.type, c.text)} className="live-workspace__card-wrap">
                          <EntitySourceCard card={c} formatTimeRange={formatTimeRange} />
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </section>
            )}

            {entityDoc && (
              <div className="entity-panel entity-panel--filter-only">
                {entityDoc.entities.length === 0 ? (
                  <p className="entity-panel__empty">No entities found after tagging and noise filtering.</p>
                ) : (
                  <div className="entity-filter-toolbar">
                    <div className="entity-filter-toolbar__top">
                      <h2 className="entity-filter-toolbar__title">Tagged entities</h2>
                      <span className="entity-filter-toolbar__meta">
                        {entityDoc.backend}
                        {entitySavedPath && (
                          <>
                            {' · '}
                            <code title={entitySavedPath}>{entitySavedPath.split('/').pop()}</code>
                          </>
                        )}
                      </span>
                    </div>
                    <p className="entity-filter-toolbar__hint">
                      Choose a type to limit <strong>Live</strong> highlights and <strong>Source cards</strong>. Numbers
                      are how many tagged spans of that type appear in the export.
                    </p>
                    <div
                      className="entity-filter__group"
                      role="group"
                      aria-label="Filter by entity type"
                    >
                      {ENTITY_TYPE_FILTER_OPTIONS.map((opt) => {
                        const count = entityFilterCounts?.[opt] ?? 0
                        const label = opt === 'ALL' ? 'All types' : opt
                        const isActive = entityTypeFilter === opt
                        const isEmptyType = opt !== 'ALL' && count === 0
                        return (
                          <button
                            key={opt}
                            type="button"
                            className={`entity-filter__opt${isActive ? ' entity-filter__opt--active' : ''}${
                              isEmptyType && !isActive ? ' entity-filter__opt--dim' : ''
                            }`}
                            aria-pressed={isActive}
                            aria-label={`${label}, ${count} ${count === 1 ? 'mention' : 'mentions'}`}
                            onClick={() => setEntityTypeFilter(opt)}
                          >
                            <span className="entity-filter__opt-label">{label}</span>
                            <span className="entity-filter__opt-count" aria-hidden>
                              {count}
                            </span>
                          </button>
                        )
                      })}
                    </div>
                  </div>
                )}
                {unsplashHint === false && enrichedCards.length > 0 && (
                  <p className="entity-panel__hint">
                    Unsplash photos skipped — add <code className="app-inline-code">UNSPLASH_ACCESS_KEY</code>{' '}
                    (Client ID from{' '}
                    <a
                      href="https://unsplash.com/developers"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="entity-panel__hint-link"
                    >
                      unsplash.com/developers
                    </a>
                    ) to the API server env.
                  </p>
                )}
              </div>
            )}

            {enrichedCards.length > 0 && (
              <div className="entity-cards-section">
                <h2 className="entity-cards-section__title">Source cards</h2>
                <p className="entity-cards-section__lead">
                  One card per unique tag: Wikipedia REST API, Unsplash when configured, and OpenStreetMap Nominatim.
                  Use the <strong>type filter</strong> above to limit which cards appear.
                </p>
                {visibleEnrichedCards.length > 0 ? (
                  <div className="entity-cards-grid">
                    {visibleEnrichedCards.map((c) => (
                      <EntitySourceCard key={c.id} card={c} formatTimeRange={formatTimeRange} />
                    ))}
                  </div>
                ) : (
                  <p className="entity-cards-section__empty">
                    No source cards for this type. Choose <strong>ALL</strong> or another category.
                  </p>
                )}
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  )
}
