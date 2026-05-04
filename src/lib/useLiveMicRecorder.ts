import { useCallback, useEffect, useRef, useState } from 'react'
import { extractWebmInitPrefix } from './webmInitPrefix'

/** Wall-clock aligned windows sent to the transcription API (matches MediaRecorder timeslice). */
export const LIVE_CHUNK_INTERVAL_MS = 10_000

type Options = {
  onChunk: (blob: Blob, chunkIndex: number) => void | Promise<void>
}

/**
 * Captures microphone audio and yields Blobs every {@link LIVE_CHUNK_INTERVAL_MS} ms for server transcription.
 */
export function useLiveMicRecorder({ onChunk }: Options) {
  const [active, setActive] = useState(false)
  const [micError, setMicError] = useState<string | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunkIndexRef = useRef(0)
  /** EBML/Segment/Tracks prefix from first slice; prepended to later slices so ffmpeg can decode each blob. */
  const webmInitPrefixRef = useRef<Uint8Array | null>(null)
  const onChunkRef = useRef(onChunk)

  useEffect(() => {
    onChunkRef.current = onChunk
  }, [onChunk])

  const stop = useCallback(() => {
    const rec = recorderRef.current
    recorderRef.current = null
    if (rec && rec.state !== 'inactive') {
      try {
        rec.stop()
      } catch {
        streamRef.current?.getTracks().forEach((t) => t.stop())
        streamRef.current = null
        setActive(false)
      }
      return
    }
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    setActive(false)
  }, [])

  const start = useCallback(async () => {
    setMicError(null)
    chunkIndexRef.current = 0
    webmInitPrefixRef.current = null
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream

      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : MediaRecorder.isTypeSupported('audio/webm')
          ? 'audio/webm'
          : ''

      if (!mimeType) {
        setMicError('This browser cannot record audio in a supported format.')
        stream.getTracks().forEach((t) => t.stop())
        streamRef.current = null
        return
      }

      const rec = new MediaRecorder(stream, { mimeType })
      recorderRef.current = rec

      rec.ondataavailable = (ev) => {
        // Do not drop small blobs: short WebM fragments after each timeslice can be <512 bytes
        // and skipping them breaks chunk indexing and the processing chain.
        if (ev.data.size === 0) return
        const idx = chunkIndexRef.current++
        void (async () => {
          try {
            const ab = await ev.data.arrayBuffer()
            const u8 = new Uint8Array(ab)
            let payload: Blob
            if (idx === 0) {
              webmInitPrefixRef.current = extractWebmInitPrefix(u8)
              if (!webmInitPrefixRef.current?.length) {
                console.warn(
                  '[PodLens live] Could not derive WebM init prefix from first slice; later slices may decode as Unknown.'
                )
              }
              payload = new Blob([u8 as BlobPart], { type: mimeType })
            } else {
              const pre = webmInitPrefixRef.current
              payload =
                pre && pre.length > 0
                  ? new Blob([pre as BlobPart, u8 as BlobPart], { type: mimeType })
                  : new Blob([u8 as BlobPart], { type: mimeType })
            }
            await Promise.resolve(onChunkRef.current(payload, idx))
          } catch (err) {
            console.warn('[PodLens live] onChunk handler error', err)
          }
        })()
      }

      rec.onstop = () => {
        streamRef.current?.getTracks().forEach((t) => t.stop())
        streamRef.current = null
        setActive(false)
      }

      rec.start(LIVE_CHUNK_INTERVAL_MS)
      setActive(true)
    } catch (e) {
      setMicError(e instanceof Error ? e.message : 'Microphone access failed')
      streamRef.current?.getTracks().forEach((t) => t.stop())
      streamRef.current = null
      setActive(false)
    }
  }, [])

  return { start, stop, active, micError, clearMicError: () => setMicError(null) }
}
