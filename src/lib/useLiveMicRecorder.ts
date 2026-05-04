import { useCallback, useEffect, useRef, useState } from 'react'

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
        void Promise.resolve(onChunkRef.current(ev.data, idx)).catch((err) => {
          console.warn('[PodLens live] onChunk handler error', err)
        })
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
