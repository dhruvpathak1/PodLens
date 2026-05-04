/**
 * Ordered live chunk processing: up to `maxConcurrent` transcribe requests run in parallel,
 * while results are applied to UI/state strictly in chunk index order (so the next 10s can
 * transcribe while earlier chunks finish).
 */

import type { TranscribeResult } from './transcribeAudio'

export const LIVE_TRANSCRIBE_MAX_CONCURRENT = 2

class AsyncSemaphore {
  private active = 0
  private readonly wait: Array<() => void> = []
  private readonly max: number

  constructor(max: number) {
    this.max = max
  }

  async acquire(): Promise<void> {
    if (this.active < this.max) {
      this.active++
      return
    }
    await new Promise<void>((resolve) => {
      this.wait.push(() => {
        this.active++
        resolve()
      })
    })
  }
  release(): void {
    this.active--
    const next = this.wait.shift()
    if (next) next()
  }
}

type BufferOk = { kind: 'ok'; chunkIndex: number; res: TranscribeResult }
type BufferErr = { kind: 'err'; chunkIndex: number; error: unknown }

export type LiveChunkPipeline = {
  submit: (blob: Blob, chunkIndex: number) => void
  reset: () => void
}

export function createOrderedLiveChunkPipeline(opts: {
  maxConcurrent: number
  transcribe: (blob: Blob, chunkIndex: number) => Promise<TranscribeResult>
  apply: (chunkIndex: number, res: TranscribeResult) => void | Promise<void>
  onTranscribeFailure: (chunkIndex: number, error: unknown) => void | Promise<void>
  setBusy: (busy: boolean) => void
}): LiveChunkPipeline {
  const sem = new AsyncSemaphore(opts.maxConcurrent)
  const buffer = new Map<number, BufferOk | BufferErr>()
  let nextApply = 0
  let gen = 0
  let drainTail = Promise.resolve()
  let busyDepth = 0

  const bumpBusy = (delta: number) => {
    busyDepth += delta
    opts.setBusy(busyDepth > 0)
  }

  const enqueueDrain = () => {
    const g = gen
    drainTail = drainTail
      .catch(() => undefined)
      .then(async () => {
        if (g !== gen) return
        while (buffer.has(nextApply)) {
          if (g !== gen) return
          const entry = buffer.get(nextApply)!
          buffer.delete(nextApply)
          nextApply++
          bumpBusy(1)
          try {
            if (entry.kind === 'err') {
              await Promise.resolve(opts.onTranscribeFailure(entry.chunkIndex, entry.error))
            } else {
              await Promise.resolve(opts.apply(entry.chunkIndex, entry.res))
            }
          } finally {
            bumpBusy(-1)
          }
        }
      })
      .catch((err) => {
        console.warn('[PodLens live] drain error', err)
      })
  }

  const submit = (blob: Blob, chunkIndex: number) => {
    const myGen = gen
    void (async () => {
      await sem.acquire()
      bumpBusy(1)
      try {
        const res = await opts.transcribe(blob, chunkIndex)
        if (myGen !== gen) return
        buffer.set(chunkIndex, { kind: 'ok', chunkIndex, res })
      } catch (e) {
        if (myGen !== gen) return
        buffer.set(chunkIndex, { kind: 'err', chunkIndex, error: e })
      } finally {
        bumpBusy(-1)
        sem.release()
      }
      if (myGen !== gen) return
      enqueueDrain()
    })()
  }

  const reset = () => {
    gen++
    buffer.clear()
    nextApply = 0
    busyDepth = 0
    drainTail = Promise.resolve()
    opts.setBusy(false)
  }

  return { submit, reset }
}
