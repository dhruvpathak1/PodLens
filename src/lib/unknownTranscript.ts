import type { TranscriptSegment } from './transcribeAudio'

/** Shown when a live chunk request fails (network / server error). */
export const UNKNOWN_TRANSCRIPT_SENTENCE = 'Unknown Sentence'

export function unknownLiveTranscriptSegments(
  chunkIndex: number,
  windowSec: number
): TranscriptSegment[] {
  const base = chunkIndex * 1000
  const off = chunkIndex * windowSec
  return [{ id: base, start: off, end: off + windowSec, text: UNKNOWN_TRANSCRIPT_SENTENCE }]
}
