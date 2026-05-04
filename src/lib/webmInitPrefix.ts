/**
 * MediaRecorder `start(timeslice)` emits a full WebM on the first slice, then
 * continuation slices that are often **cluster-only** (no EBML/Segment/Tracks).
 * ffmpeg cannot decode those alone; prepending the init bytes from the first
 * slice produces a valid standalone WebM for each window.
 *
 * @see https://www.matroska.org/technical/specs/index.html (Cluster element)
 */

/** Matroska Cluster element ID (4-byte EBML ID). */
const CLUSTER_ID = [0x1f, 0x43, 0xb6, 0x75] as const

/**
 * Byte offset of the first top-level Cluster in a WebM/Matroska bytestream, or -1.
 * Skips the first ~128 bytes to avoid matching codec private / binary noise.
 */
export function findWebmFirstClusterOffset(bytes: Uint8Array, searchStart = 128): number {
  const n = bytes.length - CLUSTER_ID.length
  for (let i = Math.max(0, searchStart); i <= n; i++) {
    let ok = true
    for (let j = 0; j < CLUSTER_ID.length; j++) {
      if (bytes[i + j] !== CLUSTER_ID[j]) {
        ok = false
        break
      }
    }
    if (ok) return i
  }
  return -1
}

/** EBML + Segment metadata + Tracks, ending before the first Cluster. */
export function extractWebmInitPrefix(firstChunk: Uint8Array): Uint8Array | null {
  const off = findWebmFirstClusterOffset(firstChunk)
  if (off <= 0) return null
  return firstChunk.subarray(0, off)
}
