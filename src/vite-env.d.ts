/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Full URL to POST transcribe, e.g. https://api.example.com/api/transcribe (required for GitHub Pages). */
  readonly VITE_TRANSCRIBE_URL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
