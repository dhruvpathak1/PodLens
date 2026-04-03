# PodLens

PodLens is an AI-powered context generator for podcasts. It listens to audio, transcribes it, and automatically enriches the content by identifying key references—such as people, companies, locations, and technologies—and adding relevant images, maps, and summaries in real-time.

## Features

- **Audio Transcription:** Powered by OpenAI's Whisper for accurate, timestamped transcripts.
- **Entity Extraction:** Automatically identifies entities like People, Companies, Locations, and Events using either spaCy or Anthropic's Claude.
- **Context Enrichment:**
  - **Wikipedia:** Pulls summaries and thumbnails for identified entities.
  - **OpenStreetMap (Nominatim):** Generates interactive map previews for mentioned locations.
  - **Unsplash:** Finds high-quality, relevant stock photos to visualize the discussion.
- **Interactive UI:** A modern React interface to view transcripts alongside their enriched context "cards".

## APIs & Technology Stack

### Frontend
- **React 19**
- **TypeScript**
- **Vite**
- **Tailwind CSS** (if applicable, though standard CSS is used in some parts)

### Backend
- **FastAPI** (Python)
- **OpenAI Whisper** (Local transcription)
- **spaCy** (Local NLP/NER)
- **Anthropic Claude API** (Optional, for high-accuracy NER)

### External Services
- **Wikipedia API:** For entity summaries.
- **Nominatim (OpenStreetMap):** For geocoding and map embeds.
- **Unsplash API:** For entity-related imagery.

---

## Installation

### Prerequisites
- **Node.js** (v18 or later recommended)
- **Python** (3.9 or later)
- **FFmpeg** (Required by Whisper for audio processing)

### 1. Clone the repository
```bash
git clone <repository-url>
cd podlens
```

### 2. Frontend Setup
```bash
npm install
```

### 3. Backend Setup
```bash
cd server
python -m venv venv

# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

pip install -r requirements.txt
cd ..
```

## Configuration

1. Copy the `.env.example` file to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and fill in your API keys:
   - `UNSPLASH_ACCESS_KEY`: Get one from [Unsplash Developers](https://unsplash.com/developers).
   - `ANTHROPIC_API_KEY`: (Optional) If you want to use Claude for entity extraction.
   - `NOMINATIM_USER_AGENT`: Set this to your app name or email (e.g., `PodLens/1.0 (your@email.com)`) as per Nominatim's usage policy.

## Running the App

You can run both the frontend and the backend simultaneously from the root directory:

```bash
npm run dev
```

- **Frontend:** [http://localhost:5173](http://localhost:5173)
- **Backend API:** [http://127.0.0.1:8000](http://127.0.0.1:8000)

The first time you run a transcription, the Whisper model (default: `base`) will be downloaded automatically. This may take a few minutes depending on your internet connection.

---

## License
[Specify License, e.g., MIT]
