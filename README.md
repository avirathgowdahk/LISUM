# AI-Based Lecture Video Transcription, Intelligent Summarization, and Short Lecture Video Generation

A comprehensive academic project for automated end-to-end processing, transcription, multi-factor educational importance scoring, semantic timestamp alignment, and duration-constrained video generation of educational lecture videos.

## 🎓 Core System Architecture

```
LONG LECTURE VIDEO
        ↓
Video Stream Validation (FFmpeg & ffprobe)
        ↓
Audio Extraction & Preprocessing (16kHz Mono WAV + VAD)
        ↓
Automatic Speech Recognition (Whisper / faster-whisper ASR)
        ↓
Timestamped Transcript & Punctuation Restoration
        ↓
Lecture Structure Classification (Definitions, Concepts, Formulas)
        ↓
Dense Vector Embedding Generation (SentenceTransformers / TF-IDF)
        ↓
Multi-Factor Importance Scoring & Progressive Redundancy Penalization
        ↓
Topic Segmentation & Topic Coverage Optimization
        ↓
Extractive / Abstractive / Hybrid Text Summarization
        ↓
Summary-to-Transcript Semantic Alignment & Timestamp Retrieval
        ↓
Context-Aware Boundary Expansion & Deduplication
        ↓
0/1 Knapsack Target Duration Video Segment Selection
        ↓
FFmpeg Segment Cutting & Video Composition with Audio Sync
        ↓
Synchronized SRT/VTT Subtitle & Keyword Overlay Generation
        ↓
Multilingual Translation (Kannada, Hindi, Tamil, Telugu, etc.)
        ↓
Final Short Summarized Lecture Video & Interactive Academic Dashboard
```

---

## 🚀 Features

- **Domain Specificity**: Engineered specifically for educational lectures (college lectures, classroom tutorials, academic presentations).
- **Central Timestamped Transcript**: Preserves immutable links between spoken words, sentence boundaries, embeddings, importance scores, summary statements, and video clip timestamps.
- **Multi-Factor Importance Scoring**: Evaluates semantic relevance, educational category value (definitions, formulas, concepts), academic terminology density, and position.
- **Summary-Transcript Semantic Alignment**: Cosine similarity matching between generated abstractive summary statements and source transcript sentences to retrieve exact timestamps.
- **0/1 Knapsack Segment Selection**: Maximizes information density subject to `total_duration <= target_duration` constraint.
- **Real-Time Progress Streaming**: Server-Sent Events (SSE) `/api/stream/<job_id>` pushing live processing stages and percentages.
- **Interactive Academic Dashboard**: Clickable timestamped transcript and topic timeline that seeks the video player directly to target timestamps.
- **Academic Evaluation Engine**: Quantitative metrics (WER, CER, ROUGE-1/2/L, BERTScore, Duration Compliance) and ablation test framework comparing Extractive vs Abstractive vs Hybrid methods.

---

## 🛠️ Project Structure

```
LSUM/
├── app.py                      # Flask Server & Route Gateway
├── pipeline_engine.py          # Master 25+ Stage Pipeline Engine
├── config.py                   # Central Configuration System
├── Dockerfile                  # Container Deployment Setup
├── requirements.txt            # Package Dependencies
│
├── models/                     # Data Models & Schemas
│   ├── transcript.py           # TranscriptSegment & Sentence dataclasses
│   ├── summary.py              # LectureSummary & SummaryStatement models
│   ├── segment.py              # VideoClip dataclass
│   ├── job.py                  # ProcessingJob & Stage state models
│   └── lecture.py              # Lecture metadata models
│
├── services/                   # Modular Pipeline Services
│   ├── video_processor.py      # Stream validation & metadata probing
│   ├── audio_processor.py      # Audio extraction & VAD preprocessing
│   ├── transcription_service.py # ASR service (Whisper singleton)
│   ├── transcript_processor.py  # Punctuation & sentence timestamping
│   ├── lecture_preprocessor.py # Educational structure classifier
│   ├── embedding_service.py    # Dense vector embeddings
│   ├── importance_service.py   # Multi-factor importance scoring
│   ├── redundancy_service.py   # MMR redundancy deduplication
│   ├── topic_service.py        # Semantic topic segmentation
│   ├── summarization_service.py# Extractive, Abstractive & Hybrid summarizers
│   ├── alignment_service.py    # Semantic summary-transcript alignment
│   ├── segment_service.py      # Boundary expansion & 0/1 Knapsack selection
│   ├── video_composer.py       # FFmpeg video composition & audio sync
│   ├── subtitle_service.py     # SRT subtitle & keyword generation
│   └── translation_service.py  # Extensible multilingual translator
│
├── routes/                     # REST & Streaming Endpoints
│   ├── upload.py               # Video file & URL ingestion APIs
│   ├── processing.py           # Summarization task triggers
│   ├── results.py              # JSON data & file download endpoints
│   └── streaming.py            # Real-time SSE progress streaming endpoint
│
├── utils/                      # Low-Level Utilities
│   ├── ffmpeg_utils.py         # FFmpeg wrappers
│   ├── timestamp_utils.py      # Timestamp parsing & merging
│   └── similarity_utils.py     # Cosine similarity matrix operations
│
├── evaluation/                 # Academic Evaluation Suite
│   ├── metrics.py              # WER, CER, ROUGE-1/2/L, Duration metrics
│   └── experiment_runner.py    # Automated ablation experiment runner
│
├── templates/
│   ├── index.html              # Academic upload & configuration interface
│   └── result.html             # Interactive result dashboard
│
└── uploads/, outputs/, cache/  # Working directories
```

---

## 💻 Installation & Usage

### 1. Requirements

- Python 3.10+
- FFmpeg installed and added to system PATH

### 2. Local Setup

```bash
git clone <repository-url>
cd LSUM
pip install -r requirements.txt
python app.py
```

Open web browser at `http://127.0.0.1:5000`.

### 3. Running Academic Evaluation & Ablation Experiments

```bash
python evaluation/experiment_runner.py <path_to_lecture_video.mp4>
```

### 4. Docker Deployment

```bash
docker build -t lecture-summarizer .
docker run -p 5000:5000 lecture-summarizer
```
