# File Search POC 🔍

A minimal RAG chatbot using Google's Gemini File Search API.

## Features

- **Spaces**: Create isolated knowledge bases (File Search Stores)
- **File Upload**: Upload documents (PDF, TXT, MD, JSON, CSV, DOCX, HTML)
- **Chat**: Ask questions and get answers grounded in your documents
- **Citations**: See which files were used to generate responses

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your API key

```bash
export GEMINI_API_KEY=your-api-key-here
```

Get your API key from [Google AI Studio](https://aistudio.google.com/apikey).

### 3. Run locally

```bash
streamlit run app.py
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `GEMINI_API_KEY` | (required) | Your Google AI API key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Model to use for chat |

## Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and select `app.py`
4. Add `GEMINI_API_KEY` in Secrets:
   ```toml
   GEMINI_API_KEY = "your-api-key-here"
   ```
5. Deploy!

## How It Works

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Upload    │────▶│  File Search     │────▶│  Chunked &      │
│   Files     │     │  Store (Space)   │     │  Embedded       │
└─────────────┘     └──────────────────┘     └─────────────────┘
                            │
                            ▼
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Chat      │────▶│  Semantic Search │────▶│  Grounded       │
│   Query     │     │  + Gemini        │     │  Response       │
└─────────────┘     └──────────────────┘     └─────────────────┘
```

Each **Space** = A Google File Search Store (persistent)  
Each **File** = Automatically chunked, embedded, and indexed  
Each **Chat** = Queries the space semantically, returns citations

