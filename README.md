# Deal Room Assistant ◆

A document intelligence tool for Private Equity due diligence, powered by Google's Gemini File Search API.

## Features

- **Deal Rooms**: Isolated workspaces for each potential acquisition or deal
- **Document Upload**: Upload CIMs, financials, legal docs, memos (PDF, DOCX, XLSX, etc.)
- **Intelligent Q&A**: Ask questions and get answers grounded in your deal materials
- **Source Citations**: See which documents were referenced in each response

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
| `GEMINI_MODEL` | `gemini-2.5-flash` | Model to use for analysis |

## Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and select `app.py`
4. Add `GEMINI_API_KEY` in Secrets:
   ```toml
   GEMINI_API_KEY = "your-api-key-here"
   ```
5. Deploy and share with your team

## How It Works

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Upload Deal    │────▶│   Deal Room     │────▶│   Chunked &     │
│  Materials      │     │   (File Store)  │     │   Embedded      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │
                               ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Due Diligence  │────▶│ Semantic Search │────▶│   Grounded      │
│  Questions      │     │   + Gemini      │     │   Analysis      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

**Deal Room** = A Google File Search Store (persistent document collection)  
**Documents** = Automatically chunked, embedded, and indexed for semantic search  
**Q&A** = Questions are matched semantically to relevant document sections

## Use Cases

- **Underwriting**: Quickly extract key metrics and assumptions from CIMs
- **Due Diligence**: Surface risks and red flags across large document sets
- **Deal Comparison**: Maintain separate rooms for each target company
- **Team Collaboration**: Share deal rooms with analysts for parallel review
