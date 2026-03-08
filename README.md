# InsightForge AI

InsightForge AI is a Streamlit workspace for exploring research papers and long-form documents.
It supports PDF and text ingestion, OCR fallback for scanned PDFs, chunk-aware analysis for large
papers, document chat with memory, summaries, key intel extraction, research briefs, action plans,
visual knowledge graphs, writing feedback, and exportable reports.

## Features

- Upload PDF or text files, paste text, or load from a local path
- OCR fallback for scanned or image-only PDFs using `tesseract`
- Chunk-aware retrieval and synthesis for very large documents
- Multi-turn chat that keeps per-document memory
- Executive summary, key intel, research brief, action plan, and feedback modes
- Knowledge graph extraction with a visual network view plus node and edge tables
- Markdown, DOCX, and PDF report exports
- Automated tests under `tests/`

## Run

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Configure the OpenAI API key on the backend.

   Streamlit UI no longer asks for the key. Use either Streamlit secrets (an example is in `.streamlit/secrets.toml.example`):

   ```toml
   # .streamlit/secrets.toml
   OPENAI_API_KEY = "your_api_key"
   ```

   Or an environment variable:

   ```bash
   export OPENAI_API_KEY="your_api_key"
   ```

3. Start the app:

   ```bash
   streamlit run streamlit_app.py
   ```

## Notes

- The API key is read only from backend configuration (`.streamlit/secrets.toml` or environment variables).
- PDFs with extractable text are supported.
- Scanned image-only PDFs use OCR when `tesseract` is available.
- A CLI fallback still exists in `main.py`, but the Streamlit app is the primary interface.
