# DocuSense AI

Evidence-grounded document intelligence for PDFs. Upload a text-based PDF, ask questions in plain English, and check every answer against the page-numbered passages it was based on. DocuSense AI also writes a structured summary of the document, which you can download as a Markdown file.

The app is built with Python, Streamlit, PyMuPDF, Sentence-Transformers, NumPy and the Google Gemini API. It uses retrieval-augmented generation (RAG): the language model receives only the passages that match a question, not the whole document.

Live demo: add your Streamlit Community Cloud link here.

## Features

Uploads are checked before anything is processed. The app validates the file type, size, PDF header, page count and password protection, and reports damaged files and PDFs with no selectable text in plain language. Each page is extracted separately, so every passage keeps the page number it came from.

Questions are matched to passages by meaning using Sentence-Transformers embeddings and cosine similarity. The best passages are sent to Gemini with instructions to answer only from them. Each answer cites passage numbers, and the Sources panel shows the page label and the full text of every cited passage.

If no passage reaches the relevance threshold, the model is not called and the app reports that the document does not contain enough evidence. If the model itself finds that the passages do not answer the question, the app shows the same message. These checks reduce unsupported answers, but they do not eliminate them, which is why the sources are always visible.

The summary covers the main purpose, key points, important dates and requirements, and a conclusion. When pages have no selectable text, or the document is too long to send in full, the summary says which part is missing. The summary can be downloaded as Markdown.

The interface uses a navy, white and light-grey theme. It shows loading indicators while the document is processed, empty states before an upload, the search method in use, and friendly messages for quota limits, unavailable models and invalid keys. A reset button clears the document, chat and summary.

## Architecture

```
PDF upload
  -> validation and page-by-page text extraction (PyMuPDF)
  -> cleaning and page-bounded chunks (120 words, 25-word overlap)
  -> embeddings (Sentence-Transformers, all-MiniLM-L6-v2)
  -> cosine-similarity search over the document's chunks (NumPy)
  -> evidence threshold; below it, the model is not called
  -> grounded prompt with numbered passages (Gemini API)
  -> citation numbers mapped to stored page labels
  -> Streamlit interface with answer and sources
```

Chunks never cross a page boundary, so each citation refers to exactly one page. The trade-off is that a sentence split by a page break appears in two chunks. The keyword fallback described below uses the same pipeline with TF-IDF vectors instead of neural embeddings.

## Project structure

```
DocuSense-AI/
  app.py                  Streamlit interface
  config.py               settings, limits and model names
  pdf_processor.py        upload checks and PDF text extraction
  chunker.py              page-bounded, overlapping chunks and summary text
  retriever.py            embeddings, keyword fallback and similarity search
  llm_service.py          Gemini prompts, citation parsing and error messages
  rag_pipeline.py         question answering and summary flow, without UI code
  scripts/list_models.py  lists the Gemini models available to your key
  sample_data/            sample PDF for manual testing
  tests/                  unit tests
  .streamlit/config.toml  theme settings
```

## Setup on Windows

Install Python 3.11 or 3.12 and Git, then run the following commands in a terminal opened in the project folder. PowerShell is shown first; Command Prompt uses the activate script noted in the comment.

```
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
copy .env.example .env
```

In Command Prompt, activate the environment with `.venv\Scripts\activate.bat` instead. Open `.env` in your editor, replace the placeholder with your Gemini API key, and save the file. Then start the app:

```
streamlit run app.py
```

The app opens at http://localhost:8501. The separate study guide, DocuSense-AI-Study-Guide.pdf, gives step-by-step instructions, including how to get an API key and how to fix common errors.

## Configuration

Every setting can be provided as an environment variable, in the `.env` file for local use, or in Streamlit Secrets when the app is deployed.

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `GEMINI_API_KEY` | For answers and summaries | none | Google AI Studio API key |
| `GEMINI_MODEL` | No | `gemini-2.5-flash` | Gemini model name. Check the names with `python scripts/list_models.py` |
| `EMBEDDING_BACKEND` | No | `semantic` | `semantic` uses Sentence-Transformers. `keyword` skips the model |
| `EMBEDDING_MODEL` | No | `sentence-transformers/all-MiniLM-L6-v2` | Sentence-Transformers model name |

The evidence thresholds (0.30 for semantic search and 0.05 for keyword matching), the chunk size, the upload limits and the summary length are constants in `config.py`. They are starting values, not measured optima, so tune them on your own documents.

## Testing

```
python -m unittest discover -s tests -v
```

The tests cover PDF validation and extraction, page-number metadata, chunking, keyword retrieval, the fallback when the semantic model is unavailable, Gemini error handling with a fake client, citation parsing, defences against text that tries to close the prompt, and the question-answering and summary flow. The semantic-search test runs only when sentence-transformers is installed. A passing run ends with OK.

## Deployment

The app is designed for Streamlit Community Cloud. Push the repository to GitHub, create an app from it with `app.py` as the main file, and add `GEMINI_API_KEY` under the app's Secrets. The study guide describes each step, including how to test the public link from a fresh browser session.

## Privacy and security

API keys are read from environment variables or Streamlit Secrets. They are never written into the code, and `.env` is excluded by `.gitignore`.

When you ask a question or generate a summary, the relevant passages of your document are sent to Google's Gemini API. Do not upload confidential documents unless you understand the provider's data terms, and check the terms that apply to your account tier.

Document text is treated as untrusted input. The prompts tell the model to ignore instructions inside passages, tags in the document text are neutralised so they cannot close the prompt block, and model output has images and HTML removed before it is displayed. Uploads are limited to 10 MB and 200 pages, and questions to 500 characters.

The app keeps uploaded documents in memory for the browser session and does not write them to disk. It has no login, so the public link lets anyone upload documents to that running instance.

## Limitations

The app reads text-based PDFs only. Scanned pages are reported and skipped because OCR is not included. Tables, charts and images are not interpreted. Sentences split by a page break are split across two chunks, which can weaken retrieval for them.

Citations show where the evidence came from, but an answer can still be incomplete, over-generalised or wrong, so the cited passage should always be read. The defences against instructions hidden in a document reduce the risk but do not remove it. Free-tier Gemini quotas limit how many questions can be answered per day. The project has not been load tested and is not hardened for production or confidential data.

Retrieval accuracy has not been measured on any document set yet.

## Future improvements

Possible next steps include OCR for scanned pages, table-aware extraction, a labelled evaluation set that reports how often the correct page is retrieved, reranking of retrieved passages, a persistent vector store such as FAISS or Chroma for document collections, streamed answers, user accounts with per-user storage, and automatic calibration of the evidence thresholds.

## Screenshots

Run the app locally, then save screenshots of the upload status, an answer with its sources and the summary into `docs/screenshots/`. Link them here, for example `![Answer with sources](docs/screenshots/answer.png)`.

## Licence

No licence has been chosen yet. Add a LICENSE file before others reuse the code.
