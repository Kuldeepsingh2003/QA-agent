# AI Agent for QA on Shared Documents (Excel) — Local Models, Streamlit UI

This project builds a **local** retrieval-augmented **question answering (QA)** system over **Excel documents** with a **Streamlit** interface.  
It converts spreadsheet rows into text chunks, retrieves the most relevant chunks using **sentence-transformers** embeddings, and extracts answers using a local **extractive QA** model.

## Features
- Upload `.xlsx` files or auto-load files from `data/`.
- Local embedding model (`sentence-transformers/all-MiniLM-L6-v2` by default).
- Local extractive QA model (`deepset/roberta-base-squad2` by default).
- Adjustable Top-K retrieval and max context length.
- Transparent context and matched chunks for verification.

## Quickstart

1. **Install dependencies** (ideally in a fresh virtual environment):
    ```bash
    pip install -r requirements.txt
    ```

2. **Run the app**:
    ```bash
    streamlit run app.py
    ```

3. **Use it**:
   - From the sidebar, either upload one or more `.xlsx` files or rely on the samples in `data/`.
   - Enter your question and click **Get Answer**.
   - Inspect the retrieved context and matched chunks under the expanders.

> The first run will download the models to your local cache. This requires internet once. Subsequent runs are fully local.
## Project Structure
qa_agent/
├── app.py
├── requirements.txt
├── README.md
├── data/
│ ├── data.xlsx
│ └── Forcast.xlsx
└── src/
├── init.py
├── data_loader.py
├── qa_engine.py
└── utils.py
## How it Works (Architecture)
. **Ingestion**: Each Excel sheet is parsed to a pandas DataFrame. Each row is converted into a compact text line like:
Rows are grouped into chunks ~500 characters long (configurable in code).

2. **Indexing**: We embed all chunks with `SentenceTransformer("all-MiniLM-L6-v2")`, L2-normalize them, and build a cosine-similarity index with `NearestNeighbors`.

3. **Retrieval**: For a question, we embed the query and fetch Top-K similar chunks.

4. **Answer Extraction**: Retrieved chunks are concatenated into a bounded **context** (controlled by **Max Context Length**). A local extractive QA model (`deepset/roberta-base-squad2`) selects an **answer span** from that context.

5. **Transparency**: The UI shows confidence score, context, and matched chunks to help you verify outputs.

## Tips for Better Answers
- Increase **Top-K** and **Max Context Length** for multi-row questions.
- Try alternative QA models:
- `distilbert-base-cased-distilled-squad` (lighter)
- `deepset/xlm-roberta-large-squad2` (bigger, multilingual)
- Clean column names and ensure important values live in plain text (avoid heavily formatted cells).

## Extending the Agent
- Add CSV, PDF, or DOCX ingestion with libraries like `pdfplumber` / `python-docx`.
- Use FAISS for larger corpora.
- Add summarization for questions that aren’t span-answerable (e.g., “summarize Q2 performance”).

## Troubleshooting
- **CUDA/CPU**: By default, models run on CPU. If you have PyTorch with CUDA, they’ll use GPU automatically.
- **Memory**: For very large spreadsheets, lower Top-K or chunk size.
- **Model Download**: First run downloads models. If you’re fully offline, pre-download and place them in your HuggingFace cache.

## License
MIT
