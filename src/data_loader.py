from typing import List, Tuple
import os
import io
import pandas as pd
import streamlit as st

def load_inputs(uploaded_files=None, fallback_dir: str = "data") -> Tuple[list, list]:
    """
    Returns:
        dfs: List[Tuple[str, pd.DataFrame]]
        sources: List[str]
    """
    dfs: List[Tuple[str, pd.DataFrame]] = []
    sources: List[str] = []

    if uploaded_files:
        for f in uploaded_files:
            file_bytes = f.read()
            try:
                xls = pd.ExcelFile(io.BytesIO(file_bytes))
            except Exception as e:
                st.error(f"Failed to read {getattr(f, 'name', 'uploaded file')}: {e}")
                continue
            for sheet in xls.sheet_names:
                df = xls.parse(sheet_name=sheet)
                name = f"{getattr(f, 'name', 'uploaded')}: {sheet}"
                dfs.append((name, df))
                sources.append(name)

    # If nothing uploaded, load every .xlsx in fallback_dir
    if not dfs and os.path.isdir(fallback_dir):
        for fname in os.listdir(fallback_dir):
            if fname.lower().endswith(".xlsx"):
                path = os.path.join(fallback_dir, fname)
                try:
                    xls = pd.ExcelFile(path)
                    for sheet in xls.sheet_names:
                        df = xls.parse(sheet_name=sheet)
                        name = f"{fname}: {sheet}"
                        dfs.append((name, df))
                        sources.append(name)
                except Exception as e:
                    st.error(f"Failed to read {fname}: {e}")

    return dfs, sources

def row_to_text(row, columns, table_name: str) -> str:
    pairs = [f"{col}: {row[col]}" for col in columns]
    return f"[{table_name}] " + "; ".join(pairs)

def excel_to_text_chunks(df: pd.DataFrame, table_name: str, max_chars: int = 500) -> List[str]:
    """
    Convert each row into a compact text snippet.
    We also chunk long tables by grouping rows until ~max_chars.
    """
    df = df.fillna("")
    cols = [str(c) for c in df.columns]
    chunks: List[str] = []
    buf = ""
    for _, row in df.iterrows():
        line = row_to_text(row, cols, table_name)
        # add newline for readability
        candidate = (buf + "\n" + line).strip()
        if len(candidate) > max_chars and buf:
            chunks.append(buf.strip())
            buf = line
        else:
            buf = candidate
    if buf:
        chunks.append(buf.strip())
    return chunks

def preview_dataframes(dfs: List[Tuple[str, pd.DataFrame]]):
    for (name, df) in dfs:
        st.markdown(f"**{name}**  —  shape: {df.shape[0]} rows × {df.shape[1]} cols")
        st.dataframe(df.head(10))
