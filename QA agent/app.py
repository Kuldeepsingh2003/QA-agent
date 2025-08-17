import streamlit as st
import pandas as pd
from src.data_loader import load_inputs, excel_to_text_chunks, preview_dataframes
from src.qa_engine import LocalRetrievalQASystem
from src.utils import set_seed

st.set_page_config(page_title="AI Agent: QA on Shared Excel Docs", layout="wide")

st.title("AI Agent for QA on Shared Documents (Local Model)")

with st.sidebar:
    st.header("Data Sources")
    st.markdown("Upload Excel files (xlsx) or use the provided samples in the data folder.")
    uploaded = st.file_uploader("Upload one or more .xlsx files", type=["xlsx"], accept_multiple_files=True)
    top_k = st.slider("Retriever Top-K", 1, 20, 5)
    max_ctx_chars = st.slider("Max Context Length (chars)", 500, 8000, 2500, step=250)
    model_name_embed = st.text_input("Embedding model (Sentence-Transformers)", "sentence-transformers/all-MiniLM-L6-v2")
    model_name_qa = st.text_input("QA model (Transformers)", "distilbert-base-cased-distilled-squad")
    seed = st.number_input("Random seed", value=42, step=1)
    rebuild = st.checkbox("Rebuild index", value=False)
    st.caption("Models are downloaded locally on first run.")

set_seed(seed)

# Load files: uploaded ones take precedence; otherwise load from data/ directory
dfs, sources = load_inputs(uploaded_files=uploaded, fallback_dir="data")

if not dfs:
    st.warning("No Excel files loaded yet. Upload files from the sidebar or place them in the data/ folder.")
    st.stop()

# Show previews
with st.expander("Preview Loaded DataFrames", expanded=False):
    preview_dataframes(dfs)

# Build text chunks for retrieval
chunks = []
for (name, df) in dfs:
    chunks.extend(excel_to_text_chunks(df, table_name=name))

st.success(f"Prepared {len(chunks)} text chunks from {len(dfs)} Excel sheet(s).")

# Initialize / cache QA system
@st.cache_resource(show_spinner=True)
def get_system(embed_model: str, qa_model: str):
    return LocalRetrievalQASystem(embedding_model_name=embed_model, qa_model_name=qa_model)

qa_system = get_system(model_name_embed, model_name_qa)
if rebuild or not qa_system.is_indexed:
    with st.spinner("Building index over chunks..."):
        qa_system.build_index(chunks)

# Q&A UI
st.subheader("Ask questions about the loaded documents")
question = st.text_input("Question", placeholder="e.g., What is the total sales in 2023?")
btn = st.button("Get Answer", type="primary")

if btn and question.strip():
    with st.spinner("Thinking locally..."):
        result = qa_system.answer(question, top_k=top_k, max_context_chars=max_ctx_chars)

    st.markdown("### Answer")

    if isinstance(result["answer"], dict):
        answer_dict = result["answer"]

        # Extract relevant columns based on question keywords
        question_words = question.lower().split()
        filtered_dict = {
            k: v for k, v in answer_dict.items()
            if any(word in k.lower() for word in question_words)
        }

        # If nothing matched, show all
        if not filtered_dict:
            filtered_dict = answer_dict

        # Display as DataFrame
        answer_df = pd.DataFrame([filtered_dict])
        for col in answer_df.select_dtypes(include="number").columns:
            answer_df[col] = answer_df[col].apply(lambda x: f"{x:,}")
        st.dataframe(answer_df, use_container_width=True)
    else:
        st.write(result["answer"])

    st.markdown(f"**Confidence:** {result['score']:.2f}")

    with st.expander("Retrieved Context", expanded=False):
        st.write(result["context"])

    with st.expander("Matched Chunks", expanded=False):
        for i, item in enumerate(result["matches"], 1):
            st.write(f"{i}. score={item['score']:.4f}")
            st.code(item["text"][:1000])


st.markdown("---")
st.caption("Tip: If answers look off, increase Top-K and Context Length, or try a different QA model.")