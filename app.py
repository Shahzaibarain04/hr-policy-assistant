import fitz  # PyMuPDF
import faiss
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="HR Policy Assistant", page_icon="📋")
st.title("📋 HR Policy Assistant")

# Initialize models and client
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# Setup Groq Client
groq_api_key = st.sidebar.text_input("Groq API Key", type="password")

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

uploaded_file = st.sidebar.file_uploader("Upload HR Policy PDF", type=["pdf"])

if uploaded_file and groq_api_key:
    client = Groq(api_key=groq_api_key)
    
    with st.spinner("Processing PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(pdf_text)
        
        # Build Vector Store
        embeddings = embedder.encode(chunks, show_progress_bar=False)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype("float32"))
    
    st.success("PDF processed successfully! Ask your questions below.")

    query = st.text_input("Ask a question about the HR Policy:")

    if query:
        query_vector = embedder.encode([query]).astype("float32")
        k = 3  # Top 3 relevant chunks
        distances, indices = index.search(query_vector, k)
        
        retrieved_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
        context = "\n\n".join(retrieved_chunks)

        prompt = f"""You are an HR Policy Assistant. Answer the question based ONLY on the provided context.
        If the answer cannot be found in the context, say "I cannot find this information in the uploaded policy."

        Context:
        {context}

        Question: {query}
        """

        with st.spinner("Generating answer..."):
            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[{"role": "user", "content": prompt}]
            )
            st.markdown("### Answer")
            st.write(response.choices[0].message.content)

elif not groq_api_key:
    st.info("Please enter your Groq API Key in the sidebar to proceed.")
else:
    st.info("Please upload an HR Policy PDF to get started.")
