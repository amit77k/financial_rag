# 1.Install Required Packages

#pip install streamlit sentence-transformers faiss-cpu rank-bm25 transformers pypdf torch spacy
#python -m spacy download en_core_web_sm


# 2. Load and Process Financial Report

#from PyPDF2 import PdfReader
#import re

def extract_text_from_pdf(pdf_path):
    """Extracts text from a financial PDF and cleans it."""
    reader = PdfReader(pdf_path)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    text = re.sub(r'\n+', '\n', text).strip()
    return text

# Load BMW Finance Annual Report
pdf_text = extract_text_from_pdf("/content/BMW_Finance_NV_Annual_Report_2023.pdf")

# 3. Chunking & Embedding Storage

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from rank_bm25 import BM25Okapi

# Load Embedding Model (Best Open-Source Financial Embedding)
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Chunking the Text
def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size//2)]

text_chunks = chunk_text(pdf_text)

# Compute Embeddings
chunk_embeddings = np.array([embedding_model.encode(chunk) for chunk in text_chunks])

# Store in FAISS Vector Database
dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(chunk_embeddings)

# Save FAISS Index
faiss.write_index(index, "financial_rag.index")

# Initialize BM25 for Keyword Search
tokenized_chunks = [chunk.split() for chunk in text_chunks]
bm25 = BM25Okapi(tokenized_chunks)

# 5. Multi-Stage Retrieval (BM25 + FAISS Hybrid)

def multistage_retrieve_top_chunks(query, k=5, bm25_k=10, alpha=0.5):
    """
    Multi-Stage Retrieval:
    1. BM25 First-Stage: Get top `bm25_k` candidates based on keyword match.
    2. FAISS Second-Stage: Run FAISS search only on these candidates.
    3. Re-Ranking: Combine BM25 and FAISS scores for final ranking.
    """
    query_embedding = embedding_model.encode([query])

    # 🔹 Stage 1: BM25 Keyword Search
    bm25_scores = bm25.get_scores(query.split())
    top_bm25_indices = np.argsort(bm25_scores)[-bm25_k:]  # Select top `bm25_k` BM25 results

    # 🔹 Stage 2: FAISS Vector Search (only on BM25-filtered chunks)
    filtered_embeddings = np.array([chunk_embeddings[i] for i in top_bm25_indices])
    faiss_index = faiss.IndexFlatL2(filtered_embeddings.shape[1])
    faiss_index.add(filtered_embeddings)

    _, faiss_ranks = faiss_index.search(query_embedding, k)
    top_faiss_indices = [top_bm25_indices[i] for i in faiss_ranks[0]]  # Map back to original indices

    # 🔹 Stage 3: Re-Ranking (Combining BM25 & FAISS Scores)
    final_scores = {}
    for i in set(top_bm25_indices) | set(top_faiss_indices):
        bm25_score = bm25_scores[i] if i in top_bm25_indices else 0
        faiss_score = -np.linalg.norm(query_embedding - chunk_embeddings[i])  # L2 distance
        final_scores[i] = alpha * bm25_score + (1 - alpha) * faiss_score  # Weighted ranking

    # Get final top K chunks
    top_chunks = sorted(final_scores, key=final_scores.get, reverse=True)[:k]
    
    return [text_chunks[i] for i in top_chunks]

#5 . FinGPT Response Generation

from transformers import pipeline

# Load FinGPT or Equivalent SLM
generator = pipeline("text-generation", model="facebook/opt-1.3b")  # Replace with "FinGPT" when available

def generate_answer(query):
    """Retrieve relevant context and generate financial answers."""
    retrieved_context = "\n".join(hybrid_retrieve_top_chunks(query))
    prompt = f"Context: {retrieved_context}\n\nQuestion: {query}\n\nAnswer:"
    response = generator(prompt, max_length=100, do_sample=True)[0]["generated_text"]
    return response

# 6. Hallucination Detection

import spacy

nlp = spacy.load("en_core_web_sm")

def extract_numbers_and_entities(text):
    """Extracts numerical values and financial terms using NER."""
    doc = nlp(text)
    numbers = [ent.text for ent in doc.ents if ent.label_ in ["MONEY", "CARDINAL", "PERCENT", "QUANTITY"]]
    return set(numbers)

def detect_hallucinations(query, generated_answer, retrieved_context):
    """Detects hallucinations by checking generated numbers/entities against retrieved text."""
    extracted_from_answer = extract_numbers_and_entities(generated_answer)
    extracted_from_context = extract_numbers_and_entities(retrieved_context)
    
    hallucinated = extracted_from_answer - extracted_from_context
    if hallucinated:
        return f"⚠️ Hallucination Alert! Unverified Data: {', '.join(hallucinated)}"
    return generated_answer


# 7. Streamlit UI


import streamlit as st

st.title("💰 Financial RAG: BMW Finance N.V. Q&A")

query = st.text_input("Enter your financial question:")
if query:
    retrieved_context = "\n".join(hybrid_retrieve_top_chunks(query))
    generated_answer = generate_answer(query)
    verified_answer = detect_hallucinations(query, generated_answer, retrieved_context)

    st.markdown(f"### **Answer:** {verified_answer}")
    st.markdown(f"**Confidence Score:** {len(retrieved_context.split()) / 500:.2f} (Higher is better)")

    with st.expander("📄 Supporting Sources"):
        st.write(retrieved_context)

# 8. Testing & Validation

# Define test queries
test_queries = [
    ("What was BMW Finance N.V.'s net income for 2023?", "High confidence"),
    ("What is the expected financial outlook for 2024?", "Low confidence"),
    ("What is the capital of France?", "Irrelevant")
]

# Run tests
for query, expected_confidence in test_queries:
    print(f"\n🔹 **Query:** {query} \n(Expected Confidence: {expected_confidence})")
    retrieved_context = "\n".join(hybrid_retrieve_top_chunks(query))
    generated_answer = generate_answer(query)
    verified_answer = detect_hallucinations(query, generated_answer, retrieved_context)
    print(f"✅ **Answer:** {verified_answer}")
    print(f"📊 **Confidence Score:** {len(retrieved_context.split()) / 500:.2f}\n")
