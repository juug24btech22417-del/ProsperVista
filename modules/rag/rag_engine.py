import streamlit as st
import numpy as np
from pypdf import PdfReader
import google.generativeai as genai
import os
import re
from concurrent.futures import ThreadPoolExecutor

def split_text_into_chunks(text, page_num, source_name, chunk_size=800, overlap=150):
    """
    Splits text into overlapping chunks, keeping track of page number and source file.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        chunks.append({
            "text": chunk_text,
            "page": page_num,
            "source": source_name
        })
        start += (chunk_size - overlap)
    return chunks

def process_document(file_obj, filename):
    """
    Extracts text from PDF/TXT and breaks it down into page-aware chunks.
    """
    chunks = []
    try:
        if filename.endswith(".pdf"):
            reader = PdfReader(file_obj)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    chunks.extend(split_text_into_chunks(text, i + 1, filename))
        else:
            # Assume plain text
            text = file_obj.read().decode("utf-8")
            chunks.extend(split_text_into_chunks(text, 1, filename))
    except Exception as e:
        st.error(f"Error reading file {filename}: {e}")
    return chunks

def get_gemini_embeddings(texts, api_key):
    """
    Generates embeddings for a list of texts using Gemini API.
    """
    genai.configure(api_key=api_key)
    # Using thread pool to embed chunks in parallel
    def _get_single_embedding(text):
        try:
            res = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            return res.get('embedding', [])
        except:
            return []

    with ThreadPoolExecutor(max_workers=8) as executor:
        embeddings = list(executor.map(_get_single_embedding, texts))
        
    # Clean up empty embeddings
    valid_embeddings = []
    for emb in embeddings:
        if emb:
            valid_embeddings.append(emb)
        else:
            # Fallback vector of zeros if failed
            valid_embeddings.append([0.0] * 768)
            
    return np.array(valid_embeddings)

def search_chunks_tfidf(query, chunks):
    """
    Fallback pure-Python keyword overlap search when Gemini API key is unavailable.
    """
    query_words = set(re.findall(r'\w+', query.lower()))
    scores = []
    
    for i, c in enumerate(chunks):
        chunk_words = re.findall(r'\w+', c['text'].lower())
        overlap = len(query_words.intersection(chunk_words))
        # Length normalization to avoid favoring long chunks
        score = overlap / (np.log(len(chunk_words) + 1) + 1e-5)
        scores.append((score, i))
        
    scores.sort(reverse=True, key=lambda x: x[0])
    return [chunks[idx] for score, idx in scores[:4]]

def search_chunks_embeddings(query, chunks, embeddings, api_key):
    """
    Performs cosine similarity search using Gemini query embeddings.
    """
    try:
        genai.configure(api_key=api_key)
        res = genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="retrieval_query"
        )
        query_emb = np.array(res.get('embedding', []))
        
        if len(query_emb) == 0 or len(embeddings) == 0:
            return search_chunks_tfidf(query, chunks)
            
        # Compute cosine similarity
        dots = np.dot(embeddings, query_emb)
        norms_emb = np.linalg.norm(embeddings, axis=1)
        norm_query = np.linalg.norm(query_emb)
        similarities = dots / (norms_emb * norm_query + 1e-8)
        
        top_indices = np.argsort(similarities)[::-1][:4]
        return [chunks[idx] for idx in top_indices]
    except Exception as e:
        return search_chunks_tfidf(query, chunks)

def perform_rag_query(query, chunks, embeddings, api_key=None):
    """
    Retrieves sources and generates a synthesized, source-cited answer.
    """
    if api_key:
        matched_chunks = search_chunks_embeddings(query, chunks, embeddings, api_key)
    else:
        matched_chunks = search_chunks_tfidf(query, chunks)
        
    context_blocks = []
    for c in matched_chunks:
        context_blocks.append(f"Source File: {c['source']}, Page: {c['page']}\nContent: {c['text']}")
        
    context_str = "\n---\n".join(context_blocks)
    
    if api_key:
        try:
            genai.configure(api_key=api_key)
            from modules.copilot.copilot import get_best_gemini_model
            model = genai.GenerativeModel(get_best_gemini_model())
            prompt = (
                f"You are a corporate intelligence agent. Based on the document excerpts provided below, "
                f"answer the question. You MUST cite the source file and page numbers inside your answer "
                f"when referencing facts (e.g. [AnnualReport.pdf, Page 12]). "
                f"If the answer cannot be found in the provided excerpts, state clearly that it is not in the text.\n\n"
                f"Document Excerpts:\n{context_str}\n\n"
                f"Question: {query}"
            )
            response = model.generate_content(prompt)
            return response.text, matched_chunks
        except Exception as e:
            return f"Error synthesizing answer via Gemini: {e}", matched_chunks
            
    # Fallback response template
    sources_text = "\n".join([f"- **{c['source']}** (Page {c['page']}): \"{c['text'][:150]}...\"" for c in matched_chunks])
    fallback_answer = (
        f"### Local Keyword Match (Gemini key not configured)\n"
        f"I found the following matching fragments in your uploaded documents. "
        f"Provide a Gemini API key to synthesize a natural answer with citations.\n\n"
        f"**Relevant Sources:**\n{sources_text}"
    )
    return fallback_answer, matched_chunks

def generate_document_summary(chunks, api_key):
    """
    Generates a concise executive description/summary of the uploaded document(s).
    """
    if not chunks:
        return "No content available to summarize."
        
    # Pick the first 5 chunks (usually contains the introduction, table of contents, executive summary)
    intro_chunks = chunks[:5]
    intro_text = "\n\n".join([f"Excerpt from page {c['page']}:\n{c['text']}" for c in intro_chunks])
    
    if api_key:
        try:
            genai.configure(api_key=api_key)
            from modules.copilot.copilot import get_best_gemini_model
            model = genai.GenerativeModel(get_best_gemini_model())
            prompt = (
                "You are an expert financial analyst. Analyze the following opening excerpts from an uploaded document. "
                "Provide a neat executive description of this document. "
                "Include: \n"
                "1. Document Identification: What is this document (e.g., Annual Report, Earnings Call Transcript, Prospectus), "
                "which company is it for, and what period or fiscal year does it cover?\n"
                "2. Executive Overview: A 3-4 sentence concise summary of the company's core focus, key themes, and financial highlights "
                "mentioned in these opening pages.\n"
                "Keep the format professional, with bold subheaders and bullet points where appropriate."
                f"\n\nDocument Opening Excerpts:\n{intro_text}"
            )
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating summary: {e}"
            
    # Fallback if no API key is present
    return "Document indexed successfully. (API key not configured to generate executive summary)"

def render_rag_interface():
    """
    Renders the beautiful RAG interface inside Streamlit.
    """
    st.markdown('<div class="dashboard-header">'
                '<div class="dashboard-title">RAG Company Research</div>'
                '<div class="dashboard-desc">Semantic Intelligence & Document QA</div>'
                '<div class="dashboard-long-desc">Upload annual reports, earnings transcripts, or SEC filings and perform source-cited semantic queries.</div>'
                '</div>', unsafe_allow_html=True)
                
    api_key = os.environ.get("GEMINI_API_KEY") or st.session_state.get("gemini_api_key")
    
    if "rag_chunks" not in st.session_state:
        st.session_state.rag_chunks = []
        st.session_state.rag_embeddings = None
        st.session_state.rag_filenames = []
        st.session_state.rag_summary = None
        
    # Sidebar config inside main viewport or sidebar
    st.markdown("### 📤 Step 1: Ingest Corporate Documents")
    uploaded_files = st.file_uploader("Upload Company Annual Reports or Earnings Transcripts (PDF or TXT):", 
                                     type=["pdf", "txt"], 
                                     accept_multiple_files=True)
                                     
    if uploaded_files:
        new_files = [f.name for f in uploaded_files]
        if new_files != st.session_state.rag_filenames:
            all_chunks = []
            with st.spinner("Parsing documents & generating chunks..."):
                for f in uploaded_files:
                    chunks = process_document(f, f.name)
                    all_chunks.extend(chunks)
            
            st.session_state.rag_chunks = all_chunks
            st.session_state.rag_filenames = new_files
            
            # Embed if key is available
            if api_key and all_chunks:
                with st.spinner("Generating semantic embeddings..."):
                    texts = [c["text"] for c in all_chunks]
                    st.session_state.rag_embeddings = get_gemini_embeddings(texts, api_key)
                st.success(f"Successfully indexed {len(all_chunks)} semantic chunks across {len(new_files)} files.")
                
                with st.spinner("Generating executive document overview..."):
                    st.session_state.rag_summary = generate_document_summary(all_chunks, api_key)
            elif all_chunks:
                st.info(f"Successfully indexed {len(all_chunks)} chunks across {len(new_files)} files. (Running in keyword-fallback mode due to missing API key)")
                st.session_state.rag_embeddings = None
                st.session_state.rag_summary = "Document indexed. Provide Gemini API key to generate executive summary."
    else:
        st.session_state.rag_chunks = []
        st.session_state.rag_embeddings = None
        st.session_state.rag_filenames = []
        st.session_state.rag_summary = None

    if st.session_state.get("rag_summary"):
        st.markdown("#### 📋 Executive Document Description")
        st.markdown(f'<div class="glass-card" style="padding:25px; line-height:1.6; margin-bottom:20px;">{st.session_state.rag_summary}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔍 Step 2: Query Corporate Intelligence")
    
    if not st.session_state.rag_chunks:
        st.info("Please upload and index a document first to start research.")
        return
        
    query = st.text_input("Enter your research question (e.g. What is the EBITDA target? List all major risks):", 
                         placeholder="e.g. What were the core capital expenditures last year?")
                         
    if st.button("Query Database", use_container_width=True):
        if query:
            with st.spinner("Searching document vector store and synthesizing response..."):
                answer, sources = perform_rag_query(query, 
                                                    st.session_state.rag_chunks, 
                                                    st.session_state.rag_embeddings, 
                                                    api_key)
                
                # Render synthesized answer
                st.markdown("#### 💬 Executive Summary")
                st.markdown(f'<div class="glass-card" style="padding:25px; line-height:1.6; margin-bottom:20px;">{answer}</div>', unsafe_allow_html=True)
                
                # Render sources
                st.markdown("#### 📄 Document Citations")
                for i, c in enumerate(sources):
                    with st.expander(f"Citation {i+1} — {c['source']} (Page {c['page']})", expanded=False):
                        st.markdown(f"*{c['text']}*")
        else:
            st.warning("Please enter a query to search.")
