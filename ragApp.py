import os
import tempfile
import fitz  # PyMuPDF
import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_ollama import ChatOllama, OllamaEmbeddings

from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)

from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Exam Assistant",
    page_icon="📖",
    layout="wide"
)

st.title("📊 Exam Assistant")
st.caption("100% Offline RAG tailored for CA, Finance, and Actuarial exam preparation.")

# Initialize Session State
if "ensemble_retriever" not in st.session_state:
    st.session_state.ensemble_retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "all_docs" not in st.session_state:
    st.session_state.all_docs = []

# -----------------------------------------------------------------------------
# 2. Sidebar Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Local Configuration")
    
    # Model Selector - Recommended models for math & finance
    selected_model = st.selectbox(
        "Local Model (Ollama)",
        ["llama3.2", "deepseek-r1:8b", "qwen2.5:7b", "qwen2.5:14b"],
        index=0,
        help="For complex math and CA calculations, 'deepseek-r1:8b' or 'qwen2.5:7b/14b' offer significantly higher mathematical accuracy."
    )
    
    st.divider()
    st.header("🎯 Study Mode")
    study_mode = st.radio(
        "Select Mode",
        ["CA Exam Solver (With Working Notes)", "Strict PDF Search", "Formula & Concept Extractor"],
        help="CA Exam Solver structures outputs according to official examination evaluation criteria."
    )
    
    st.divider()
    if st.button("🗑️ Reset Assistant", use_container_width=True):
        st.session_state.ensemble_retriever = None
        st.session_state.chat_history = []
        st.session_state.processed_files = []
        st.session_state.all_docs = []
        st.rerun()

# -----------------------------------------------------------------------------
# 3. Financial PDF Extraction & Indexing
# -----------------------------------------------------------------------------
def load_large_pdf(uploaded_file):
    """Fast extraction using PyMuPDF while attempting block preservation for tables."""
    docs = []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    doc = fitz.open(tmp_path)
    total_pages = len(doc)
    
    for page_num in range(total_pages):
        page = doc[page_num]
        # Extract text preserving layout blocks (helps maintain table lines)
        text = page.get_text("blocks")
        
        page_content = []
        for block in text:
            # block[4] contains the actual text string
            block_text = block[4].strip()
            if block_text:
                page_content.append(block_text)
                
        full_page_text = "\n\n".join(page_content)
        
        if full_page_text.strip():
            docs.append(
                Document(
                    page_content=full_page_text,
                    metadata={"source": uploaded_file.name, "page": page_num + 1}
                )
            )
    doc.close()
    os.remove(tmp_path)
    return docs, total_pages

uploaded_file = st.file_uploader("Upload Study Material / Past Papers (PDF)", type=["pdf"])

if uploaded_file:
    if uploaded_file.name not in st.session_state.processed_files:
        progress_bar = st.progress(0, text="Extracting financial text and tables...")
        
        # Step 1: Extraction
        raw_docs, total_pages = load_large_pdf(uploaded_file)
        st.session_state.all_docs = raw_docs
        
        # Step 2: Chunking (Larger chunks & overlap to preserve financial tables and formulas)
        progress_bar.progress(30, text=f"Processing {total_pages} pages into financial contexts...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, 
            chunk_overlap=300,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(raw_docs)
        
        # Step 3: Embeddings & Vector Store
        progress_bar.progress(60, text=f"Embedding {len(splits)} chunks with nomic-embed-text...")
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        faiss_vectorstore = FAISS.from_documents(splits, embeddings)
        faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 6})
        
        # Step 4: BM25 Sparse Index
        progress_bar.progress(85, text="Building keyword search for specific financial terms & numbers...")
        bm25_retriever = BM25Retriever.from_documents(splits)
        bm25_retriever.k = 6
        
        # Step 5: Hybrid Ensemble
        progress_bar.progress(95, text="Building Hybrid Retriever...")
        st.session_state.ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5]  # Equal weight for numerical exact match & semantic meaning
        )
        
        st.session_state.processed_files = [uploaded_file.name]
        progress_bar.progress(100, text="Complete!")
        st.success(f"Indexed {total_pages} pages ({len(splits)} chunks) successfully.")

# -----------------------------------------------------------------------------
# 4. Exam-Centric RAG Chain Construction
# -----------------------------------------------------------------------------
def get_rag_chain():
    if not st.session_state.ensemble_retriever:
        return None
        
    retriever = st.session_state.ensemble_retriever
    llm = ChatOllama(model=selected_model, temperature=0.1)

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question, rephrase it into a standalone question."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # Dynamic Prompts based on Mode
    if study_mode == "CA Exam Solver (With Working Notes)":
        system_prompt = (
            "You are a Senior Chartered Accountant and Financial Mathematics Professor evaluation assistant.\n"
            "Answer the user's question or solve the numerical problem step-by-step strictly based on the provided Document Context.\n"
            "Structure your answer using standard professional exam guidelines:\n"
            "1. **Core Concept / Standard / Formula**: State relevant definitions or governing standards.\n"
            "2. **Given Data**: List all numerical values extracted from the question/context.\n"
            "3. **Step-by-Step Calculation & Working Notes**: Show every arithmetic step clearly. Do not skip intermediate calculations.\n"
            "4. **Final Answer**: Clearly highlight the final numerical value or conclusion.\n\n"
            "If the context lacks required formulas or data, state what is missing.\n\n"
            "Document Context:\n{context}"
        )
    elif study_mode == "Formula & Concept Extractor":
        system_prompt = (
            "You are an exam revision assistant. Extract all formulas, key ratios, accounting standards, "
            "and primary definitions mentioned in or relevant to the user's prompt using the context below.\n"
            "Present formulas in clean markdown, define each variable, and provide a brief example of usage.\n\n"
            "Document Context:\n{context}"
        )
    else:  # Strict PDF Search
        system_prompt = (
            "You are a strict document assistant. Answer using ONLY the facts explicitly stated in the context.\n"
            "Do not extrapolate or calculate beyond what is directly supported.\n\n"
            "Document Context:\n{context}"
        )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, combine_docs_chain)

# -----------------------------------------------------------------------------
# 5. Practice & Quiz Generator Feature
# -----------------------------------------------------------------------------
if st.session_state.ensemble_retriever:
    with st.expander("📝 Generate Practice Exam Questions"):
        col1, col2 = st.columns([3, 1])
        with col1:
            topic = st.text_input("Enter Topic for Practice Question (e.g., 'Capital Budgeting', 'Ind AS 115', 'Black-Scholes')")
        with col2:
            q_type = st.selectbox("Type", ["Numerical Problem", "Theory / Case Study", "Multiple Choice (MCQ)"])
            
        if st.button("Generate Exam Question"):
            if topic:
                gen_llm = ChatOllama(model=selected_model, temperature=0.3)
                prompt = (
                    f"Based on the concepts of '{topic}', generate 1 high-quality exam-style {q_type} "
                    f"suitable for a CA/Finance exam. Provide the Question first, followed by the complete Solution with Working Notes hidden under a solution header."
                )
                res = gen_llm.invoke(prompt)
                st.markdown(res.content)

# -----------------------------------------------------------------------------
# 6. Chat Interface
# -----------------------------------------------------------------------------
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question, paste a problem, or ask for a calculation..."):
    if not st.session_state.ensemble_retriever:
        st.error("Please upload your PDF study material first.")
    else:
        st.chat_message("user").markdown(prompt)
        rag_chain = get_rag_chain()
        
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            
            lc_history = [
                ("human" if msg["role"] == "user" else "ai", msg["content"]) 
                for msg in st.session_state.chat_history
            ]

            res = rag_chain.invoke({
                "input": prompt,
                "chat_history": lc_history
            })
            
            answer = res["answer"]
            source_docs = res.get("context", [])

            response_placeholder.markdown(answer)
            
            if source_docs:
                with st.expander("📌 Source Context & Working Citations"):
                    for idx, doc in enumerate(source_docs):
                        page = doc.metadata.get("page", "N/A")
                        st.markdown(f"**[{idx+1}] Page {page}:**")
                        st.caption(f'"{doc.page_content.strip()[:300]}..."')
                        st.divider()

        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})