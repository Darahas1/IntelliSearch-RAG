import streamlit as st
import os
import sys
import tempfile
import time

# Add parent directory to path to find main_logic
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from main_logic import HybridRAG, setup_environment, load_and_split_document

# --- Page Configuration ---
st.set_page_config(
    page_title="IntelliSearch RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling (Modern & Clean) ---
st.markdown("""
<style>
    /* Chat bubbles */
    .stChatMessage {
        border-radius: 10px;
        margin-bottom: 10px;
    }
    /* Sidebar adjustments */
    [data-testid="stSidebar"] {
        background-color: #grey;
    }
    /* Header */
    h1 {
        color: #1E3A8A;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_system" not in st.session_state:
    try:
        groq_key, tavily_key = setup_environment()
        st.session_state.rag_system = HybridRAG(groq_key, tavily_key)
        st.session_state.processing_complete = False
        st.session_state.current_file_name = None
    except ValueError as e:
        st.error(f"❌ Configuration Error: {e}")
        st.stop()

# --- Sidebar: File Upload ---
with st.sidebar:
    st.header("📂 Document Center")
    uploaded_file = st.file_uploader(
        "Upload a Knowledge Base (PDF/TXT)", 
        type=["pdf", "txt"],
        help="Upload a document to chat with it."
    )

    if uploaded_file:
        # Check if this specific file has already been processed to avoid redundant computation
        if st.session_state.current_file_name != uploaded_file.name:
            with st.status("🚀 Processing document...", expanded=True) as status:
                st.write("Reading file...")
                
                # Create a temp file to store the upload
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                try:
                    st.write("Splitting and Embedding...")
                    docs = load_and_split_document(tmp_file_path)
                    st.session_state.rag_system.setup_document_retriever(docs)
                    
                    # Update state
                    st.session_state.processing_complete = True
                    st.session_state.current_file_name = uploaded_file.name
                    status.update(label="✅ Ready to Chat!", state="complete", expanded=False)
                    
                except Exception as e:
                    st.error(f"Error processing file: {e}")
                finally:
                    # Cleanup
                    if os.path.exists(tmp_file_path):
                        os.remove(tmp_file_path)
        else:
            st.success(f"Active Document: {uploaded_file.name}")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- Main Chat Interface ---
st.title("🧠 IntelliSearch Assistant")
st.caption("Hybrid RAG: Powered by Llama 3.3, FAISS, and Tavily Web Search")

# 1. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If there are context details (only for assistant messages), show them in an expander
        if "context" in message and message["context"]:
             with st.expander("🔍 View Sources & Context"):
                st.markdown("**Document Context:**")
                st.text(message["context"]["doc"])
                st.divider()
                st.markdown("**Web Context:**")
                st.text(message["context"]["web"])

# 2. Handle User Input
if prompt := st.chat_input("Ask a question about your document or the web..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_data = st.session_state.rag_system.ask(prompt)
            
            answer = response_data["answer"]
            doc_ctx = response_data["document_context"]
            web_ctx = response_data["web_context"]
            
            st.markdown(answer)
            
            # Show sources immediately for this turn
            with st.expander("🔍 View Sources & Context"):
                st.markdown("**Document Context:**")
                st.text(doc_ctx)
                st.divider()
                st.markdown("**Web Context:**")
                st.text(web_ctx)

    # Save assistant message to history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer,
        "context": {"doc": doc_ctx, "web": web_ctx}
    })