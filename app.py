"""
File Search POC - A simple RAG chatbot using Google's File Search API
"""
import streamlit as st
from google import genai
from google.genai import types
import time
import os
import uuid

# --- Configuration ---
def get_config(key: str, default: str = None):
    """Get config from Streamlit secrets or environment variables."""
    # Try Streamlit secrets first (for cloud deployment)
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass  # No secrets file exists, fall back to env vars
    # Fall back to environment variables
    return os.getenv(key, default)

MODEL = get_config("GEMINI_MODEL", "gemini-2.5-flash")
API_KEY = get_config("GEMINI_API_KEY")

# --- Initialize Google GenAI Client ---
@st.cache_resource
def get_client():
    if API_KEY:
        return genai.Client(api_key=API_KEY)
    return genai.Client()  # Will use GEMINI_API_KEY env var

client = get_client()

# --- Helper Functions ---
def list_stores():
    """List all file search stores."""
    try:
        return list(client.file_search_stores.list())
    except Exception as e:
        st.error(f"Error listing stores: {e}")
        return []

def create_store(name: str):
    """Create a new file search store."""
    try:
        store = client.file_search_stores.create(config={"display_name": name})
        return store
    except Exception as e:
        st.error(f"Error creating store: {e}")
        return None

def delete_store(store_name: str):
    """Delete a file search store."""
    try:
        client.file_search_stores.delete(name=store_name, config={"force": True})
        return True
    except Exception as e:
        st.error(f"Error deleting store: {e}")
        return False

def get_store_info(store_name: str):
    """Get store info including document counts."""
    try:
        return client.file_search_stores.get(name=store_name)
    except Exception as e:
        st.error(f"Error getting store info: {e}")
        return None

def upload_file(store_name: str, file):
    """Upload a file to a store."""
    # Use a simple ASCII temp filename to avoid encoding issues
    file_ext = os.path.splitext(file.name)[1] if '.' in file.name else ''
    temp_path = f"/tmp/{uuid.uuid4().hex}{file_ext}"
    
    with open(temp_path, "wb") as f:
        f.write(file.getbuffer())
    
    try:
        operation = client.file_search_stores.upload_to_file_search_store(
            file_search_store_name=store_name,
            file=temp_path,
            config={"display_name": file.name}
        )
        
        # Wait for upload to complete
        while not client.operations.get(operation).done:
            time.sleep(1)
        
        os.remove(temp_path)
        return True
    except Exception as e:
        st.error(f"Error uploading file: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False

def chat(store_name: str, message: str, history: list):
    """Send a message and get a response with file search."""
    # Build conversation contents
    contents = []
    for msg in history:
        contents.append(types.Content(
            role=msg["role"],
            parts=[types.Part(text=msg["content"])]
        ))
    contents.append(types.Content(
        role="user",
        parts=[types.Part(text=message)]
    ))
    
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                tools=[types.Tool(
                    file_search=types.FileSearch(
                        file_search_store_names=[store_name]
                    )
                )]
            )
        )
        
        # Extract citations
        citations = []
        if response.candidates and response.candidates[0].grounding_metadata:
            grounding = response.candidates[0].grounding_metadata
            if grounding.grounding_chunks:
                citations = list({
                    c.retrieved_context.title 
                    for c in grounding.grounding_chunks 
                    if c.retrieved_context and c.retrieved_context.title
                })
        
        return response.text, citations
    except Exception as e:
        return f"Error: {e}", []

# --- Page Config ---
st.set_page_config(
    page_title="File Search POC",
    page_icon="🔍",
    layout="wide"
)

# --- Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Outfit:wght@300;400;600;700&display=swap');
    
    :root {
        --bg-dark: #0a0a0f;
        --bg-card: #12121a;
        --accent: #00d4aa;
        --accent-dim: #00d4aa33;
        --text: #e8e8ed;
        --text-dim: #8888a0;
        --border: #2a2a3a;
    }
    
    .stApp {
        background: linear-gradient(145deg, #0a0a0f 0%, #0f0f18 50%, #0a0f14 100%);
    }
    
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
    }
    
    .main-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4aa, #00a8ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: var(--text-dim);
        margin-bottom: 2rem;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d14 0%, #12121a 100%);
        border-right: 1px solid var(--border);
    }
    
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stTextInput label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: var(--text-dim);
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    /* Card styling */
    .info-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    
    .info-card h4 {
        font-family: 'Outfit', sans-serif;
        color: var(--accent);
        margin: 0 0 0.5rem 0;
        font-size: 0.9rem;
    }
    
    /* Chat messages */
    .stChatMessage {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
    }
    
    /* Citation pill */
    .citation {
        display: inline-block;
        background: var(--accent-dim);
        border: 1px solid var(--accent);
        border-radius: 20px;
        padding: 0.25rem 0.75rem;
        margin: 0.25rem 0.25rem 0.25rem 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: var(--accent);
    }
    
    /* Buttons */
    .stButton > button {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 212, 170, 0.2);
    }
    
    /* File uploader */
    .stFileUploader {
        border: 2px dashed var(--border) !important;
        border-radius: 12px !important;
        background: var(--bg-card) !important;
    }
    
    /* Status indicators */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-active { background: var(--accent); box-shadow: 0 0 8px var(--accent); }
    .status-empty { background: var(--text-dim); }
    
    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-dark); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--accent); }
    
    /* File list item */
    .file-item {
        background: #1a1a24;
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin-bottom: 0.5rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: var(--text);
    }
</style>
""", unsafe_allow_html=True)

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = {}  # Dict: store_name -> list of messages

if "current_store" not in st.session_state:
    st.session_state.current_store = None

# --- Sidebar: Space Management ---
with st.sidebar:
    st.markdown('<p class="main-title">🔍 File Search</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">RAG POC with Gemini</p>', unsafe_allow_html=True)
    
    st.divider()
    
    # List existing stores
    stores = list_stores()
    store_options = {s.display_name or s.name: s.name for s in stores}
    
    if store_options:
        selected_display = st.selectbox(
            "SELECT SPACE",
            options=list(store_options.keys()),
            index=0
        )
        st.session_state.current_store = store_options[selected_display]
    else:
        st.info("No spaces yet. Create one below.")
        st.session_state.current_store = None
    
    st.divider()
    
    # Create new store
    st.markdown("##### ➕ New Space")
    new_store_name = st.text_input("Space name", placeholder="e.g., Research Papers", label_visibility="collapsed")
    if st.button("Create Space", use_container_width=True, type="primary"):
        if new_store_name:
            with st.spinner("Creating space..."):
                store = create_store(new_store_name)
                if store:
                    st.success(f"Created: {new_store_name}")
                    st.rerun()
        else:
            st.warning("Enter a name")
    
    # Delete current store
    if st.session_state.current_store:
        st.divider()
        if st.button("🗑️ Delete Current Space", use_container_width=True):
            if delete_store(st.session_state.current_store):
                st.session_state.current_store = None
                if st.session_state.current_store in st.session_state.messages:
                    del st.session_state.messages[st.session_state.current_store]
                st.rerun()
    
    # Model info
    st.divider()
    st.markdown(f"""
    <div class="info-card">
        <h4>⚙️ Config</h4>
        <code style="color: #8888a0; font-size: 0.75rem;">{MODEL}</code>
    </div>
    """, unsafe_allow_html=True)

# --- Main Content ---
if st.session_state.current_store:
    col1, col2 = st.columns([1, 2])
    
    # --- Left Column: Files ---
    with col1:
        st.markdown("### 📁 Files")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Drop files here",
            type=["txt", "pdf", "md", "json", "csv", "docx", "html"],
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            if st.button("Upload", use_container_width=True, type="primary"):
                with st.spinner(f"Uploading {uploaded_file.name}..."):
                    if upload_file(st.session_state.current_store, uploaded_file):
                        st.success("Uploaded!")
                        st.rerun()
        
        # Show document count from store info
        st.markdown("---")
        store_info = get_store_info(st.session_state.current_store)
        
        if store_info:
            active = int(store_info.active_documents_count or 0)
            pending = int(store_info.pending_documents_count or 0)
            failed = int(store_info.failed_documents_count or 0)
            
            if active > 0 or pending > 0:
                st.markdown(f"""
                <div class="file-item">
                    <span style="color: #00d4aa;">●</span> {active} document{'s' if active != 1 else ''} ready
                </div>
                """, unsafe_allow_html=True)
                
                if pending > 0:
                    st.markdown(f"""
                    <div class="file-item">
                        <span style="color: #ffaa00;">●</span> {pending} processing...
                    </div>
                    """, unsafe_allow_html=True)
                
                if failed > 0:
                    st.markdown(f"""
                    <div class="file-item">
                        <span style="color: #ff4444;">●</span> {failed} failed
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(
                    '<p style="color: #8888a0; font-size: 0.85rem; text-align: center; padding: 2rem;">No files yet</p>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                '<p style="color: #8888a0; font-size: 0.85rem; text-align: center; padding: 2rem;">No files yet</p>',
                unsafe_allow_html=True
            )
    
    # --- Right Column: Chat ---
    with col2:
        st.markdown("### 💬 Chat")
        
        # Initialize message history for this store
        if st.session_state.current_store not in st.session_state.messages:
            st.session_state.messages[st.session_state.current_store] = []
        
        messages = st.session_state.messages[st.session_state.current_store]
        
        # Chat container
        chat_container = st.container(height=500)
        
        with chat_container:
            if not messages:
                st.markdown(
                    '<p style="color: #8888a0; text-align: center; padding: 4rem;">Ask anything about your files...</p>',
                    unsafe_allow_html=True
                )
            
            for msg in messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg.get("citations"):
                        citation_html = "".join([f'<span class="citation">📎 {c}</span>' for c in msg["citations"]])
                        st.markdown(f"<div style='margin-top: 0.5rem;'>{citation_html}</div>", unsafe_allow_html=True)
        
        # Chat input
        if prompt := st.chat_input("Ask about your documents..."):
            # Add user message
            messages.append({"role": "user", "content": prompt})
            
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
            
            # Get response
            with st.spinner("Thinking..."):
                response_text, citations = chat(
                    st.session_state.current_store,
                    prompt,
                    messages[:-1]  # Exclude the just-added user message
                )
            
            # Add assistant message
            messages.append({
                "role": "model",
                "content": response_text,
                "citations": citations
            })
            
            st.rerun()
        
        # Clear chat
        if messages and st.button("Clear Chat", use_container_width=True):
            st.session_state.messages[st.session_state.current_store] = []
            st.rerun()

else:
    # No store selected
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem;">
        <p class="main-title" style="font-size: 3rem;">🔍</p>
        <h2 style="color: #e8e8ed; margin-bottom: 1rem;">Welcome to File Search POC</h2>
        <p style="color: #8888a0; max-width: 500px; margin: 0 auto;">
            Create a space in the sidebar to get started. Each space is a separate 
            knowledge base where you can upload documents and chat with them.
        </p>
    </div>
    """, unsafe_allow_html=True)

