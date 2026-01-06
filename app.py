"""
Deal Room Assistant - Document intelligence for Private Equity due diligence
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
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)

MODEL = get_config("GEMINI_MODEL", "gemini-2.5-flash")
API_KEY = get_config("GEMINI_API_KEY")

# --- Initialize Google GenAI Client ---
@st.cache_resource
def get_client():
    if API_KEY:
        return genai.Client(api_key=API_KEY)
    return genai.Client()

client = get_client()

# --- Helper Functions ---
def list_stores():
    """List all deal rooms."""
    try:
        return list(client.file_search_stores.list())
    except Exception as e:
        st.error(f"Error listing deal rooms: {e}")
        return []

def create_store(name: str):
    """Create a new deal room."""
    try:
        store = client.file_search_stores.create(config={"display_name": name})
        return store
    except Exception as e:
        st.error(f"Error creating deal room: {e}")
        return None

def delete_store(store_name: str):
    """Delete a deal room."""
    try:
        client.file_search_stores.delete(name=store_name, config={"force": True})
        return True
    except Exception as e:
        st.error(f"Error deleting deal room: {e}")
        return False

def get_store_info(store_name: str):
    """Get deal room info including document counts."""
    try:
        return client.file_search_stores.get(name=store_name)
    except Exception as e:
        st.error(f"Error getting deal room info: {e}")
        return None

def upload_file(store_name: str, file):
    """Upload a document to a deal room."""
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
        
        while not client.operations.get(operation).done:
            time.sleep(1)
        
        os.remove(temp_path)
        return True
    except Exception as e:
        st.error(f"Error uploading document: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False

def chat(store_name: str, message: str, history: list):
    """Query documents in the deal room."""
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
    page_title="Deal Room Assistant",
    page_icon="◆",
    layout="wide"
)

# --- Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
    
    :root {
        --bg-primary: #0f1419;
        --bg-secondary: #1a1f26;
        --bg-tertiary: #242b33;
        --accent: #3b82f6;
        --accent-hover: #2563eb;
        --accent-dim: rgba(59, 130, 246, 0.15);
        --success: #10b981;
        --warning: #f59e0b;
        --error: #ef4444;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --border: #2d3748;
        --border-light: #3d4a5c;
    }
    
    * {
        font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    .stApp {
        background: var(--bg-primary);
    }
    
    /* Hide all sidebar toggle/collapse buttons */
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"],
    button[kind="headerNoPadding"],
    [data-testid="baseButton-headerNoPadding"],
    .st-emotion-cache-1egp75f,
    [aria-label="Collapse sidebar"],
    [aria-label="Expand sidebar"] {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
        overflow: hidden !important;
    }
    
    /* Typography */
    h1, h2, h3, h4 {
        font-weight: 600 !important;
        color: var(--text-primary) !important;
    }
    
    .brand {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.25rem;
    }
    
    .brand-icon {
        width: 32px;
        height: 32px;
        background: linear-gradient(135deg, var(--accent), #8b5cf6);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        color: white;
    }
    
    .brand-text {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-primary);
        letter-spacing: -0.02em;
    }
    
    .tagline {
        font-size: 0.75rem;
        color: var(--text-muted);
        margin-bottom: 1.5rem;
        font-family: 'IBM Plex Mono', monospace !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: var(--bg-secondary);
        border-right: 1px solid var(--border);
    }
    
    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }
    
    /* Form labels */
    .stSelectbox label, .stTextInput label {
        font-size: 0.7rem !important;
        font-weight: 500 !important;
        color: var(--text-muted) !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem !important;
    }
    
    /* Input fields - improved visibility */
    .stTextInput input {
        background: var(--bg-tertiary) !important;
        border: 1px solid var(--border-light) !important;
        border-radius: 6px !important;
        color: var(--text-primary) !important;
        font-size: 0.9rem !important;
        padding: 0.6rem 0.75rem !important;
    }
    
    .stTextInput input::placeholder {
        color: var(--text-muted) !important;
    }
    
    .stTextInput input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px var(--accent-dim) !important;
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background: var(--bg-tertiary) !important;
        border: 1px solid var(--border-light) !important;
        border-radius: 6px !important;
    }
    
    /* Buttons */
    .stButton > button {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        border-radius: 6px !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.15s ease !important;
    }
    
    .stButton > button[data-testid="baseButton-primary"] {
        background: var(--accent) !important;
        border: none !important;
    }
    
    .stButton > button[data-testid="baseButton-primary"]:hover {
        background: var(--accent-hover) !important;
    }
    
    .stButton > button[data-testid="baseButton-secondary"] {
        background: transparent !important;
        border: 1px solid var(--border-light) !important;
        color: var(--text-secondary) !important;
    }
    
    .stButton > button[data-testid="baseButton-secondary"]:hover {
        background: var(--bg-tertiary) !important;
        border-color: var(--text-muted) !important;
    }
    
    /* Section label */
    .section-label {
        font-size: 0.7rem;
        font-weight: 500;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.75rem;
    }
    
    /* Card */
    .card {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Document status */
    .doc-status {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 0.75rem;
        background: var(--bg-tertiary);
        border-radius: 6px;
        font-size: 0.8rem;
        color: var(--text-secondary);
        margin-bottom: 0.5rem;
        font-family: 'IBM Plex Mono', monospace !important;
    }
    
    .doc-status .dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
    }
    
    .dot-ready { background: var(--success); }
    .dot-pending { background: var(--warning); }
    .dot-failed { background: var(--error); }
    
    /* Citation */
    .citation {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        background: var(--accent-dim);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 4px;
        padding: 0.2rem 0.5rem;
        margin: 0.2rem 0.2rem 0.2rem 0;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.7rem;
        color: var(--accent);
    }
    
    /* Chat */
    .stChatMessage {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }
    
    .stChatInput > div {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-light) !important;
        border-radius: 8px !important;
    }
    
    .stChatInput input {
        color: var(--text-primary) !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: var(--bg-secondary) !important;
        border: 1px dashed var(--border-light) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: var(--accent) !important;
    }
    
    /* Divider */
    hr {
        border-color: var(--border) !important;
        margin: 1rem 0 !important;
    }
    
    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 2rem 1rem;
        color: var(--text-muted);
        font-size: 0.85rem;
    }
    
    /* Welcome */
    .welcome {
        text-align: center;
        padding: 4rem 2rem;
        max-width: 480px;
        margin: 0 auto;
    }
    
    .welcome-icon {
        width: 64px;
        height: 64px;
        background: linear-gradient(135deg, var(--accent), #8b5cf6);
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        color: white;
        margin: 0 auto 1.5rem;
    }
    
    .welcome h2 {
        font-size: 1.5rem;
        margin-bottom: 0.75rem;
    }
    
    .welcome p {
        color: var(--text-secondary);
        font-size: 0.9rem;
        line-height: 1.6;
    }
    
    /* Config card */
    .config-card {
        background: var(--bg-tertiary);
        border-radius: 6px;
        padding: 0.75rem;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.7rem;
        color: var(--text-muted);
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--border-light); }
</style>
""", unsafe_allow_html=True)

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = {}

if "current_store" not in st.session_state:
    st.session_state.current_store = None

# --- Sidebar ---
with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div class="brand-icon">◆</div>
        <span class="brand-text">Deal Room</span>
    </div>
    <div class="tagline">Due Diligence Assistant</div>
    """, unsafe_allow_html=True)
    
    # List existing deal rooms
    stores = list_stores()
    store_options = {s.display_name or s.name: s.name for s in stores}
    
    if store_options:
        selected_display = st.selectbox(
            "ACTIVE DEAL ROOM",
            options=list(store_options.keys()),
            index=0
        )
        st.session_state.current_store = store_options[selected_display]
    else:
        st.markdown('<p class="empty-state">No deal rooms yet</p>', unsafe_allow_html=True)
        st.session_state.current_store = None
    
    st.divider()
    
    # Create new deal room
    st.markdown('<div class="section-label">New Deal Room</div>', unsafe_allow_html=True)
    new_store_name = st.text_input(
        "Deal room name",
        placeholder="e.g., Acme Corp Acquisition",
        label_visibility="collapsed"
    )
    if st.button("Create", use_container_width=True, type="primary"):
        if new_store_name:
            with st.spinner("Creating deal room..."):
                store = create_store(new_store_name)
                if store:
                    st.success(f"Created: {new_store_name}")
                    st.rerun()
        else:
            st.warning("Enter a deal name")
    
    # Delete current deal room
    if st.session_state.current_store:
        st.divider()
        if st.button("Delete Deal Room", use_container_width=True, type="secondary"):
            if delete_store(st.session_state.current_store):
                if st.session_state.current_store in st.session_state.messages:
                    del st.session_state.messages[st.session_state.current_store]
                st.session_state.current_store = None
                st.rerun()
    
    # Model info
    st.divider()
    st.markdown(f'<div class="config-card">Model: {MODEL}</div>', unsafe_allow_html=True)

# --- Main Content ---
if st.session_state.current_store:
    col1, col2 = st.columns([1, 2.5])
    
    # --- Left Column: Documents ---
    with col1:
        st.markdown("#### Documents")
        
        uploaded_file = st.file_uploader(
            "Upload deal materials",
            type=["txt", "pdf", "md", "json", "csv", "docx", "xlsx", "html"],
            label_visibility="collapsed",
            help="CIMs, financials, legal docs, memos"
        )
        
        if uploaded_file:
            if st.button("Upload", use_container_width=True, type="primary"):
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    if upload_file(st.session_state.current_store, uploaded_file):
                        st.success("Document indexed")
                        st.rerun()
        
        st.markdown("---")
        st.markdown('<div class="section-label">Indexed Documents</div>', unsafe_allow_html=True)
        
        store_info = get_store_info(st.session_state.current_store)
        
        if store_info:
            active = int(store_info.active_documents_count or 0)
            pending = int(store_info.pending_documents_count or 0)
            failed = int(store_info.failed_documents_count or 0)
            
            if active > 0 or pending > 0:
                if active > 0:
                    st.markdown(f'''
                    <div class="doc-status">
                        <span class="dot dot-ready"></span>
                        {active} document{"s" if active != 1 else ""} ready
                    </div>
                    ''', unsafe_allow_html=True)
                
                if pending > 0:
                    st.markdown(f'''
                    <div class="doc-status">
                        <span class="dot dot-pending"></span>
                        {pending} processing
                    </div>
                    ''', unsafe_allow_html=True)
                
                if failed > 0:
                    st.markdown(f'''
                    <div class="doc-status">
                        <span class="dot dot-failed"></span>
                        {failed} failed
                    </div>
                    ''', unsafe_allow_html=True)
            else:
                st.markdown('<p class="empty-state">No documents uploaded</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="empty-state">No documents uploaded</p>', unsafe_allow_html=True)
    
    # --- Right Column: Chat ---
    with col2:
        st.markdown("#### Ask about this deal")
        
        if st.session_state.current_store not in st.session_state.messages:
            st.session_state.messages[st.session_state.current_store] = []
        
        messages = st.session_state.messages[st.session_state.current_store]
        
        chat_container = st.container(height=480)
        
        with chat_container:
            if not messages:
                st.markdown('''
                <p class="empty-state">
                    Ask questions about the deal materials.<br>
                    <span style="font-size: 0.75rem; opacity: 0.7;">
                        e.g., "Summarize the key financial metrics" or "What are the main risks?"
                    </span>
                </p>
                ''', unsafe_allow_html=True)
            
            for msg in messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg.get("citations"):
                        citation_html = "".join([
                            f'<span class="citation">↗ {c}</span>' 
                            for c in msg["citations"]
                        ])
                        st.markdown(f"<div style='margin-top: 0.5rem;'>{citation_html}</div>", unsafe_allow_html=True)
        
        if prompt := st.chat_input("Ask about the deal materials..."):
            messages.append({"role": "user", "content": prompt})
            
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
            
            with st.spinner("Analyzing documents..."):
                response_text, citations = chat(
                    st.session_state.current_store,
                    prompt,
                    messages[:-1]
                )
            
            messages.append({
                "role": "model",
                "content": response_text,
                "citations": citations
            })
            
            st.rerun()
        
        if messages:
            col_a, col_b = st.columns([3, 1])
            with col_b:
                if st.button("Clear", use_container_width=True, type="secondary"):
                    st.session_state.messages[st.session_state.current_store] = []
                    st.rerun()

else:
    st.markdown("""
    <div class="welcome">
        <div class="welcome-icon">◆</div>
        <h2>Deal Room Assistant</h2>
        <p>
            Upload deal materials — CIMs, financials, legal documents — and ask questions 
            to accelerate your due diligence and underwriting process.
        </p>
        <p style="margin-top: 1rem; font-size: 0.8rem; color: var(--text-muted);">
            Create a deal room in the sidebar to get started.
        </p>
    </div>
    """, unsafe_allow_html=True)
