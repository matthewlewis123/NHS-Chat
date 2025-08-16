import streamlit as st
from typing import Dict, List
from pathlib import Path
import os

try:
    from query_rag import RAGSystem
except ImportError as e:
    st.error(f"Import error: {e}. Please ensure all required modules are available.")
    st.stop()


# --- assets / logo setup ---

# Get the correct path to the logo file
current_dir = Path(__file__).parent
LOGO_PATH = current_dir / "nhs_logo.png"  

# Check if logo exists, if not use a fallback
if LOGO_PATH.exists():
    logo_path_str = str(LOGO_PATH)
else:
    # Fallback if logo doesn't exist
    logo_path_str = None

LOGO_ALT = "NHS logo"

# set a page icon (use emoji as fallback if logo file doesn't exist)
page_icon = "🩺"  # Use emoji instead of file path for better compatibility
st.set_page_config(page_title="NHS Clinical Assistant", layout="wide", page_icon=page_icon)

# Initialize RAG System
def get_rag_system():
    """Initialize the RAG system"""
    try:
        return RAGSystem()
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")
        return None

# Initialize RAG system once at startup
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = get_rag_system()

rag_system = st.session_state.rag_system
if rag_system is None:
    st.error("RAG system failed to initialize. Please check your configuration.")
    st.stop()

# --- Helper Functions ---
def display_sources(sources_data: List[Dict]):
    """Display sources with clean NHS formatting"""
    if not sources_data:
        st.markdown("No sources available for this response.")
        return

    for idx, source_info in enumerate(sources_data):
        # Get metadata from source_info
        metadata = source_info.get('metadata', {})
        clean_section = metadata.get('clean_section', 'Unknown Section')
        url = metadata.get('url', '')
        
        source_text = f"<div class='source-item'><div class='source-title'>Source {idx+1}: {clean_section}</div>"
        if url:
            source_text += f"<div class='source-link'>🔗 <a href='{url}' target='_blank'>View online</a></div>"
        source_text += "</div>"
        st.markdown(source_text, unsafe_allow_html=True)


def initialize_session_state():
    # Common state
    if "app_mode" not in st.session_state:
        st.session_state.app_mode = "NHS Chat"

    # Chat specific state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "query" not in st.session_state:
        st.session_state.query = ""
    if "processing_query" not in st.session_state:
        st.session_state.processing_query = False 
    if "query_to_run_next" not in st.session_state: 
        st.session_state.query_to_run_next = None 
    if "similarity_k" not in st.session_state:
        st.session_state.similarity_k = 5
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = "gemini-2.5-flash"


initialize_session_state()

# --- STYLING ---
st.markdown("""
<style>
  /* Import a modern font */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

  :root{
    --bg:#f6f8fa;
    --card:#ffffff;
    --accent:#1f7a8c;    /* deep teal */
    --accent-2:#2b6777;
    --muted:#6b7280;
    --green:#16a34a;
  }

  html, body, [class*="css"]  {
    font-family: 'Inter', system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    color: #0f1724;
    background: linear-gradient(180deg, var(--bg) 0%, #ffffff 100%);
  }

  /* Main container */
  .main .block-container {
    padding-top: 8px;
    padding-left: 28px;
    padding-right: 28px;
    max-width: 1200px;
    margin: 0 auto;
  }

  /* Header */
  .app-header{
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:12px;
    margin-bottom:12px;
  }
  .app-title {
    display:flex;
    align-items:center;
    gap:12px;
  }
  .logo {
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    color: white;
    width:44px;
    height:44px;
    display:flex;
    align-items:center;
    justify-content:center;
    font-weight:700;
    border-radius:10px;
    box-shadow: 0 6px 18px rgba(43,103,119,0.12);
  }
  .title-text { font-size:20px; font-weight:700; letter-spacing: -0.2px; }

  /* Card-like chat area */
  .chat-card {
    background: var(--card);
    border-radius: 12px;
    padding: 18px;
    box-shadow: 0 6px 20px rgba(20,23,30,0.04);
  }

  /* Chat bubbles */
  .chat-bubble {
    max-width:78%;
    padding:12px 14px;
    border-radius:12px;
    margin-bottom:10px;
    line-height:1.4;
    box-shadow: 0 4px 12px rgba(15,23,36,0.04);
  }
  .chat-bubble.user {
    background: linear-gradient(90deg, rgba(255,255,255,1), rgba(255,255,255,1));
    border: 1px solid rgba(16,24,40,0.06);
    margin-left:auto;
    border-bottom-right-radius:6px;
  }
  .chat-bubble.assistant {
    background: linear-gradient(180deg, rgba(31,122,140,0.06), rgba(43,103,119,0.03));
    border-left: 4px solid var(--accent);
    border-bottom-left-radius:6px;
    margin-right:auto;
  }

  .chat-meta {
    font-size:12px;
    color:var(--muted);
    margin-bottom:6px;
  }

  /* Sources */
  .source-item { padding:10px 12px; border-radius:8px; background:#fbfcfd; margin-bottom:8px; border:1px solid rgba(16,24,40,0.03); }
  .source-title { font-weight:600; color:var(--accent-2); margin-bottom:4px; }
  .source-link a { color:var(--accent); text-decoration:none; font-weight:600; }
  .source-link a:hover { text-decoration:underline; }

  /* Buttons / chips */
  .stButton>button {
    border-radius:999px !important;
    padding:8px 14px !important;
    background: linear-gradient(90deg, #ffffff, #f7fafb) !important;
    border: 1px solid rgba(16,24,40,0.06) !important;
    color: #0f1724 !important;
    box-shadow: 0 4px 10px rgba(12,18,28,0.04);
  }
  .stButton>button:active { transform: translateY(1px); }

  /* Suggested chips */
  .suggested-chips { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:12px; }
  .chip {
    display:inline-flex;
    gap:8px;
    align-items:center;
    padding:8px 12px;
    border-radius:999px;
    background: rgba(47,128,237,0.06);
    color: #164e63;
    border:1px solid rgba(47,128,237,0.12);
    cursor:pointer;
    font-weight:600;
    font-size:13px;
  }

  /* small helpers */
  .muted { color:var(--muted); font-size:13px; }
  hr.stDivider { border: none; height:1px; background: linear-gradient(90deg, transparent, rgba(15,23,36,0.06), transparent); margin: 16px 0; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("🩺 NHS Clinical Assistant")

    st.header("⚙️ Settings") 

    llm_options = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"]
    try:
        current_llm_index = llm_options.index(st.session_state.llm_model)
    except ValueError:
        current_llm_index = 0
        st.session_state.llm_model = llm_options[0]

    selected_llm = st.selectbox(
        "LLM Model", 
        options=llm_options,
        key="llm_model_selector", 
        index=current_llm_index
    )
    if selected_llm != st.session_state.llm_model:
        st.session_state.llm_model = selected_llm
    
    st.markdown("---")
    
    def new_chat_callback(): 
        st.session_state.chat_history = []
        st.session_state.query = ""

    if st.button("🗑️ New Chat", key="new_chat", on_click=new_chat_callback): 
        pass


# --- MAIN APPLICATION AREA ---
# Header area with logo, title and small info
header_cols = st.columns([0.12, 0.76, 0.12])
with header_cols[0]:
    if logo_path_str and os.path.exists(logo_path_str):
        # Use HTML to control sizing while preserving quality
        import base64
        with open(logo_path_str, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode()
        
        st.markdown(f"""
        <div style="display: flex; justify-content: center; align-items: center;">
            <img src="data:image/png;base64,{img_base64}" 
                 style="max-width: 100%; height: auto; max-height: 60px; object-fit: contain;" 
                 alt="NHS Logo">
        </div>
        """, unsafe_allow_html=True)
    else:
        # Fallback: display emoji or text if logo not found
        st.markdown('<div style="font-size:56px;">🩺</div>', unsafe_allow_html=True)



with header_cols[1]:
    st.markdown("""
    <div style="display:flex; flex-direction:column; justify-content:center;">
      <div class="title-text">NHS Clinical Assistant</div>
      <div class="muted">Trusted NHS content · Evidence-backed summaries</div>
    </div>
    """, unsafe_allow_html=True)

with header_cols[2]:
    st.markdown(f"""
    <div style="display:flex; gap:12px; align-items:center; justify-content:flex-end;">
      <div style="font-size:13px; color:var(--muted);">Model:</div>
      <div style="font-weight:700; color:var(--accent-2);">{st.session_state.llm_model}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="chat-card">', unsafe_allow_html=True)

def submit_and_process_query(query_to_send: str, display_query_text: str):
    st.session_state.processing_query = True 
    
    try:
        with st.spinner("Retrieving relevant NHS information..."):
            response_chunks = []
            sources_data = []
            temp_response_placeholder = st.empty()
            
            for chunk, chunk_sources_data in rag_system.query_rag_stream(
                query_to_send,
                st.session_state.llm_model, 
                info_source="NHS",
                similarity_k=st.session_state.similarity_k,
            ):
                response_chunks.append(chunk)
                sources_data = chunk_sources_data 
                
                temp_response_placeholder.markdown(
                    f"<div class='chat-bubble assistant'>{''.join(response_chunks)}</div>",
                    unsafe_allow_html=True
                )
            
            final_response = ''.join(response_chunks)
            temp_response_placeholder.empty()

            st.session_state.chat_history.append({
                "query_sent": query_to_send,
                "display_query": display_query_text,
                "response": final_response,
                "sources_data": sources_data,
                "llm_model": st.session_state.llm_model
            })
            
    except Exception as e:
        st.error(f"Error processing query: {e}")
    finally:
        st.session_state.processing_query = False
        st.rerun()

# Display chat history
for i, chat_entry in enumerate(st.session_state.chat_history):
    # user
    st.markdown(f"<div class='chat-bubble user'><div class='chat-meta'>You</div><div>{chat_entry['display_query']}</div></div>", unsafe_allow_html=True)
    
    response_info = f"(LLM: {chat_entry.get('llm_model', 'N/A')})"
    
    st.markdown(f"<div class='chat-bubble assistant'><div class='chat-meta'>Assistant {response_info}</div><div>{chat_entry['response']}</div></div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:8px; font-weight:600;'>📚 Sources</div>", unsafe_allow_html=True)
    with st.expander("View Sources", expanded=False):
        sources_data = chat_entry.get("sources_data", [])
        if sources_data:
            display_sources(sources_data)
        else:
            st.markdown("No sources available for this response.")
    st.markdown('<hr class="stDivider">', unsafe_allow_html=True)

# Suggested queries - styled as chips
st.markdown("<div style='margin-top:6px; margin-bottom:8px; font-weight:600;'>💡 Suggested Queries</div>", unsafe_allow_html=True)
suggested_queries_list = [
    "What are the symptoms of ADHD in adults?",
    "How is type 2 diabetes diagnosed?",
    "What are the treatment options for depression?"
]

# Render chips in a row. Use buttons beneath for accessibility & state handling.
chip_cols = st.columns([1,1,1])
for idx, sq in enumerate(suggested_queries_list):
    with chip_cols[idx]:
        if st.button(sq, key=f"suggested_{idx}", disabled=st.session_state.processing_query):
            st.session_state.processing_query = True
            st.session_state.query_to_run_next = sq
            st.rerun()

st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

# User input section
user_query = st.chat_input(
    "e.g., What are the symptoms of ADHD?", 
    max_chars=1000, 
    disabled=st.session_state.processing_query
)

if user_query:
    st.session_state.processing_query = True
    st.session_state.query_to_run_next = user_query
    st.rerun()

# Process query if one is set to run next
if st.session_state.get("query_to_run_next"):
    query_to_process = st.session_state.query_to_run_next
    st.session_state.query_to_run_next = None  # Clear it so it doesn't run again
    submit_and_process_query(query_to_process, query_to_process)

st.markdown('</div>', unsafe_allow_html=True)

# --- Footer with Licensing Information ---
st.markdown("---")
st.caption("""
**Data Usage and Licensing:**
This tool utilizes information from NHS sources, which is made available under their respective open licensing terms.
- **NHS:** Content is used under the terms of the Open Government Licence. For full details, please refer to the [NHS Terms and Conditions](https://www.nhs.uk/our-policies/terms-and-conditions/).

Always consult the official sources for the most accurate, complete, and up-to-date information.
""")