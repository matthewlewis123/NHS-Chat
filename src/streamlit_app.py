import streamlit as st
from typing import Dict, List

try:
    from query_rag import RAGSystem
except ImportError as e:
    st.error(f"Import error: {e}. Please ensure all required modules are available.")
    st.stop()


# --- Page Configuration and Initialization ---
st.set_page_config(page_title="NHS Clinical Assistant", layout="wide")


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
        
        source_text = f"**Source {idx+1}:** {clean_section}"
        st.markdown(source_text)

        if url:
            st.markdown(f"   🔗 [View Online]({url})")

        st.markdown("---")


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
    .main {background-color: #f9f9f9; font-family: Arial, sans-serif;}
    h1, h2, h3, h4, h5, h6 {color: #2b6777;}
    h1 {font-weight: bold;}
    [data-testid="stSidebar"] {background-color: #e8f0fe; padding: 10px;}
    .result-box {
        border-left: 4px solid #4CAF50;
        padding: 10px;
        background-color: #fff;
        margin-bottom: 10px;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    div.stTextArea > div { border-radius: 8px; }
    textarea { font-family: Arial, sans-serif; font-size: 16px; color: #333; resize: vertical; }
    .stButton>button { border-radius: 5px; }
    div.stSelectbox > label {
        font-size: 16px !important;
        font-weight: bold !important;
    }
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
st.title("🩺 NHS Clinical Assistant")
st.markdown("Ask questions and get relevant information from trusted NHS health condition sources.")

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
                    f"<div style='border-left: 4px solid #4CAF50; padding-left: 10px;'>{''.join(response_chunks)}</div>",
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
    st.markdown(f"👤 **You:** {chat_entry['display_query']}")
    
    response_info = f"(LLM: {chat_entry.get('llm_model', 'N/A')})"
    
    st.markdown(f"🤖 **Assistant** {response_info}:")
    st.markdown(
        f"<div style='border-left: 4px solid #4CAF50; padding-left: 10px; margin-bottom: 10px;'>{chat_entry['response']}</div>",
        unsafe_allow_html=True
    )

    st.subheader("📚 Sources:")
    with st.expander("View Sources", expanded=False):
        sources_data = chat_entry.get("sources_data", [])
        if sources_data:
            display_sources(sources_data)
        else:
            st.markdown("No sources available for this response.")
    st.markdown("---")

# Suggested queries
st.markdown("<h6>💡 Suggested Queries:</h6>", unsafe_allow_html=True)
suggested_queries_list = [
    "What are the symptoms of ADHD in adults?",
    "How is type 2 diabetes diagnosed?",
    "What are the treatment options for depression?"
]
sq_cols = st.columns(len(suggested_queries_list))
for idx, sq_text_item in enumerate(suggested_queries_list):
    if sq_cols[idx].button(
        sq_text_item, 
        key=f"suggested_{idx}", 
        disabled=st.session_state.processing_query
    ):
        st.session_state.processing_query = True
        st.session_state.query_to_run_next = sq_text_item
        st.rerun()


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

# --- Footer with Licensing Information ---
st.markdown("---")
st.caption("""
**Data Usage and Licensing:**
This tool utilizes information from NHS sources, which is made available under their respective open licensing terms.
- **NHS:** Content is used under the terms of the Open Government Licence. For full details, please refer to the [NHS Terms and Conditions](https://www.nhs.uk/our-policies/terms-and-conditions/).

Always consult the official sources for the most accurate, complete, and up-to-date information.
""")