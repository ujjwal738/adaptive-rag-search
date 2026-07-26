import streamlit as st
import requests
import uuid

# Premium UI Configuration
st.set_page_config(
    page_title="Adaptive RAG Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #a0aec0;
        margin-bottom: 2rem;
    }
    
    .card {
        background-color: #1a202c;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #2d3748;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

API_BASE_URL = "http://localhost:8000"

st.markdown('<h1 class="main-title">Adaptive RAG Search</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Intelligent Document Retrieval and Cognitive Search Pipeline</p>', unsafe_allow_html=True)

# Sidebar layout for document uploading & analytics
with st.sidebar:
    st.header("🗂️ Document Management")
    st.write("Upload knowledge documents (PDF or TXT) to index into the FAISS vector database.")
    
    description = st.text_input("Document Description", placeholder="e.g., Q2 Financial Report, Python Developer Guide")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])
    
    if st.button("🚀 Upload & Index", use_container_width=True):
        if not description:
            st.error("Please provide a document description.")
        elif not uploaded_file:
            st.error("Please choose a file to upload.")
        else:
            with st.spinner("Processing document chunks and building vector store..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    headers = {"X-Description": description}
                    res = requests.post(f"{API_BASE_URL}/rag/documents/upload", files=files, headers=headers)
                    if res.status_code == 200 and res.json().get("status") is True:
                        st.success("🎉 Document successfully indexed!")
                    else:
                        st.error(f"Failed to index: {res.text}")
                except Exception as e:
                    st.error(f"Error connecting to backend: {e}")

    st.markdown("---")
    st.header("📊 RAG Performance Analytics")
    try:
        stats_res = requests.get(f"{API_BASE_URL}/rag/feedback/stats")
        if stats_res.status_code == 200:
            stats = stats_res.json()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Ratings", stats.get("total_feedback_count", 0))
            with col2:
                st.metric("Avg Rating", f"{stats.get('average_rating', 0.0)} ⭐")
            
            st.caption("Route Breakdown:")
            for r_name, r_info in stats.get("route_breakdown", {}).items():
                st.write(f"- **{r_name}**: {r_info.get('count')} ratings (avg: {r_info.get('average_rating')} ⭐)")
    except Exception:
        st.info("Backend analytics offline or unavailable.")

# Chat interface
st.subheader("💬 Chat Assistant")

# Display previous chat history
for idx, item in enumerate(st.session_state["chat_history"]):
    role = item["role"]
    text = item["text"]
    with st.chat_message(role):
        st.write(text)
        if role == "assistant":
            route = item.get("route", "general")
            query = item.get("user_query", "")
            with st.expander("⭐ Rate this response", expanded=False):
                with st.form(key=f"feedback_form_{idx}"):
                    rating = st.slider("Rating (1 = Poor, 5 = Excellent)", 1, 5, 5, key=f"slider_{idx}")
                    comment = st.text_input("Optional feedback", key=f"comment_{idx}")
                    submitted = st.form_submit_button("Submit Feedback")
                    if submitted:
                        try:
                            fb_payload = {
                                "session_id": st.session_state["session_id"],
                                "query": query,
                                "rating": rating,
                                "feedback_text": comment,
                                "route": route
                            }
                            fb_res = requests.post(f"{API_BASE_URL}/rag/feedback", json=fb_payload)
                            if fb_res.status_code == 200:
                                st.success("Thank you for your feedback!")
                            else:
                                st.error("Failed to submit feedback.")
                        except Exception as e:
                            st.error(f"Error submitting feedback: {e}")

# Input for new message
if user_query := st.chat_input("Ask a question about the uploaded documents..."):
    # Show user message
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state["chat_history"].append({"role": "user", "text": user_query})
    
    # Generate assistant message
    with st.chat_message("assistant"):
        with st.spinner("Running cognitive graph routing & retrieval..."):
            try:
                payload = {
                    "query": user_query,
                    "session_id": st.session_state["session_id"]
                }
                res = requests.post(f"{API_BASE_URL}/rag/query", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    ans = data["result"]["content"]
                    route = data.get("route", "general")
                    st.write(ans)
                    st.session_state["chat_history"].append({
                        "role": "assistant",
                        "text": ans,
                        "user_query": user_query,
                        "route": route
                    })
                    st.rerun()
                else:
                    st.error(f"Error {res.status_code}: {res.text}")
            except Exception as e:
                st.error(f"Connection to RAG API failed: {e}")

