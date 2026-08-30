import os
import json
import urllib.parse
import requests
import warnings
import streamlit as st
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from dotenv import load_dotenv

import db  # MongoDB persistence layer

# =====================================================================
# 0. Load Environment Variables & Set Hugging Face Token First
# =====================================================================
load_dotenv()

# Explicitly assign HF_TOKEN to OS environment for SentenceTransformers & HuggingFace Hub
if os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Suppress HF Token warnings if any residual notice attempts to log
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")

# =====================================================================
# 1. Page Configuration & Fancy Custom CSS
# =====================================================================
st.set_page_config(
    page_title="BiblioBot",
    # page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a glassmorphism theme, animated accents, and card styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Sora:wght@600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }

    /* ---------- App Background ---------- */
    .stApp {
        background:
            radial-gradient(circle at 15% 10%, rgba(168, 85, 247, 0.12), transparent 40%),
            radial-gradient(circle at 85% 0%, rgba(99, 102, 241, 0.12), transparent 45%),
            radial-gradient(circle at 50% 100%, rgba(14, 165, 233, 0.08), transparent 50%),
            #0d0d17;
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1020px;
    }

    /* ---------- Scrollbar ---------- */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #a855f7, #6366f1);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover { background: #c084fc; }

    /* ---------- Top Header Banner ---------- */
    .header-container {
        position: relative;
        overflow: hidden;
        background: linear-gradient(135deg, #1c1c2e 0%, #24243c 55%, #2a1f3d 100%);
        padding: 30px 34px;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow:
            0 10px 40px 0 rgba(88, 28, 135, 0.35),
            inset 0 1px 0 rgba(255, 255, 255, 0.06);
        margin-bottom: 28px;
    }

    .header-container::before {
        content: "";
        position: absolute;
        top: -60%;
        right: -10%;
        width: 340px;
        height: 340px;
        background: radial-gradient(circle, rgba(168,85,247,0.25) 0%, transparent 70%);
        pointer-events: none;
    }

    .header-title {
        font-family: 'Sora', sans-serif;
        font-size: 2.5rem !important;
        font-weight: 800;
        background: linear-gradient(90deg, #c084fc, #818cf8 45%, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
        letter-spacing: -0.02em;
    }

    .header-sub {
        color: #94a3b8;
        font-size: 0.98rem;
        margin: 0;
        font-weight: 500;
    }

    /* Metric Cards inside Header */
    .stat-badge {
        position: relative;
        background: rgba(255, 255, 255, 0.045);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 14px 22px;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, border-color 0.2s ease;
        min-width: 90px;
    }
    .stat-badge:hover {
        transform: translateY(-3px);
        border-color: rgba(168, 85, 247, 0.5);
    }
    .stat-badge .num {
        font-family: 'Sora', sans-serif;
        font-size: 1.6rem;
        font-weight: 700;
        background: linear-gradient(90deg, #f0abfc, #a5b4fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-badge .lbl {
        font-size: 0.70rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #7c8aa5;
        font-weight: 600;
        margin-top: 2px;
    }

    /* ---------- Chat Messages Styling ---------- */
    .stChatMessage {
        border-radius: 16px;
        padding: 16px 18px;
        margin-bottom: 14px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        background: rgba(255, 255, 255, 0.025);
        backdrop-filter: blur(6px);
        box-shadow: 0 4px 18px rgba(0,0,0,0.18);
        transition: border-color 0.2s ease;
    }
    .stChatMessage:hover {
        border-color: rgba(168, 85, 247, 0.25);
    }

    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, rgba(99,102,241,0.10), rgba(99,102,241,0.02));
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
        background: linear-gradient(135deg, rgba(168,85,247,0.10), rgba(14,165,233,0.03));
    }

    /* ---------- Styled Badges for In-Chat Result Formatting ---------- */
    .badge-lib {
        background: linear-gradient(90deg, #059669, #10b981);
        color: white;
        padding: 3px 10px;
        border-radius: 8px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.35);
    }
    .badge-wish {
        background: linear-gradient(90deg, #2563eb, #38bdf8);
        color: white;
        padding: 3px 10px;
        border-radius: 8px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.35);
    }

    /* Sidebar Book Cards */
    .book-card {
        display: flex;
        gap: 12px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 8px;
        margin-bottom: 8px;
        align-items: center;
    }
    .book-card img {
        border-radius: 6px;
        object-fit: cover;
    }
    .book-card-title {
        color: #f1f5f9;
        font-size: 0.88rem;
        font-weight: 700;
        margin-bottom: 2px;
    }
    .book-card-author {
        color: #94a3b8;
        font-size: 0.78rem;
    }

    /* ---------- Input Field Focus Upgrade ---------- */
    .stChatInputContainer textarea, [data-testid="stChatInput"] textarea {
        border-radius: 14px !important;
        border: 1px solid rgba(168, 85, 247, 0.25) !important;
        background: rgba(255, 255, 255, 0.03) !important;
    }
    [data-testid="stChatInput"]:focus-within textarea {
        border-color: #a855f7 !important;
        box-shadow: 0 0 0 3px rgba(168, 85, 247, 0.15) !important;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #14141f 0%, #191927 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] .stButton button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        transform: translateX(2px);
    }
    section[data-testid="stSidebar"] .stButton button[kind="primary"] {
        background: linear-gradient(90deg, #a855f7, #6366f1) !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(139, 92, 246, 0.4) !important;
    }

    section[data-testid="stSidebar"] h3 {
        font-family: 'Sora', sans-serif;
        color: #e2e8f0;
        font-size: 1rem;
    }

    section[data-testid="stSidebar"] details {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 2px 6px;
    }

    hr, section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.08) !important;
    }

    section[data-testid="stSidebar"] .stCaption, .stCaption {
        color: #64748b !important;
    }

    .stSpinner > div {
        border-top-color: #a855f7 !important;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# 2. Pipeline Backend Initialization & Cover API Helpers
# =====================================================================
def resolve_file_path(default_filename):
    """Case-insensitive path resolver ensuring exact target directory resolution."""
    for folder in ["Data", "data"]:
        if os.path.exists(folder):
            for file in os.listdir(folder):
                if file.lower() == default_filename.lower():
                    return os.path.join(folder, file)
    return os.path.join("Data", default_filename)

LIBRARY_PATH = resolve_file_path("library.csv")
WISHLIST_PATH = resolve_file_path("wishlist.csv")
GOODREADS_PATH = resolve_file_path("Goodreads.csv")
# DEFAULT_COVER = "https://via.placeholder.com/150x220?text=No+Cover"
DEFAULT_COVER = "https://covers.openlibrary.org/b/id/0-M.jpg"

def fetch_open_library_cover(title):
    """Query Open Library Search API to retrieve cover image URL."""
    try:
        encoded_title = urllib.parse.quote(title)
        url = f"https://openlibrary.org/search.json?title={encoded_title}&limit=1"
        response = requests.get(url, timeout=4)
        if response.status_code == 200:
            data = response.json()
            if data.get("docs") and "cover_i" in data["docs"][0]:
                cover_id = data["docs"][0]["cover_i"]
                return f"https://covers.openlibrary.org/b/id/{cover_id}-M.jpg"
    except Exception:
        pass
    return DEFAULT_COVER

@st.cache_resource
def initialize_rag_system():
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    if not GROQ_API_KEY:
        st.error("Error: GROQ_API_KEY not found. Please verify your .env file.")
        st.stop()

    groq_client = Groq(api_key=GROQ_API_KEY)
    chroma_client = chromadb.PersistentClient(path="./vector_store")

    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    library_col = chroma_client.get_or_create_collection(
        name="library_catalog", embedding_function=embedding_function
    )
    wishlist_col = chroma_client.get_or_create_collection(
        name="user_wishlist", embedding_function=embedding_function
    )
    goodreads_col = chroma_client.get_or_create_collection(
        name="goodreads_dataset", embedding_function=embedding_function
    )

    return groq_client, library_col, wishlist_col, goodreads_col

groq_client, library_collection, wishlist_collection, goodreads_collection = initialize_rag_system()

# Dynamic MongoDB connection verification (un-cached to allow dynamic retry)
mongo_connected = db.get_db() is not None

def ingest_source_data(csv_path, collection, source_name, fetch_covers=False, batch_size=2000):
    """Reads a CSV file source, normalizes features, and vectors items in batches to respect ChromaDB limits."""
    if not os.path.exists(csv_path):
        return 0

    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip')
    except Exception:
        try:
            df = pd.read_csv(csv_path, engine='python', on_bad_lines='skip')
        except Exception:
            return 0

    df.columns = df.columns.str.strip().str.lower()
    df = df.astype(str)
    df.replace(["nan", "None", "<NA>", "null"], "", inplace=True)
    df.fillna("", inplace=True)

    if 'cover_url' not in df.columns:
        df['cover_url'] = ""

    # Helper function to match column aliases dynamically
    def get_column_value(row, candidate_cols, default_val):
        for col in candidate_cols:
            if col in row.index and row[col].strip():
                return row[col].strip()
        return default_val

    updated_df = False
    documents, metadatas, ids = [], [], []

    for index, row in df.iterrows():
        b_title = get_column_value(row, ['title', 'book_title', 'name'], 'Unknown Title')
        b_author = get_column_value(row, ['author', 'authors', 'book_author', 'writer'], 'Unknown Author')
        b_genre = get_column_value(row, ['genre', 'genres', 'categories', 'category'], 'General')
        b_rating = get_column_value(row, ['rating', 'average_rating', 'avg_rating', 'user_rating'], 'Unrated')
        b_desc = get_column_value(row, ['description', 'summary', 'overview', 'book_description'], '')
        b_cover = get_column_value(row, ['cover_url', 'image_url', 'cover'], '')

        # Auto-fetch cover if enabled and missing
        if fetch_covers and (not b_cover or b_cover == DEFAULT_COVER or b_cover == ""):
            b_cover = fetch_open_library_cover(b_title)
            df.at[index, 'cover_url'] = b_cover
            updated_df = True

        profile = (
            f"Source: {source_name}. Title: {b_title}. Author: {b_author}. "
            f"Genre: {b_genre}. Rating: {b_rating}. Cover URL: {b_cover}. Description: {b_desc}"
        )
        documents.append(profile)
        metadatas.append({
            "title": str(b_title).lower(),
            "source": source_name,
            "genre": str(b_genre),
            "cover_url": b_cover
        })
        ids.append(f"{source_name.lower().replace(' ', '_')}_{index}")

    if updated_df:
        try:
            df.to_csv(csv_path, index=False)
        except Exception:
            pass

    if documents:
        existing = collection.get()
        if existing['ids']:
            collection.delete(ids=existing['ids'])

        # Chunk insertion into batches to stay below ChromaDB's max batch size of 5461
        total_items = len(documents)
        for i in range(0, total_items, batch_size):
            collection.add(
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
                ids=ids[i : i + batch_size]
            )

    return len(documents)

# Cached Ingestion: Executes once at startup, skipped on Streamlit re-runs
@st.cache_resource
def load_and_ingest_all_sources():
    lib_c = ingest_source_data(LIBRARY_PATH, library_collection, "Library", fetch_covers=True)
    wish_c = ingest_source_data(WISHLIST_PATH, wishlist_collection, "Wishlist", fetch_covers=True)
    good_c = ingest_source_data(GOODREADS_PATH, goodreads_collection, "Goodreads Dataset", fetch_covers=False)
    return lib_c, wish_c, good_c

lib_count, wish_count, goodreads_count = load_and_ingest_all_sources()

# =====================================================================
# 3. Session State & URL Parameter Hydration
# =====================================================================
query_params = st.query_params
if "session_id" in query_params:
    st.session_state.session_id = query_params["session_id"]
elif "session_id" not in st.session_state:
    st.session_state.session_id = db.new_session_id()
    st.query_params["session_id"] = st.session_state.session_id

def switch_session(target_session_id):
    st.session_state.session_id = target_session_id
    st.query_params["session_id"] = target_session_id
    if "messages" in st.session_state:
        del st.session_state["messages"]
    st.rerun()

# =====================================================================
# 4. Sidebar Navigation & Dynamic Chat History
# =====================================================================
def render_sidebar_catalog(csv_path):
    if not os.path.exists(csv_path):
        st.info("No items found.")
        return

    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip')
    except Exception:
        st.info("No items found.")
        return

    if df.empty:
        st.info("No items found.")
        return

    df.columns = df.columns.str.strip().str.lower()
    for _, row in df.iterrows():
        title = row.get('title', 'Unknown')
        author = row.get('author', 'Unknown')
        cover = row.get('cover_url', DEFAULT_COVER)
        if not cover or str(cover).lower() == "nan":
            cover = DEFAULT_COVER

        st.markdown(
            f"""
            <div class="book-card">
                <img src="{cover}" width="45" height="65" alt="Cover">
                <div>
                    <div class="book-card-title">{title}</div>
                    <div class="book-card-author">by {author}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

with st.sidebar:
    st.markdown(
        "<div style='font-family:Sora,sans-serif;font-weight:800;font-size:1.4rem;"
        "background:linear-gradient(90deg,#c084fc,#818cf8);-webkit-background-clip:text;"
        "-webkit-text-fill-color:transparent;'>BiblioBot</div>",
        unsafe_allow_html=True
    )
    # st.caption("Multisource RAG Engine v3.0")
    st.markdown("---")

    if st.button("➕ Start New Chat", use_container_width=True, type="primary"):
        switch_session(db.new_session_id())

    st.markdown("### 💬 Chat History")
    if mongo_connected:
        recent_sessions = db.list_recent_sessions(limit=15)
        
        active_history_found = False
        for s in recent_sessions:
            s_id = s["_id"]
            history = db.get_chat_history(s_id)
            user_msgs = [m["content"] for m in history if m["role"] == "user"]
            
            if not user_msgs:
                continue

            active_history_found = True
            preview_title = user_msgs[0]
            if len(preview_title) > 28:
                preview_title = preview_title[:25] + "..."

            is_active = (s_id == st.session_state.session_id)
            btn_label = f"💬 {preview_title}" if not is_active else f"👉 **{preview_title}**"
            btn_type = "secondary" if not is_active else "primary"

            if st.button(btn_label, key=f"sess_{s_id}", use_container_width=True, type=btn_type):
                if not is_active:
                    switch_session(s_id)

        if not active_history_found:
            st.caption("No chat history yet. Send a message to start!")
    else:
        st.caption("MongoDB disconnected. History unavailable.")

    st.markdown("---")
    st.markdown("### 🗂️ Catalog Manager")
    # st.write("Explore your synced vector source files below:")

    with st.expander("📖 Library Holdings"):
        render_sidebar_catalog(LIBRARY_PATH)

    with st.expander("✨ Wishlist Items"):
        render_sidebar_catalog(WISHLIST_PATH)

    st.markdown("---")
    st.caption("Powered by Groq • ChromaDB • MongoDB")

# =====================================================================
# 5. Automated Book Generation & CRUD Operations
# =====================================================================
def autofill_book_metadata(title):
    prompt = f"""
    Provide core library catalog meta attributes for the book title: "{title}".
    You must output strictly a valid JSON object matching this structure:
    {{
        "title": "Exact Book Title",
        "author": "Author Name",
        "genre": "Main Genre classification",
        "rating": "A standard historical review rating between 1.0 and 5.0",
        "description": "A robust 2-sentence summary detailing plot elements and themes."
    }}
    Do not add text before or after the JSON payload string.
    """
    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        res_text = completion.choices[0].message.content.strip()
        meta = json.loads(res_text)
    except Exception:
        meta = {"title": title, "author": "Unknown", "genre": "General", "rating": "3.5", "description": "Auto-generated placeholder record."}

    # Fetch Open Library Cover URL
    meta["cover_url"] = fetch_open_library_cover(meta.get("title", title))
    return meta

def modify_csv_database(action, target_db, book_title, session_id=None):
    target_clean = str(target_db).lower().strip() if target_db else ""

    # Flexible matching logic so 'library', 'Library', or 'library_catalog' targets the library
    if "lib" in target_clean:
        csv_file = LIBRARY_PATH
        collection = library_collection
        source_label = "Library"
    else:
        csv_file = WISHLIST_PATH
        collection = wishlist_collection
        source_label = "Wishlist"

    data_dir = os.path.dirname(csv_file) or "Data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    required_cols = ['title', 'author', 'genre', 'rating', 'description', 'cover_url']

    if not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0:
        pd.DataFrame(columns=required_cols).to_csv(csv_file, index=False)

    try:
        df = pd.read_csv(csv_file, on_bad_lines='skip')
    except Exception:
        df = pd.DataFrame(columns=required_cols)

    df.columns = df.columns.str.strip().str.lower()

    # Ensure all target columns exist before processing
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    df = df.astype(str)
    df.replace(["nan", "None", "<NA>"], "", inplace=True)
    df.fillna("", inplace=True)

    result_message = None

    if action == "add":
        with st.spinner(f"✨ Auto-generating metadata & fetching Open Library cover for **'{book_title}'**..."):
            meta = autofill_book_metadata(book_title)

        df = df[df['title'].str.lower() != book_title.lower()]
        new_row = pd.DataFrame([meta])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(csv_file, index=False)
        ingest_source_data(csv_file, collection, source_label, fetch_covers=True)
        
        badge_style = "badge-lib" if "lib" in target_clean else "badge-wish"
        cover_img = meta.get('cover_url', DEFAULT_COVER)

        result_message = (
            f"✅ **Added to <span class='{badge_style}'>{source_label}</span>**\n\n"
            f"<div style='display:flex; gap:16px; margin-top:8px; align-items:flex-start;'>"
            f"<img src='{cover_img}' width='100' style='border-radius:10px; box-shadow:0 4px 12px rgba(0,0,0,0.3);'>"
            f"<div>"
            f"**Title:** {meta['title']}  \n"
            f"**Author:** {meta['author']} | **Genre:** {meta['genre']} | **Rating:** ⭐ {meta['rating']}  \n\n"
            f"_{meta['description']}_"
            f"</div>"
            f"</div>"
        )

    elif action == "remove":
        initial_count = len(df)
        df = df[df['title'].str.lower() != book_title.lower()]
        if len(df) == initial_count:
            result_message = f"❌ Could not locate **'{book_title}'** in your **{source_label}** records."
        else:
            df.to_csv(csv_file, index=False)
            ingest_source_data(csv_file, collection, source_label, fetch_covers=False)
            result_message = f"🗑️ Successfully removed **'{book_title}'** from your **{source_label}** records."

    if session_id:
        db.log_db_mutation(session_id, action, target_db, book_title, result_message)

    # Invalidate Streamlit cache to update item counters
    st.cache_resource.clear()

    return result_message

def get_all_items_summary(csv_path, source_name):
    if not os.path.exists(csv_path):
        return f"No items in {source_name}."
    
    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip')
    except Exception:
        return f"Unable to read items in {source_name}."

    if df.empty:
        return f"Your {source_name} is currently empty."
    
    df.columns = df.columns.str.strip().str.lower()
    items = []
    for _, row in df.iterrows():
        title = row.get('title', 'Unknown')
        author = row.get('author', 'Unknown')
        cover = row.get('cover_url', '')
        cover_img = f"<img src='{cover}' width='40' style='vertical-align:middle; border-radius:4px; margin-right:8px;'>" if cover else ""
        items.append(f"{cover_img}• **{title}** by *{author}*")

    return f"**Items in {source_name} ({len(items)} total):**\n\n" + "\n\n".join(items)

# =====================================================================
# 6. Multisource Pipeline Execution Logic
# =====================================================================
def run_multisource_pipeline(user_query, session_id=None, chat_history=None):
    query_lower = user_query.lower()

    # 1. Direct Catalog Listing Interceptor
    if any(phrase in query_lower for phrase in ["all books", "list books", "show my library", "show wishlist", "my books", "show catalog", "books in my library"]):
        if "wishlist" in query_lower and "library" not in query_lower:
            return get_all_items_summary(WISHLIST_PATH, "Wishlist")
        elif "library" in query_lower and "wishlist" not in query_lower:
            return get_all_items_summary(LIBRARY_PATH, "Library")
        else:
            lib_text = get_all_items_summary(LIBRARY_PATH, "Library")
            wish_text = get_all_items_summary(WISHLIST_PATH, "Wishlist")
            return f"{lib_text}\n\n---\n\n{wish_text}"

    # 2. Scope & Intent Classification Gate
    classifier_prompt = f"""
    Analyze the following user query and classify its intent and domain scope.
    
    User Query: "{user_query}"
    
    Determine:
    1. "is_book_related": Set to true IF the user is asking about books, reading, authors, literature, library management, wishlist management, or requesting book recommendations (even if asking for a book related to music, songs, movies, history, etc.), OR asking follow-up questions referencing previous book interactions. Set to false IF the query asks to perform non-literary actions.
    2. "action": "add", "remove", or null.
    3. "target": "library", "wishlist", or null.
    4. "book_title": Extracted book title if action is add/remove, otherwise null.

    Output strictly a JSON object matching:
    {{
        "is_book_related": true or false,
        "action": "add" or "remove" or null,
        "target": "library" or "wishlist" or null,
        "book_title": "Cleaned Book Title" or null
    }}
    Do not add any text or markdown formatting outside the raw JSON object.
    """

    try:
        classifier_completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": classifier_prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )

        classified_data = json.loads(classifier_completion.choices[0].message.content.strip())
        
        if not classified_data.get("is_book_related", True):
            return "⚠️ **Out of Scope:** I am **BiblioBot**, a library assistant dedicated exclusively to book recommendations, catalog queries, and reading list management. I cannot play audio, play songs, or answer non-literary requests."

        action = classified_data.get("action")
        target = classified_data.get("target")
        book_title = classified_data.get("book_title")

        if action in ["add", "remove"] and target in ["library", "wishlist"] and book_title:
            return modify_csv_database(action, target, book_title, session_id=session_id)

    except Exception:
        pass

    # 3. Standard RAG Search Workflow across all 3 Collections
    lib_context = [doc for sublist in library_collection.query(query_texts=[user_query], n_results=5)['documents'] for doc in sublist]
    wish_context = [doc for sublist in wishlist_collection.query(query_texts=[user_query], n_results=5)['documents'] for doc in sublist]
    good_context = [doc for sublist in goodreads_collection.query(query_texts=[user_query], n_results=5)['documents'] for doc in sublist]

    combined_context = "\n".join(lib_context + wish_context + good_context)

    system_instruction = f"""
You are "BiblioBot", an intelligent Retrieval-Augmented Multisource Conversational Recommender System specializing strictly in books, literature, and library collections.

### RETRIEVED MULTISOURCE CONTEXT:
{combined_context if combined_context else "No direct matches located in local database streams."}

### OPERATIONAL MANDATES:
1. STRICT DOMAIN BOUNDARY: You MUST ONLY answer requests regarding books, literary recommendations, reading, and library catalog management. If a user asks you to perform non-book actions, politely explain that you are a book recommendation system.
2. SOURCE DISTINCTION: When recommending or referencing items, explicitly mention if a book is currently in the user's **Library**, **Wishlist**, or from the **Goodreads Dataset**.
3. SHOW COVER IMAGES: Whenever you recommend or mention a specific book title in your response, embed its cover image using HTML image formatting: `<img src="COVER_URL" width="90" style="border-radius:8px; float:left; margin-right:12px;">`. Extract the `Cover URL:` parameter from the retrieved context above if available.
4. GENERAL LITERARY KNOWLEDGE: If a query cannot be answered from local context, rely on global book knowledge to provide recommendations.
5. CONVERSATIONAL MEMORY: Maintain awareness of previous messages in this conversation thread and seamlessly handle context, follow-ups, and relative queries.
6. BREVITY: Keep your answer concise (3-4 sentences maximum per recommendation).
"""

    messages_payload = [{"role": "system", "content": system_instruction}]

    if chat_history:
        recent_history = [
            {"role": m["role"], "content": m["content"]}
            for m in chat_history[-10:]
            if m["role"] in ["user", "assistant"]
        ]
        messages_payload.extend(recent_history)
    else:
        messages_payload.append({"role": "user", "content": user_query})

    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages_payload,
            temperature=0.3,
            max_tokens=1024
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"System Connection Error: {str(e)}"

# =====================================================================
# 7. Streamlit Interface Render Loop & Persistent Hydration
# =====================================================================

header_html = f"""
<div class="header-container">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; position: relative; z-index: 1;">
        <div>
            <div class="header-title">BiblioBot Workspace</div>
            <div class="header-sub">Conversational AI Recommender &amp; Library Management Engine</div>
        </div>
        <div style="display: flex; gap: 12px; margin-top: 10px;">
            <div class="stat-badge">
                <div class="num">{lib_count or 0}</div>
                <div class="lbl">Library</div>
            </div>
            <div class="stat-badge">
                <div class="num">{wish_count or 0}</div>
                <div class="lbl">Wishlist</div>
            </div>
        </div>
    </div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

if "messages" not in st.session_state:
    saved_history = db.get_chat_history(st.session_state.session_id) if mongo_connected else None
    
    if saved_history:
        st.session_state.messages = [
            {"role": doc["role"], "content": doc["content"]}
            for doc in saved_history
        ]
    else:
        welcome_text = "Hello! I am monitoring your collections. Ask for book recommendations or issue commands like *'Add Dune to my wishlist'*."
        st.session_state.messages = [{"role": "assistant", "content": welcome_text}]

for msg in st.session_state.messages:
    avatar = "🤖" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"], unsafe_allow_html=True)

if user_input := st.chat_input("Ask BiblioBot or give a command (e.g., 'Add 1984 to my library')..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    if mongo_connected:
        db.log_message(st.session_state.session_id, "user", user_input)

    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🤖"):
        response = run_multisource_pipeline(
            user_input, 
            session_id=st.session_state.session_id, 
            chat_history=st.session_state.messages
        )
        st.markdown(response, unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": response})
    
    if mongo_connected:
        db.log_message(st.session_state.session_id, "assistant", response)
    
    st.rerun()