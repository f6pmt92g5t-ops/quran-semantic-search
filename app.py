import streamlit as st
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer, util


# =========================
# Page Settings
# =========================

st.set_page_config(
    page_title="Quran Semantic Search",
    layout="wide"
)


# =========================
# Load Model & Data
# =========================

@st.cache_resource
def load_model():
    return SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2"
    )


@st.cache_data
def load_data():

    segments = pd.read_csv(
        "segments.csv"
    )

    verses = pd.read_csv(
        "verses.csv"
    )

    embeddings = np.load(
        "segment_embeddings.npy"
    )

    return segments, verses, embeddings


model = load_model()

segments_df, verses_df, segment_embeddings = load_data()



# =========================
# Arabic Normalization
# =========================

def normalize_arabic(text):

    text = str(text)

    # Remove tashkeel
    text = re.sub(
        r"[\u064B-\u065F\u0670\u06D6-\u06ED]",
        "",
        text
    )

    # Normalize Alef
    text = re.sub(
        r"[إأآٱ]",
        "ا",
        text
    )

    # Remove tatweel
    text = text.replace(
        "ـ",
        ""
    )

    # Spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()



# =========================
# Prepare Exact Search
# =========================

@st.cache_data
def prepare_verses():

    df = verses_df.copy()

    df["clean_text"] = df["text"].apply(
        normalize_arabic
    )

    return df


verses_clean = prepare_verses()



# =========================
# Exact Search
# =========================

def exact_search(query, limit=10):

    q = normalize_arabic(query)

    results = []


    for _, row in verses_clean.iterrows():

        text = row["clean_text"]


        if q in text:

            results.append({

                "type": "Exact Match",
                "sura": row["sura"],
                "aya": row["aya"],
                "text": row["text"],
                "score": 1.0

            })


            if len(results) >= limit:
                break


    return results



# =========================
# Improved Semantic Search
# =========================

def semantic_search(query, top_k=50):

    q = normalize_arabic(query)


    query_embedding = model.encode(
        q
    )


    semantic_scores = util.cos_sim(
        query_embedding,
        segment_embeddings
    )[0].cpu().numpy()



    # Keyword relevance
    query_words = set(
        q.split()
    )


    keyword_scores = []


    for _, row in segments_df.iterrows():

        text = normalize_arabic(
            row["text"]
        )

        text_words = set(
            text.split()
        )


        common = len(
            query_words.intersection(text_words)
        )


        keyword_scores.append(
            common / max(len(query_words), 1)
        )


    keyword_scores = np.array(
        keyword_scores
    )


    # Combined score
    final_scores = (
        0.7 * semantic_scores
        +
        0.3 * keyword_scores
    )


    indices = np.argsort(
        final_scores
    )[::-1][:top_k]


    results = []


    for idx in indices:

        row = segments_df.iloc[idx]


        results.append({

            "type": "Semantic Match",
            "sura": row["sura"],
            "aya": row["aya"],
            "text": row["text"],
            "score": float(final_scores[idx])

        })


    return results



# =========================
# Hybrid Search
# =========================

def hybrid_search(query):

    exact_results = exact_search(
        query
    )


    semantic_results = semantic_search(
        query
    )


    final = []

    seen = set()


    for item in exact_results + semantic_results:

        key = (
            item["sura"],
            item["aya"],
            item["text"]
        )


        if key not in seen:

            final.append(item)
            seen.add(key)


    return final
  # =========================
# Streamlit Interface
# =========================

st.title("📖 Quran Semantic Search")

st.write(
    "Hybrid Search: Exact Match + Semantic Search"
)


query = st.text_input(
    "Enter your search query:"
)



# =========================
# Pagination
# =========================

def paginate(results, page, page_size=10):

    start = (page - 1) * page_size
    end = start + page_size

    return results[start:end]



# =========================
# Search Button
# =========================

if st.button("Search"):

    if query.strip():

        results = hybrid_search(query)


        # Separate results

        exact_results = [
            r for r in results
            if r["type"] == "Exact Match"
        ]


        semantic_results = [
            r for r in results
            if r["type"] == "Semantic Match"
        ]


        # Save in session

        st.session_state["exact_results"] = exact_results
        st.session_state["semantic_results"] = semantic_results
        st.session_state["page"] = 1



# =========================
# Display Exact Results
# =========================

if "exact_results" in st.session_state:


    exact_results = st.session_state["exact_results"]
    semantic_results = st.session_state["semantic_results"]


    if len(exact_results) > 0:

        st.subheader(
            "📌 Exact Matches"
        )


        for r in exact_results:

            st.success(
                f"Sura {r['sura']} - Ayah {r['aya']}"
            )

            st.write(
                r["text"]
            )

            st.write(
                "Match Score: 1.000"
            )

            st.divider()



    else:

        st.info(
            "No exact matches found"
        )



    # =========================
    # Semantic Results
    # =========================


    st.subheader(
        "🧠 Semantic Similar Results"
    )


    page = st.session_state.get(
        "page",
        1
    )


    page_results = paginate(
        semantic_results,
        page,
        10
    )


    for i, r in enumerate(
        page_results,
        start=1
    ):


        st.info(
            f"Rank {((page-1)*10)+i}"
        )


        st.write(
            f"Sura {r['sura']} - Ayah {r['aya']}"
        )


        st.write(
            r["text"]
        )


        st.write(
            f"Semantic Score: {r['score']:.3f}"
        )


        st.divider()



    # =========================
    # Page Buttons
    # =========================


    total_pages = max(
        1,
        (len(semantic_results) + 9) // 10
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        if st.button(
            "⬅ Previous"
        ):

            if page > 1:

                st.session_state["page"] -= 1



    with col2:

        st.write(
            f"Page {page} / {total_pages}"
        )



    with col3:

        if st.button(
            "Next ➡"
        ):

            if page < total_pages:

                st.session_state["page"] += 1



else:

    st.caption(
        "Enter a Quranic query to search"
    )
