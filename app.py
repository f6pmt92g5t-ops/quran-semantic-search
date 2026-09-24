import streamlit as st
import pandas as pd
import numpy as np
import re

from sentence_transformers import SentenceTransformer, util


# =========================
# Page Config
# =========================

st.set_page_config(
    page_title="Quran Semantic Search",
    layout="wide"
)



# =========================
# Load Model
# =========================

@st.cache_resource
def load_model():

    return SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2"
    )



model = load_model()



# =========================
# Load Data
# =========================

@st.cache_data
def load_data():

    verses = pd.read_csv(
        "verses.csv"
    )

    segments = pd.read_csv(
        "segments.csv"
    )

    embeddings = np.load(
        "segment_embeddings.npy"
    )

    return verses, segments, embeddings



verses_df, segments_df, embeddings = load_data()



# =========================
# Arabic Normalize
# =========================

def normalize_arabic(text):

    text = str(text)


    # Remove tashkeel

    text = re.sub(
        r"[\u064B-\u065F\u0670\u06D6-\u06ED]",
        "",
        text
    )


    # Normalize letters

    text = re.sub(
        "[إأآٱ]",
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
# Word Cleaning
# =========================

def clean_word(word):

    word = normalize_arabic(word)


    prefixes = [
        "وال",
        "بال",
        "ال",
        "و",
        "ب"
    ]


    for prefix in prefixes:

        if word.startswith(prefix) and len(word) > len(prefix)+2:

            word = word[len(prefix):]

            break


    return word
  # =========================
# Semantic Search Engine
# =========================

def semantic_search(query, top_k=100):

    query_clean = normalize_arabic(query)


    # Query embedding

    query_embedding = model.encode(
        query_clean
    )


    # Semantic similarity

    semantic_scores = util.cos_sim(
        query_embedding,
        embeddings
    )[0].cpu().numpy()



    # Query words

    query_words = set(
        clean_word(w)
        for w in query_clean.split()
        if len(w) > 1
    )



    final_scores = []



    for idx, semantic_score in enumerate(semantic_scores):


        text = normalize_arabic(
            segments_df.iloc[idx]["text"]
        )


        text_words = set(
            clean_word(w)
            for w in text.split()
            if len(w) > 1
        )


        # الكلمات المشتركة

        common = len(
            query_words.intersection(
                text_words
            )
        )


        word_score = (
            common /
            max(len(query_words), 1)
        )



        # رفع أولوية الكلمات الموجودة

        if common > 0:

            word_score += 0.4



        if word_score > 1:

            word_score = 1



        # الدرجة النهائية

        final_score = (

            (0.65 * semantic_score)

            +

            (0.35 * word_score)

        )


        final_scores.append(
            final_score
        )



    final_scores = np.array(
        final_scores
    )



    # ترتيب النتائج

    best_indices = np.argsort(
        final_scores
    )[::-1][:top_k]



    results = []



    for idx in best_indices:


        row = segments_df.iloc[idx]


        results.append({

            "sura": row["sura"],

            "aya": row["aya"],

            "text": row["text"],

            "score": float(
                final_scores[idx]
            )

        })


    return results





# =========================
# Remove Duplicates
# =========================

def remove_duplicates(results):

    final = []

    seen = set()


    for r in results:


        key = (
            r["sura"],
            r["aya"],
            r["text"]
        )


        if key not in seen:

            final.append(r)

            seen.add(key)



    return final
  # =========================
# Streamlit Interface
# =========================

st.title(
    "📖 Quran Semantic Search"
)


st.write(
    "Semantic Search for Quran verses"
)



query = st.text_input(
    "اكتب البحث:"
)



if "results" not in st.session_state:

    st.session_state.results = []



if "page" not in st.session_state:

    st.session_state.page = 1




# =========================
# Search Button
# =========================

if st.button("بحث"):


    if query.strip():


        results = semantic_search(
            query
        )


        results = remove_duplicates(
            results
        )


        st.session_state.results = results

        st.session_state.page = 1



# =========================
# Show Results
# =========================

results = st.session_state.results



if results:


    st.subheader(
        "🔎 نتائج البحث الدلالي"
    )


    page_size = 10


    total_pages = max(
        1,
        (len(results) + page_size - 1)
        // page_size
    )



    page = st.session_state.page



    start = (
        page - 1
    ) * page_size



    end = start + page_size



    current_results = results[start:end]



    for index, result in enumerate(
        current_results,
        start=start + 1
    ):


        st.markdown(
            f"### {index}"
        )


        st.write(
            f"**السورة:** {result['sura']}  |  **الآية:** {result['aya']}"
        )


        st.write(
            result["text"]
        )


        st.write(
            f"درجة التشابه: {result['score']:.3f}"
        )


        st.divider()



    # =========================
    # Pagination
    # =========================

    col1, col2, col3 = st.columns(3)



    with col1:

        if st.button("⬅ السابق"):

            if page > 1:

                st.session_state.page -= 1

                st.rerun()



    with col2:

        st.write(
            f"صفحة {page} من {total_pages}"
        )



    with col3:

        if st.button("التالي ➡"):

            if page < total_pages:

                st.session_state.page += 1

                st.rerun()



else:


    st.info(
        "اكتب كلمة أو جملة للبحث"
    )

  
