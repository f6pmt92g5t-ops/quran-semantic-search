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

    embeddings = np.load(
        "verse_embeddings.npy"
    )

    return verses, embeddings



verses_df, verse_embeddings = load_data()



# =========================
# Normalize Arabic
# =========================

def normalize_arabic(text):

    text = str(text)


    text = re.sub(
        r"[\u064B-\u065F\u0670\u06D6-\u06ED]",
        "",
        text
    )


    text = re.sub(
        "[إأآٱ]",
        "ا",
        text
    )


    text = text.replace(
        "ـ",
        ""
    )


    text = re.sub(
        r"\s+",
        " ",
        text
    )


    return text.strip()



# =========================
# Root Cleaning
# =========================

def get_root(word):

    word = normalize_arabic(word)


    word = re.sub(
        r"[^ء-ي]",
        "",
        word
    )


    prefixes = [
        "وال",
        "بال",
        "فال",
        "كال",
        "لل",
        "ال",
        "و",
        "ب"
    ]


    changed = True


    while changed:

        changed = False


        for p in prefixes:

            if word.startswith(p) and len(word) > len(p)+2:

                word = word[len(p):]

                changed = True

                break


    return word
# =========================
# Hybrid Semantic Search
# =========================

def semantic_search(query, top_k=100):


    query_clean = normalize_arabic(query)

    query_root = get_root(query)



    # Query embedding

    query_embedding = model.encode(
        query_root
    )



    # Cosine similarity

    semantic_scores = util.cos_sim(
        query_embedding,
        verse_embeddings
    )[0].cpu().numpy()



    results = []



    for i, semantic_score in enumerate(semantic_scores):


        verse = normalize_arabic(
            verses_df.iloc[i]["clean_text"]
            if "clean_text" in verses_df.columns
            else verses_df.iloc[i]["text"]
        )



        words = [
            get_root(w)
            for w in verse.split()
        ]



        # =====================
        # Matching Level
        # =====================

        match_type = 0



        # العبارة كاملة

        if query_clean in verse:

            match_type = 3



        # نفس الكلمة بعد إزالة السوابق

        elif query_root in words:

            match_type = 2



        # مشتقات مثل النجوم

        elif any(
            query_root in w
            for w in words
        ):

            match_type = 1



        # =====================
        # Final Ranking
        # =====================

        final_score = (

            (0.5 * match_type)

            +

            (0.5 * float(semantic_score))

        )



        results.append({

            "score": final_score,

            "semantic": float(semantic_score),

            "match": match_type,

            "sura": verses_df.iloc[i]["sura"],

            "aya": verses_df.iloc[i]["aya"],

            "text": verses_df.iloc[i]["text"]

        })



    # ترتيب النتائج

    results = sorted(

        results,

        key=lambda x: x["score"],

        reverse=True

    )



    return results[:top_k]



# =========================
# Remove duplicates
# =========================

def remove_duplicates(results):


    output = []

    seen = set()



    for r in results:


        key = (

            r["sura"],

            r["aya"]

        )


        if key not in seen:

            output.append(r)

            seen.add(key)



    return output
# =========================
# Streamlit Interface
# =========================


st.title(
    "📖 Quran Semantic Search"
)


st.write(
    "البحث الدلالي في القرآن الكريم"
)



query = st.text_input(
    "اكتب كلمة أو جملة للبحث:"
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
# Display Results
# =========================

results = st.session_state.results



if results:


    st.subheader(
        "🔎 نتائج البحث"
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



    for index, r in enumerate(
        current_results,
        start=start + 1
    ):


        st.markdown(
            f"### {index}"
        )


        st.write(
            f"**سورة:** {r['sura']} | **آية:** {r['aya']}"
        )


        st.write(
            r["text"]
        )


        # تحويل الدرجة إلى نسبة تقريبية

        similarity = max(
            0,
            min(
                100,
                int(r["score"] * 100)
            )
        )


        st.write(
            f"درجة التشابه: {similarity}%"
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
            f"الصفحة {page} من {total_pages}"
        )



    with col3:

        if st.button("التالي ➡"):

            if page < total_pages:

                st.session_state.page += 1

                st.rerun()



else:


    st.info(
        "ابدأ بكتابة كلمة للبحث"
    )
