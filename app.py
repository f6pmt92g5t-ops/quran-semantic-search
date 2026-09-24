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
# Load Files
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
# Arabic Normalization
# =========================

def normalize_arabic(text):

    text = str(text)


    # remove tashkeel
    text = re.sub(
        r"[\u064B-\u065F\u0670\u06D6-\u06ED]",
        "",
        text
    )


    # normalize letters

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
# Prepare Quran Text
# =========================

@st.cache_data
def prepare_data():

    df = verses_df.copy()


    df["clean_text"] = df["text"].apply(
        normalize_arabic
    )


    return df



verses_clean = prepare_data()



# =========================
# Arabic Word Similar Search
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


    for p in prefixes:

        if word.startswith(p) and len(word) > len(p)+2:

            word = word[len(p):]

            break


    return word



def word_search(query, limit=10):

    query_word = clean_word(query)

    results = []


    for _, row in verses_clean.iterrows():

        words = row["clean_text"].split()


        best_score = 0


        for w in words:

            w = clean_word(w)


            if w == query_word:

                best_score = 1.0
                break


            elif (
                query_word in w
                or
                w in query_word
            ):

                best_score = max(
                    best_score,
                    0.8
                )


        if best_score > 0:

            results.append({

                "type": "Word Match",
                "sura": row["sura"],
                "aya": row["aya"],
                "text": row["text"],
                "score": best_score

            })


    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )


    return results[:limit]
# =========================
# Semantic Search
# =========================

def semantic_search(query, top_k=50):

    query_clean = normalize_arabic(query)


    query_embedding = model.encode(
        query_clean
    )


    scores = util.cos_sim(
        query_embedding,
        embeddings
    )[0].cpu().numpy()



    # Keyword bonus
    query_words = set(
        query_clean.split()
    )


    final_scores = []


    for idx, score in enumerate(scores):

        text = normalize_arabic(
            segments_df.iloc[idx]["text"]
        )


        text_words = set(
            text.split()
        )


        common = len(
            query_words.intersection(text_words)
        )


        keyword_bonus = (
            common /
            max(len(query_words), 1)
        )


        final_score = (
            (0.75 * score)
            +
            (0.25 * keyword_bonus)
        )


        final_scores.append(
            final_score
        )


    final_scores = np.array(
        final_scores
    )


    best_indices = np.argsort(
        final_scores
    )[::-1][:top_k]



    results = []


    for idx in best_indices:

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


    word_results = word_search(
        query
    )


    semantic_results = semantic_search(
        query
    )


    results = []

    seen = set()



    # Word results first

    for item in word_results + semantic_results:


        key = (
            item["sura"],
            item["aya"],
            item["text"]
        )


        if key not in seen:

            results.append(item)

            seen.add(key)



    return results
# =========================
# Streamlit Interface
# =========================

st.title("📖 Quran Semantic Search")

st.write(
    "Hybrid Search: Word Match + Semantic Search"
)


query = st.text_input(
    "اكتب البحث:"
)



# حفظ النتائج
if "results" not in st.session_state:

    st.session_state.results = []



if "page" not in st.session_state:

    st.session_state.page = 1



# =========================
# Search Button
# =========================

if st.button("بحث"):


    if query.strip():

        st.session_state.results = hybrid_search(
            query
        )

        st.session_state.page = 1



# =========================
# Display Results
# =========================

results = st.session_state.results



if results:


    word_results = [
        r for r in results
        if r["type"] == "Word Match"
    ]


    semantic_results = [
        r for r in results
        if r["type"] == "Semantic Match"
    ]



    # =========================
    # Word Matches
    # =========================

    st.subheader(
        "📌 نتائج مطابقة الكلمات"
    )


    if word_results:


        for r in word_results:


            st.success(
                f"سورة {r['sura']} - آية {r['aya']}"
            )


            st.write(
                r["text"]
            )


            st.write(
                f"درجة التطابق: {r['score']:.2f}"
            )


            st.divider()


    else:

        st.info(
            "لا توجد نتائج مطابقة للكلمة"
        )




    # =========================
    # Semantic Matches
    # =========================

    st.subheader(
        "🧠 نتائج مشابهة بالمعنى"
    )


    page_size = 10


    total_pages = max(
        1,
        (len(semantic_results) + page_size - 1)
        // page_size
    )



    page = st.session_state.page



    start = (
        page - 1
    ) * page_size


    end = start + page_size



    current_results = semantic_results[
        start:end
    ]



    for index, r in enumerate(
        current_results,
        start=start + 1
    ):


        st.info(
            f"النتيجة {index}"
        )


        st.write(
            f"سورة {r['sura']} - آية {r['aya']}"
        )


        st.write(
            r["text"]
        )


        st.write(
            f"درجة التشابه: {r['score']:.3f}"
        )


        st.divider()



    # =========================
    # Pagination Buttons
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


    st.caption(
        "اكتب كلمة أو معنى للبحث في القرآن"
    )
