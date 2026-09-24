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
# Load Resources
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

    # dagger alef
    text = re.sub(
        r"\u0670",
        "\u0627",
        text
    )

    # remove tashkeel
    text = re.sub(
        r"[\u064B-\u065F\u06D6-\u06ED]",
        "",
        text
    )

    # unify alef
    text = re.sub(
        r"[\u0625\u0623\u0622\u0671]",
        "ا",
        text
    )

    # remove tatweel
    text = re.sub(
        r"\u0640",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()



# =========================
# Exact Search
# =========================

@st.cache_data
def prepare_exact_data(df):

    df = df.copy()

    df["clean_text"] = df["text"].apply(
        normalize_arabic
    )

    return df


verses_clean = prepare_exact_data(
    verses_df
)


def exact_search(query, limit=5):

    q = normalize_arabic(query)

    if len(q) < 2:
        return []

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
# Semantic Search
# =========================

def semantic_search(query, top_k=10):

    q = normalize_arabic(query)

    query_embedding = model.encode(
        q
    )

    scores = util.cos_sim(
        query_embedding,
        segment_embeddings
    )[0].cpu().numpy()


    indices = np.argsort(
        scores
    )[::-1][:top_k]


    results = []


    for idx in indices:

        row = segments_df.iloc[idx]

        results.append({

            "type": "Semantic Match",
            "sura": row["sura"],
            "aya": row["aya"],
            "text": row["text"],
            "score": float(scores[idx])

        })


    return results



# =========================
# Hybrid Search
# =========================

def hybrid_search(query):

    exact = exact_search(query)

    semantic = semantic_search(query)


    final = []

    seen = set()


    for item in exact + semantic:

        key = (
            item["sura"],
            item["aya"]
        )

        if key not in seen:

            final.append(item)
            seen.add(key)


    return final



# =========================
# Interface
# =========================


st.title(
    "📖 Quran Semantic Search"
)


query = st.text_input(
    "Enter your search:"
)


if st.button("Search"):


    if query.strip():


        results = hybrid_search(
            query
        )


        st.subheader(
            "Search Results"
        )


        for r in results:


            if r["type"] == "Exact Match":

                st.success(
                    "📌 Exact Match"
                )

            else:

                st.info(
                    "🧠 Semantic Match"
                )


            st.write(
                f"Surah {r['sura']} - Ayah {r['aya']}"
            )

            st.write(
                r["text"]
            )

            st.write(
                f"Score: {r['score']:.3f}"
            )

            st.divider()


    else:

        st.warning(
            "Please enter a search query"
        )
