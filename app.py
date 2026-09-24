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



verses_df, verse_embeddings = load_data(
  # =========================
# Arabic Normalization
# =========================

def normalize_arabic(text):

    text = str(text)


    # إزالة التشكيل

    text = re.sub(
        r"[\u064B-\u065F\u0670\u06D6-\u06ED]",
        "",
        text
    )


    # توحيد الألف

    text = re.sub(
        "[إأآٱ]",
        "ا",
        text
    )


    # إزالة التطويل

    text = text.replace(
        "ـ",
        ""
    )


    # إزالة المسافات الزائدة

    text = re.sub(
        r"\s+",
        " ",
        text
    )


    return text.strip()



# =========================
# Remove Basmala
# =========================

def remove_basmala(text):

    text = normalize_arabic(text)


    basmala = "بسم الله الرحمن الرحيم"


    if text.startswith(basmala):

        text = text.replace(
            basmala,
            "",
            1
        ).strip()


    return text



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



    # إنشاء Embedding للسؤال

    query_embedding = model.encode(
        query_root,
        normalize_embeddings=True
    )



    # حساب التشابه الدلالي

    semantic_scores = util.cos_sim(
        query_embedding,
        verse_embeddings
    )[0].cpu().numpy()



    results = []



    for i, semantic_score in enumerate(semantic_scores):


        original_text = (
            verses_df.iloc[i]["clean_text"]
            if "clean_text" in verses_df.columns
            else verses_df.iloc[i]["text"]
        )


        # تنظيف البسملة أثناء البحث فقط

        verse = remove_basmala(
            original_text
        )


        verse_clean = normalize_arabic(
            verse
        )


        words = [
            get_root(w)
            for w in verse_clean.split()
        ]



        # =====================
        # Keyword Score
        # =====================

        keyword_score = 0



        # تطابق العبارة كاملة

        if query_clean in verse_clean:

            keyword_score = 1.0



        # نفس الكلمة بعد إزالة السوابق

        elif query_root in words:

            keyword_score = 0.8



        # مشتقات مثل نجم / نجوم

        elif any(
            query_root in w
            for w in words
        ):

            keyword_score = 0.5



        # =====================
        # Final Score
        # =====================

        final_score = (

            (0.65 * keyword_score)

            +

            (0.35 * float(semantic_score))

        )



        results.append({

            "score": final_score,

            "semantic": float(semantic_score),

            "keyword": keyword_score,

            "sura": verses_df.iloc[i]["sura"],

            "aya": verses_df.iloc[i]["aya"],

            "text": original_text

        })



    # ترتيب النتائج

    results = sorted(

        results,

        key=lambda x: x["score"],

        reverse=True

    )


    return results[:top_k]
)
