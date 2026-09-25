# -*- coding: utf-8 -*-
"""
Semantic Search for the Holy Quran — Streamlit app
====================================================
Loads precomputed data (segments.csv, verses.csv, segment_embeddings.npy)
and the same pretrained embedding model used during development, then lets
the user search the Quran by meaning.

Files required in the SAME folder as this script:
    - segments.csv            (sura, aya, part_num, text)
    - segment_embeddings.npy  (10705 x 384 float array, matches segments.csv row order)
    - verses.csv              (sura, aya, text)  -- kept for reference / future use

Run locally with:
    streamlit run app.py
"""

import re
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="البحث الدلالي في القرآن الكريم",
    page_icon="📖",
    layout="centered",
)

RESULTS_PER_PAGE = 10

# ---------------------------------------------------------------------------
# Text normalization (must match exactly the pipeline used to build the
# saved embeddings, including the dagger-Alef fix from Chapter Four).
# ---------------------------------------------------------------------------
ARABIC_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")


def remove_diacritics(text: str) -> str:
    # Dagger Alef (U+0670) represents a real letter, not decoration —
    # convert it to a full Alef instead of deleting it, then strip the
    # remaining diacritics.
    text = re.sub(r"\u0670", "\u0627", text)
    return ARABIC_DIACRITICS.sub("", text)


def normalize_arabic(text: str) -> str:
    text = remove_diacritics(text)
    text = re.sub(r"[\u0610-\u061A\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]", "", text)
    text = re.sub(r"[\u0625\u0623\u0622\u0671]", "\u0627", text)  # unify Alef forms
    text = re.sub(r"\u0640", "", text)  # remove Tatweel
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Cached loaders — these run once per app session, not on every search
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="جاري تحميل نموذج البحث الدلالي...")
def load_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


@st.cache_data(show_spinner="جاري تحميل بيانات القرآن...")
def load_data():
    segments_df = pd.read_csv("segments.csv")
    embeddings = np.load("segment_embeddings.npy")
    if len(segments_df) != embeddings.shape[0]:
        raise ValueError(
            f"segments.csv has {len(segments_df)} rows but "
            f"segment_embeddings.npy has {embeddings.shape[0]} — "
            "these files must be regenerated together."
        )
    return segments_df, embeddings


# ---------------------------------------------------------------------------
# Search logic
# ---------------------------------------------------------------------------
def search(query: str, model, segments_df: pd.DataFrame, embeddings: np.ndarray):
    query_normalized = normalize_arabic(query)
    query_embedding = model.encode(query_normalized)
    scores = util.cos_sim(query_embedding, embeddings)[0].numpy()

    ranked_idx = scores.argsort()[::-1]
    results = segments_df.iloc[ranked_idx].copy()
    results["score"] = scores[ranked_idx]
    return results.reset_index(drop=True)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def main():
    st.markdown(
        "<h1 style='text-align: center;'>📖 البحث الدلالي في القرآن الكريم</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; color: gray;'>"
‎        "اكتب فكرة أو موضوعًا بالعربية، وسيعرض النظام الآيات الأقرب بالمعنى."
        "</p>",
        unsafe_allow_html=True,
    )

    model = load_model()
    segments_df, embeddings = load_data()

    # Reset to page 1 whenever the query text actually changes
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    if "page" not in st.session_state:
        st.session_state.page = 1

    query = st.text_input(
‎        "ابحث بالمعنى",
        placeholder="مثال: الصبر على البلاء",
        label_visibility="collapsed",
    )

    if query != st.session_state.last_query:
        st.session_state.page = 1
        st.session_state.last_query = query

    if not query.strip():
        st.info("اكتب سؤالك أو الفكرة اللي تبحث عنها بالأعلى، ثم اضغط Enter.")
        return

    results = search(query, model, segments_df, embeddings)

    if results.empty:
        st.warning("لم يتم العثور على نتائج.")
        return

    # --- Top result, shown prominently above everything else ---
    top = results.iloc[0]
    st.markdown("### أقرب نتيجة بالمعنى")
    st.markdown(
        f"""
        <div style="background-color:#f0f7f0; border-right:5px solid #2e7d32;
                    padding:18px; border-radius:8px; margin-bottom:24px;
                    direction: rtl; text-align: right;">
            <p style="font-size:22px; line-height:2;">{top['text']}</p>
            <p style="color:#2e7d32; font-weight:bold;">
‎                سورة {top['sura']} - آية {top['aya']}
                &nbsp;&nbsp;|&nbsp;&nbsp; درجة التشابه: {top['score']:.3f}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Remaining results, paginated (never discarded, just paged) ---
    remaining = results.iloc[1:].reset_index(drop=True)
    total_pages = max(1, -(-len(remaining) // RESULTS_PER_PAGE))  # ceil division
    st.session_state.page = min(st.session_state.page, total_pages)

    st.markdown(f"#### آيات أخرى ذات صلة ({len(remaining)} نتيجة)")

    start = (st.session_state.page - 1) * RESULTS_PER_PAGE
    end = start + RESULTS_PER_PAGE
    page_rows = remaining.iloc[start:end]

    for _, row in page_rows.iterrows():
        st.markdown(
            f"""
            <div style="border-bottom:1px solid #e0e0e0; padding:12px 0;
                        direction: rtl; text-align: right;">
                <p style="font-size:18px; line-height:1.9;">{row['text']}</p>
                <p style="color:#666; font-size:14px;">
‎                    سورة {row['sura']} - آية {row['aya']}
                    &nbsp;&nbsp;|&nbsp;&nbsp; درجة التشابه: {row['score']:.3f}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- Pagination controls ---
    if total_pages > 1:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.session_state.page > 1:
                if st.button("⬅ السابق"):
                    st.session_state.page -= 1
                    st.rerun()
        with col2:
            st.markdown(
                f"<p style='text-align:center;'>صفحة {st.session_state.page} "
                f"من {total_pages}</p>",
                unsafe_allow_html=True,
            )
        with col3:
            if st.session_state.page < total_pages:
                if st.button("التالي ➡"):
                    st.session_state.page += 1
                    st.rerun()


if __name__ == "__main__":
    main()