# -*- coding: utf-8 -*-
"""
Semantic Search for the Holy Quran — Streamlit app
====================================================
Two independent result layers, shown as separate sections:

  1) Literal match  (تطابق حرفي) — exact substring match against the full
     verse text, after normalization. If the user types (part of) a verse,
     that verse is shown immediately, at the top.
  2) Semantic match  (تشابه بالمعنى) — the existing embedding-based search,
     completely unchanged: it still ranks all indexed segments by cosine
     similarity, for thematic / conceptual queries.

These two layers are independent and are never blended into a single score.
No stemming and no keyword-weighted scoring is used anywhere in this file.

Files required in the SAME folder as this script:
    - segments.csv            (sura, aya, part_num, text)
    - segment_embeddings.npy  (must have the SAME row count as segments.csv)
    - verses.csv              (sura, aya, text)  -- used for the literal layer

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
    page_title="البحث في القرآن الكريم",
    page_icon="📖",
    layout="centered",
)

RESULTS_PER_PAGE = 10
MAX_LITERAL_MATCHES = 5

# كلمات وظيفية/نحوية شائعة جدًا (ضمائر، حروف جر وعطف، أفعال مساعدة) —
# استبعادها من مطابقة الكلمة المفردة تمنع نتائج عشوائية (لو ظهرت كلمة
# "من" أو "على" بكلام المستخدم، ما نبي نطابق أي آية فيها هالحرف لأنه
# موجود بكل صفحة تقريبًا). هذه قائمة كلمات شائعة قياسية، وليست تجذيعًا:
# لا نُغيّر شكل أي كلمة ولا نستخرج جذرها، فقط نستبعد كلمات معيّنة بالضبط.
LITERAL_STOPWORDS = {
    "من", "الى", "إلى", "على", "في", "ان", "إن", "انه", "إنه", "ما", "لا",
    "الذين", "الا", "إلا", "ولا", "وما", "ثم", "لكم", "او", "أو", "له",
    "الذي", "التي", "هو", "هي", "هم", "هن", "انت", "أنت", "انتم", "أنتم",
    "نحن", "ذلك", "ذالك", "تلك", "هذا", "هاذا", "هذه", "اذا", "إذا", "قد",
    "بل", "ام", "أم", "حتى", "حتي", "كل", "عن", "مع", "قبل", "بعد", "عند",
    "ايضا", "أيضا", "غير", "بين", "دون", "لدى", "نحو", "حيث", "لان", "لأن",
    "لكن", "انما", "إنما", "اذ", "إذ", "كان", "كانوا", "كانت", "يكون",
    "تكون", "قال", "قل", "قالوا", "قالت", "لم", "لن", "لو", "ولو", "فان",
    "فإن", "وان", "وإن", "به", "لها", "لهم", "لهن", "منه", "منها", "منهم",
    "فيه", "فيها", "عليه", "عليها", "عليهم", "عليكم", "كما", "بما", "مما",
    "انا", "أنا", "اني", "إني", "الله", "الرحمن", "الرحيم", "بسم",
}

# ---------------------------------------------------------------------------
# Text normalization (must match exactly the pipeline used to build the
# saved embeddings, including the dagger-Alef fix from Chapter Four).
# ---------------------------------------------------------------------------
ARABIC_DIACRITICS = re.compile(r"[ً-ٰٟۖ-ۭ]")


def remove_diacritics(text: str) -> str:
    # Dagger Alef (U+0670) represents a real letter, not decoration —
    # convert it to a full Alef instead of deleting it, then strip the
    # remaining diacritics.
    text = re.sub(r"ٰ", "ا", text)
    return ARABIC_DIACRITICS.sub("", text)


def normalize_arabic(text: str) -> str:
    text = remove_diacritics(text)
    text = re.sub(r"[ؐ-ؚۖ-ۜ۟-۪ۨ-ۭ]", "", text)
    text = re.sub(r"[إأآٱ]", "ا", text)  # unify Alef forms
    text = re.sub(r"ـ", "", text)  # remove Tatweel
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_for_literal_match(text: str) -> str:
    """طبقة تسامح إضافية للتطابق الحرفي فقط (لا تُستخدم مع النموذج الدلالي):
    توحّد الألف المقصورة (ى) مع الياء (ي)، لأن رسم القرآن العثماني يكتب كلمات
    كثيرة الاستخدام مثل "الذى" بألف مقصورة بينما يكتبها المستخدم العادي "الذي"
    بالياء — نفس الكلمة، رسمان مختلفان."""
    text = normalize_arabic(text)
    text = text.replace("ى", "ي")
    return text


# نطبّع قائمة كلمات التوقف بنفس طريقة تطبيع النصوص (توحيد الألف المقصورة مع
# الياء وغيرها)، وإلا كلمة مثل "على" (بألف مقصورة) ما تطابق نسختها المطبَّعة
# "علي" (بياء) اللي تطلع من نص المستخدم بعد التطبيع.
LITERAL_STOPWORDS = {normalize_for_literal_match(w) for w in LITERAL_STOPWORDS}


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
            "these files must be regenerated together (same run, same order)."
        )

    verses_df = pd.read_csv("verses.csv")
    verses_df["text_norm"] = verses_df["text"].apply(normalize_for_literal_match)

    return segments_df, embeddings, verses_df


# ---------------------------------------------------------------------------
# Layer 1: literal match (exact substring, after normalization)
# ---------------------------------------------------------------------------
def literal_search(query: str, verses_df: pd.DataFrame) -> pd.DataFrame:
    query_norm = normalize_for_literal_match(query)
    if not query_norm:
        return verses_df.iloc[0:0]

    # المستوى الأول: تطابق العبارة كاملة (سواء كتب المستخدم آية كاملة،
    # أو جزء منها، أو عبارة أطول تحتوي على نص الآية)
    mask = verses_df["text_norm"].apply(
        lambda v: query_norm in v or v in query_norm
    )
    matches = verses_df[mask].copy()

    # المستوى الثاني: لو ما فيه تطابق للعبارة كاملة (مثلاً المستخدم كتب
    # وصفًا مثل "الصيام وأحكامه" وليس نص آية)، نفتش كل كلمة من كلامه على
    # حدة — بدون أي اشتقاق أو تجذيع، مجرد فحص وجود الكلمة كما كتبها
    # المستخدم بالضبط داخل الآية.
    if matches.empty:
        words = [
            w for w in query_norm.split()
            if len(w) >= 3 and w not in LITERAL_STOPWORDS
        ]
        if words:
            # مطابقة كلمة كاملة منفصلة (مو أي جزء من كلمة أطول) — وإلا
            # "الحج" تطابق "الحجر" غلطًا لأنها جزء منها حرفيًا
            word_mask = verses_df["text_norm"].apply(
                lambda v: any(w in v.split() for w in words)
            )
            matches = verses_df[word_mask].copy()

    if matches.empty:
        return matches

    # الأقرب طولاً للاستعلام أولاً (تطابق أدق يُعرض أولاً)
    matches["length_diff"] = (matches["text_norm"].str.len() - len(query_norm)).abs()
    matches = matches.sort_values("length_diff").head(MAX_LITERAL_MATCHES)
    return matches.drop(columns=["length_diff"])


# ---------------------------------------------------------------------------
# Layer 2: semantic search (unchanged embedding-based ranking)
# ---------------------------------------------------------------------------
def semantic_search(query: str, model, segments_df: pd.DataFrame, embeddings: np.ndarray):
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
        "<h1 style='text-align: center;'>📖 البحث في القرآن الكريم</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; color: gray;'>"
        "اكتب آية أو جزءًا منها للتطابق الحرفي، أو فكرة/موضوعًا للبحث بالمعنى."
        "</p>",
        unsafe_allow_html=True,
    )

    model = load_model()
    segments_df, embeddings, verses_df = load_data()

    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    if "page" not in st.session_state:
        st.session_state.page = 1

    query = st.text_input(
        "ابحث",
        placeholder="مثال: والنجم إذا هوى — أو: الصبر على البلاء",
        label_visibility="collapsed",
    )

    if query != st.session_state.last_query:
        st.session_state.page = 1
        st.session_state.last_query = query

    if not query.strip():
        st.info("اكتب آية أو فكرة بالأعلى، ثم اضغط Enter.")
        return

    # ------------------------------------------------------------------
    # Layer 1: literal match — shown first, only when something is found
    # ------------------------------------------------------------------
    literal_matches = literal_search(query, verses_df)

    if not literal_matches.empty:
        st.markdown("### 🎯 تطابق حرفي")
        for _, row in literal_matches.iterrows():
            st.markdown(
                f"""
                <div style="background-color:#eef6ff; border-right:5px solid #1565c0;
                            padding:18px; border-radius:8px; margin-bottom:14px;
                            direction: rtl; text-align: right;">
                    <p style="font-size:22px; line-height:2;">{row['text']}</p>
                    <p style="color:#1565c0; font-weight:bold;">
                        سورة {row['sura']} - آية {row['aya']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ------------------------------------------------------------------
    # Layer 2: semantic search — always shown, independent of layer 1
    # ------------------------------------------------------------------
    results = semantic_search(query, model, segments_df, embeddings)

    if results.empty:
        st.warning("لم يتم العثور على نتائج مشابهة بالمعنى.")
        return

    st.markdown("### 🔎 نتائج مشابهة بالمعنى")

    top = results.iloc[0]
    st.markdown(
        f"""
        <div style="background-color:#f0f7f0; border-right:5px solid #2e7d32;
                    padding:18px; border-radius:8px; margin-bottom:24px;
                    direction: rtl; text-align: right;">
            <p style="font-size:22px; line-height:2;">{top['text']}</p>
            <p style="color:#2e7d32; font-weight:bold;">
                سورة {top['sura']} - آية {top['aya']}
                &nbsp;&nbsp;|&nbsp;&nbsp; درجة التشابه: {top['score']:.3f}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
                    سورة {row['sura']} - آية {row['aya']}
                    &nbsp;&nbsp;|&nbsp;&nbsp; درجة التشابه: {row['score']:.3f}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

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
