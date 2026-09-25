# -*- coding: utf-8 -*-
"""
Semantic Search for the Holy Quran — Streamlit app
====================================================
Two fully independent search modes, shown as separate tabs (matching the
reference site's "بحث نصي" / "بحث دلالي" layout):

  1) بحث نصي (root-based text search) — extracts the Arabic root of every
     significant word in the query using NLTK's ISRI light stemmer
     (Taghva, Elkhoury & Coombs, 2005 — a published, citable algorithm),
     then returns every verse containing a word with a matching root.
     This groups all grammatical derivatives of a word together
     (e.g. النجم / النجوم / نجم / والنجم / بالنجم all share the root "نجم"),
     which is exactly what a reader means by "look up this word".

  2) بحث دلالي (semantic search) — the original embedding-based search,
     completely unchanged: it still ranks all indexed segments by cosine
     similarity, for thematic / conceptual queries that don't share any
     literal words with the target verses (e.g. "الصبر على البلاء" ->
     "أفلا يشكرون").

These two modes are independent and are never blended into a single score.
Root-based matching (mode 1) is a deliberate methodology decision — see the
project report, Section 5.4.1, for the discussion of why light stemming was
adopted for the text-search mode while the semantic mode remains a pure,
stemming-free embedding search.

Files required in the SAME folder as this script:
    - segments.csv            (sura, aya, part_num, text)
    - segment_embeddings.npy  (must have the SAME row count as segments.csv)
    - verses.csv              (sura, aya, text)  -- used for the root-search mode

Run locally with:
    streamlit run app.py
"""

import re
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from nltk.stem.isri import ISRIStemmer

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="البحث في القرآن الكريم",
    page_icon="📖",
    layout="centered",
)

RESULTS_PER_PAGE = 10
MIN_WORD_LEN = 2  # الحد الأدنى لطول الكلمة عشان نحاول نستخرج جذرها

# كلمات وظيفية/نحوية شائعة جدًا (ضمائر، حروف جر وعطف، أفعال مساعدة) —
# نستبعدها قبل استخراج الجذر، وإلا كلمة زي "من" أو "في" بيطلع لها جذر
# يطابق آلاف الآيات بدون أي معنى حقيقي مشترك.
STOPWORDS = {
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
ARABIC_DIACRITICS = re.compile(r"[ً-ٰٟۖ-ۭ]")


def remove_diacritics(text: str) -> str:
    # Dagger Alef (U+0670) represents a real letter, not decoration —
    # convert it to a full Alef instead of deleting it, then strip the
    # remaining diacritics.
    text = re.sub(r"ٰ", "ا", text)
    return ARABIC_DIACRITICS.sub("", text)


def normalize_arabic(text: str) -> str:
    text = remove_diacritics(text)
    text = re.sub(r"[ؐ-ؚۖ-ۜ۟-۪ۨ-ۭ]", "", text)
    text = re.sub(r"[إأآٱ]", "ا", text)  # unify Alef forms
    text = re.sub(r"ـ", "", text)  # remove Tatweel
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_for_text_search(text: str) -> str:
    """طبقة تسامح إضافية لوضع البحث النصي فقط (لا تُستخدم مع النموذج الدلالي):
    توحّد الألف المقصورة (ى) مع الياء (ي)، لأن رسم القرآن العثماني يكتب كلمات
    كثيرة الاستخدام مثل "الذى" بألف مقصورة بينما يكتبها المستخدم العادي "الذي"
    بالياء — نفس الكلمة، رسمان مختلفان."""
    text = normalize_arabic(text)
    text = text.replace("ى", "ي")
    return text


# نطبّع قائمة كلمات التوقف بنفس طريقة تطبيع النصوص، وإلا كلمة مثل "على"
# (بألف مقصورة) ما تطابق نسختها المطبَّعة "علي" (بياء).
STOPWORDS = {normalize_for_text_search(w) for w in STOPWORDS}

_stemmer = ISRIStemmer()


def get_root(word: str) -> str:
    """يرجّع جذر الكلمة عبر خوارزمية ISRI (تجذيع خفيف موثّق أكاديميًا،
    Taghva/Elkhoury/Coombs 2005) المتوفرة في مكتبة NLTK. لا نخترع أي قواعد
    تجذيع يدوية هنا — نعتمد بالكامل على الخوارزمية المنشورة."""
    try:
        return _stemmer.stem(word)
    except Exception:
        return word


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
    verses_df["text_norm"] = verses_df["text"].apply(normalize_for_text_search)

    return segments_df, embeddings, verses_df


@st.cache_data(show_spinner="جاري تجهيز فهرس الجذور (يُبنى مرة واحدة فقط)...")
def build_root_index(verses_df: pd.DataFrame):
    """يبني فهرسًا: جذر الكلمة -> مجموعة أرقام الآيات (index في verses_df)
    التي تحتوي على كلمة بهذا الجذر. يُبنى مرة واحدة ويُخزَّن مؤقتًا
    (cache) عشان ما نعيد استخراج الجذور من جديد مع كل عملية بحث."""
    root_to_verses: dict[str, set[int]] = {}
    for idx, text_norm in zip(verses_df.index, verses_df["text_norm"]):
        roots_in_this_verse = set()
        for w in text_norm.split():
            if len(w) < MIN_WORD_LEN or w in STOPWORDS:
                continue
            root = get_root(w)
            if not root or len(root) < 2:
                continue
            roots_in_this_verse.add(root)
        for r in roots_in_this_verse:
            root_to_verses.setdefault(r, set()).add(idx)
    return root_to_verses


# ---------------------------------------------------------------------------
# Mode 1: بحث نصي — root-based text search (ISRI stemmer)
# ---------------------------------------------------------------------------
def root_search(query: str, verses_df: pd.DataFrame, root_to_verses: dict):
    """يستخرج جذر كل كلمة مهمة بالاستعلام، ثم يرجّع كل الآيات التي فيها
    كلمة واحدة على الأقل تشترك بنفس الجذر — مرتّبة تنازليًا حسب عدد
    الجذور المشتركة (آية تطابق أكثر من كلمة من الاستعلام تطلع أولًا)."""
    query_norm = normalize_for_text_search(query)
    if not query_norm:
        return verses_df.iloc[0:0], []

    words = [w for w in query_norm.split() if len(w) >= MIN_WORD_LEN and w not in STOPWORDS]
    if not words:
        return verses_df.iloc[0:0], []

    query_roots = []
    for w in words:
        r = get_root(w)
        if r and r not in query_roots:
            query_roots.append(r)

    if not query_roots:
        return verses_df.iloc[0:0], query_roots

    match_count: dict[int, int] = {}
    for r in query_roots:
        for idx in root_to_verses.get(r, ()):
            match_count[idx] = match_count.get(idx, 0) + 1

    if not match_count:
        return verses_df.iloc[0:0], query_roots

    ordered_idx = sorted(match_count.keys(), key=lambda i: (-match_count[i], i))
    matches = verses_df.loc[ordered_idx].copy()
    matches["root_match_count"] = [match_count[i] for i in ordered_idx]
    return matches.reset_index(drop=True), query_roots


# ---------------------------------------------------------------------------
# Mode 2: بحث دلالي — semantic search (unchanged embedding-based ranking)
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
# Shared UI helpers
# ---------------------------------------------------------------------------
def render_verse_card(text: str, sura, aya, badge: str = "", color: str = "#1565c0",
                       bg: str = "#eef6ff", size: str = "22px"):
    st.markdown(
        f"""
        <div style="background-color:{bg}; border-right:5px solid {color};
                    padding:18px; border-radius:8px; margin-bottom:14px;
                    direction: rtl; text-align: right;">
            <p style="font-size:{size}; line-height:2;">{text}</p>
            <p style="color:{color}; font-weight:bold;">
                سورة {sura} - آية {aya} {badge}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def paginate(df: pd.DataFrame, page_key: str):
    total_pages = max(1, -(-len(df) // RESULTS_PER_PAGE))  # ceil division
    st.session_state[page_key] = min(st.session_state.get(page_key, 1), total_pages)
    page = st.session_state[page_key]

    start = (page - 1) * RESULTS_PER_PAGE
    end = start + RESULTS_PER_PAGE
    page_rows = df.iloc[start:end]

    if total_pages > 1:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if page > 1 and st.button("⬅ السابق", key=f"{page_key}_prev"):
                st.session_state[page_key] -= 1
                st.rerun()
        with col2:
            st.markdown(
                f"<p style='text-align:center;'>صفحة {page} من {total_pages}</p>",
                unsafe_allow_html=True,
            )
        with col3:
            if page < total_pages and st.button("التالي ➡", key=f"{page_key}_next"):
                st.session_state[page_key] += 1
                st.rerun()

    return page_rows


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
        "بحث نصي بالجذر اللغوي، أو بحث دلالي بالمعنى — اختر التبويب المناسب."
        "</p>",
        unsafe_allow_html=True,
    )

    model = load_model()
    segments_df, embeddings, verses_df = load_data()
    root_to_verses = build_root_index(verses_df)

    tab_text, tab_semantic = st.tabs(["🔤 بحث نصي", "✨ بحث دلالي"])

    # ------------------------------------------------------------------
    # Tab 1: بحث نصي — root-based matching
    # ------------------------------------------------------------------
    with tab_text:
        st.caption(
            "يبحث عن كل الآيات التي تحتوي على أي صيغة مشتقة من نفس جذر الكلمة "
            "(مثال: البحث عن \"النجم\" يطلع \"والنجم إذا هوى\" و\"النجم الثاقب\" معًا)."
        )
        text_query = st.text_input(
            "بحث نصي",
            placeholder="مثال: النجم، الصبر، يوسف...",
            label_visibility="collapsed",
            key="text_query_input",
        )

        if text_query.strip():
            if text_query != st.session_state.get("last_text_query", ""):
                st.session_state.text_page = 1
                st.session_state.last_text_query = text_query

            matches, query_roots = root_search(text_query, verses_df, root_to_verses)

            if matches.empty:
                st.warning("لم يتم العثور على آيات تطابق جذر هذه الكلمة/الكلمات.")
            else:
                st.success(f"تم العثور على {len(matches)} آية (بحث بالجذر اللغوي)")
                page_rows = paginate(matches, "text_page")
                for _, row in page_rows.iterrows():
                    badge = ""
                    if row.get("root_match_count", 1) > 1:
                        badge = f"&nbsp;&nbsp;|&nbsp;&nbsp; تطابق {int(row['root_match_count'])} كلمات"
                    render_verse_card(row["text"], row["sura"], row["aya"], badge=badge)
        else:
            st.info("اكتب كلمة أو أكثر بالأعلى، ثم اضغط Enter.")

    # ------------------------------------------------------------------
    # Tab 2: بحث دلالي — semantic search (unchanged)
    # ------------------------------------------------------------------
    with tab_semantic:
        st.caption(
            "يبحث عن الآيات الأقرب بالمعنى (وليس بلفظ الكلمة)، مفيد للمواضيع "
            "والأفكار العامة مثل \"الصبر على البلاء\"."
        )
        semantic_query = st.text_input(
            "بحث دلالي",
            placeholder="مثال: الصبر على البلاء، التوكل على الله...",
            label_visibility="collapsed",
            key="semantic_query_input",
        )

        if semantic_query.strip():
            if semantic_query != st.session_state.get("last_semantic_query", ""):
                st.session_state.semantic_page = 1
                st.session_state.last_semantic_query = semantic_query

            results = semantic_search(semantic_query, model, segments_df, embeddings)

            if results.empty:
                st.warning("لم يتم العثور على نتائج مشابهة بالمعنى.")
            else:
                top = results.iloc[0]
                st.markdown("#### 🥇 أقرب نتيجة")
                render_verse_card(
                    top["text"], top["sura"], top["aya"],
                    badge=f"&nbsp;&nbsp;|&nbsp;&nbsp; درجة التشابه: {top['score']:.3f}",
                    color="#2e7d32", bg="#f0f7f0",
                )

                remaining = results.iloc[1:].reset_index(drop=True)
                st.markdown(f"#### آيات أخرى ذات صلة ({len(remaining)} نتيجة)")
                page_rows = paginate(remaining, "semantic_page")
                for _, row in page_rows.iterrows():
                    render_verse_card(
                        row["text"], row["sura"], row["aya"],
                        badge=f"&nbsp;&nbsp;|&nbsp;&nbsp; درجة التشابه: {row['score']:.3f}",
                        color="#666", bg="#ffffff", size="18px",
                    )
        else:
            st.info("اكتب فكرة أو موضوعًا بالأعلى، ثم اضغط Enter.")


if __name__ == "__main__":
    main()
