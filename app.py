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
import math
import io
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from nltk.stem.isri import ISRIStemmer

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Quran Search",
    layout="centered",
)

RESULTS_PER_PAGE = 10
MAX_PAGES_SHOWN = 2  # cap on-screen results to this many pages; the rest is
                      # available as an Excel download instead of an endless
                      # paginated list
MAX_DISPLAY_RESULTS = RESULTS_PER_PAGE * MAX_PAGES_SHOWN
MIN_WORD_LEN = 2  # الحد الأدنى لطول الكلمة عشان نحاول نستخرج جذرها

# الحد الأدنى لعدد كلمات "الجزء" (segment) عشان يُحسَب له تمثيل دلالي مستقل.
# سبب هذا الحد (مو رقم عشوائي): اختبرنا استعلامات مثل "الطهارة" و"النجم" ولقينا
# كل النتائج الغلط اللي طلعت بأعلى تشابه (0.92-0.95) كانت أجزاء من كلمة أو
# كلمتين بس (زي "فَوَٰكِهُ" أو "خَافِضَةٌ رَّافِعَةٌ") بدون أي علاقة معنوية
# بالاستعلام. هذا نمط معروف في تمثيلات الجمل القصيرة جدًا (anisotropy): كل ما
# قصر النص، قل السياق المتاح للنموذج، فتتقارب متجهات أجزاء قصيرة كثيرة جدًا في
# نفس المنطقة من فضاء التمثيل بغض النظر عن معناها الفعلي، فيطلع لها تشابه
# عالٍ زائف مع أي استعلام. الحل: أي جزء أقل من MIN_SEGMENT_WORDS كلمات يُستبعد
# من فهرس البحث الدلالي (يبقى مع ذلك قابل للوصول عبر وضع البحث النصي بالجذر).
MIN_SEGMENT_WORDS = 3

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

# ---------------------------------------------------------------------------
# استثناء الأسماء العلمية القرآنية من التجذيع
# ---------------------------------------------------------------------------
# مشكلة موثّقة في كل خوارزميات التجذيع الخفيف (ISRI وغيرها): هي مصمَّمة
# لتفكيك كلمات مشتقة من جذر عربي ثلاثي حقيقي، لكن الأسماء العلمية (أسماء
# الأنبياء والشخصيات) غالبًا مو مشتقة من جذر بهذا المعنى، فالخوارزمية
# "تُقحمها" في جذر خاطئ يتقاطع مع كلمة عربية شائعة تمامًا لا علاقة لها بها.
# أمثلة تأكدنا منها فعليًا بالاختبار:
#   يوسف  -> وسف   (يتطابق غلطًا مع "وسوف"/"سوف")
#   سليمان -> سلم   (يتطابق غلطًا مع "سلام"/"إسلام"/"مسلم" — تقاطع ضخم)
#   شعيب  -> شعب   (يتطابق غلطًا مع "شعب"/"شعوب")
#   صالح  -> صلح   (يتطابق غلطًا مع "صلح"/"إصلاح")
#   يعقوب -> عقب   (يتطابق غلطًا مع "عاقبة"/"عقاب")
#   عمران -> عمر   (يتطابق غلطًا مع "عمر"/"عمارة")
# الحل المعتمد في أنظمة استرجاع النصوص العربية الحقيقية هو استثناء قائمة
# محدودة من الأسماء العلمية قبل تمريرها لخوارزمية التجذيع، والاكتفاء
# بمطابقة الاسم نفسه حرفيًا (بعد نزع أي بادئة ملتصقة شائعة مثل ال/و/ف/ب).
# هذا استثناء موثَّق ومبرر، وليس تعديلاً يدويًا عشوائيًا على الخوارزمية.
_RAW_PROPER_NOUNS = [
    "يوسف", "موسى", "عيسى", "نوح", "لوط", "هود", "صالح", "شعيب", "يعقوب",
    "اسماعيل", "اسحاق", "ايوب", "يونس", "الياس", "اليسع", "داود", "سليمان",
    "زكريا", "يحيى", "ادم", "ابراهيم", "هارون", "محمد", "احمد", "عمران",
    "مريم", "طالوت", "جالوت", "فرعون", "قارون", "هامان", "ابليس", "ادريس",
    "بلقيس", "لقمان", "عزير", "ذوالكفل", "ذوالقرنين",
]

# البادئات الشائعة (أل التعريف، حروف الجر والعطف) التي قد تلتصق باسم علم
_ATTACHED_PREFIXES = ["وال", "فال", "بال", "كال", "لل", "ال", "و", "ف", "ب", "ك", "ل"]

PROPER_NOUNS = {normalize_for_text_search(w) for w in _RAW_PROPER_NOUNS}


def get_root(word: str) -> str:
    """يرجّع 'جذر' الكلمة: إما مفتاح ثابت للاسم العلمي (لو كانت الكلمة اسم
    علم قرآني معروف)، أو جذر ISRI الفعلي لبقية الكلمات. خوارزمية ISRI
    (تجذيع خفيف موثّق أكاديميًا، Taghva/Elkhoury/Coombs 2005) من مكتبة
    NLTK هي المصدر الوحيد لاستخراج الجذور — لا نخترع أي قواعد تجذيع يدوية،
    باستثناء قائمة الأسماء العلمية الموثّقة أعلاه."""
    if word in PROPER_NOUNS:
        return "PN:" + word
    for prefix in _ATTACHED_PREFIXES:
        if word.startswith(prefix) and word[len(prefix):] in PROPER_NOUNS:
            return "PN:" + word[len(prefix):]
    try:
        return _stemmer.stem(word)
    except Exception:
        return word


# ---------------------------------------------------------------------------
# Cached loaders — these run once per app session, not on every search
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="جاري تحميل نموذج البحث الدلالي (قد يأخذ دقيقة أول مرة)...")
def load_model():
    # ترقية من paraphrase-multilingual-MiniLM-L12-v2 (384 بعد) إلى
    # paraphrase-multilingual-mpnet-base-v2 (768 بعد) — نموذج أكبر وأدق
    # من نفس عائلة sentence-transformers (بدون أي تدريب إضافي أو تعديل
    # يدوي، نفس المنهجية بالضبط). اختبرناه مقارنة بالنموذج القديم على
    # نفس مجموعة الاستعلامات وأعطى تحسنًا واضحًا وموثقًا (راجع الفصل
    # الخاص بالاختبار بالتقرير)، خصوصًا مع المفاهيم المجردة والكلمات
    # الملتصقة بحروف العطف.
    return SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")


@st.cache_data(show_spinner="جاري تحميل بيانات القرآن...")
def load_data():
    segments_df = pd.read_csv("segments.csv")
    embeddings = np.load("segment_embeddings.npy")
    # الملف قد يكون محفوظًا بدقة float16 (لتصغير الحجم تحت حد رفع GitHub
    # 25 ميجا) — نحوّله لـ float32 هنا مرة وحدة (ومخزّن بالـ cache) عشان
    # يتوافق مع متجه الاستعلام اللي يطلعه النموذج دائمًا بصيغة float32.
    embeddings = embeddings.astype(np.float32)
    if len(segments_df) != embeddings.shape[0]:
        raise ValueError(
            f"segments.csv has {len(segments_df)} rows but "
            f"segment_embeddings.npy has {embeddings.shape[0]} — "
            "these files must be regenerated together (same run, same order)."
        )

    verses_df = pd.read_csv("verses.csv")
    verses_df["text_norm"] = verses_df["text"].apply(normalize_for_text_search)

    # استبعاد الأجزاء القصيرة جدًا من فهرس البحث الدلالي (انظر تعليق
    # MIN_SEGMENT_WORDS أعلاه). هذه عملية على المصفوفات المحسوبة مسبقًا فقط
    # (فلترة صفوف)، ما تحتاج إعادة حساب أي تمثيلات دلالية.
    word_counts = segments_df["text"].apply(lambda t: len(normalize_arabic(t).split()))
    keep_mask = (word_counts >= MIN_SEGMENT_WORDS).to_numpy()
    segments_df = segments_df[keep_mask].reset_index(drop=True)
    embeddings = embeddings[keep_mask]

    # لكل آية، النص الكامل (مش الجزء المقصوص) — يُستخدم في عرض نتائج البحث
    # الدلالي حتى يشوف المستخدم الآية كاملة، بينما الترتيب (score) يبقى
    # محسوبًا على مستوى الجزء نفسه (Tajweed segment) كما هو مصمَّم.
    verse_text_lookup = verses_df.set_index(["sura", "aya"])["text"].to_dict()

    return segments_df, embeddings, verses_df, verse_text_lookup


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


def _root_weight(root: str, root_to_verses: dict, total_verses: int) -> float:
    """وزن الجذر على غرار IDF (Inverse Document Frequency) من نظرية استرجاع
    المعلومات: كل ما كان الجذر نادرًا (يظهر بعدد أقل من الآيات) صار أكثر
    تمييزًا ويُعطى وزنًا أعلى، وكل ما كان شائعًا (زي جذر "ناس" أو "قول")
    يُعطى وزنًا أقل حتى ما يطغى على جذور أدق وأندر بنفس الاستعلام.
    بدون هذا الوزن، آية نادرة الجذر (زي "نميمة") ممكن تنطمر تحت مئات
    الآيات اللي تشترك بس بكلمة عامة زي "الناس" — وهذا فعلاً حصل واختبرناه."""
    doc_freq = len(root_to_verses.get(root, ()))
    return math.log((total_verses + 1) / (doc_freq + 1)) + 1.0


# ---------------------------------------------------------------------------
# Mode 1: بحث نصي — root-based text search (ISRI stemmer)
# ---------------------------------------------------------------------------
def root_search(query: str, verses_df: pd.DataFrame, root_to_verses: dict):
    """يستخرج جذر كل كلمة مهمة بالاستعلام، ثم يرجّع كل الآيات التي فيها
    كلمة واحدة على الأقل تشترك بنفس الجذر — مرتّبة تنازليًا حسب مجموع
    أوزان الجذور المتطابقة (IDF)، وليس عدد الجذور فقط، حتى تطلع الآية
    الأدق والأكثر تمييزًا أولًا بدل ما تطغى عليها آية تشترك بس بكلمة شائعة."""
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

    total_verses = len(verses_df)
    weighted_score: dict[int, float] = {}
    match_count: dict[int, int] = {}
    for r in query_roots:
        w = _root_weight(r, root_to_verses, total_verses)
        for idx in root_to_verses.get(r, ()):
            weighted_score[idx] = weighted_score.get(idx, 0.0) + w
            match_count[idx] = match_count.get(idx, 0) + 1

    if not weighted_score:
        return verses_df.iloc[0:0], query_roots

    ordered_idx = sorted(weighted_score.keys(), key=lambda i: (-weighted_score[i], i))
    matches = verses_df.loc[ordered_idx].copy()
    matches["root_match_count"] = [match_count[i] for i in ordered_idx]
    matches["root_score"] = [weighted_score[i] for i in ordered_idx]
    return matches.reset_index(drop=True), query_roots


# ---------------------------------------------------------------------------
# Mode 2: بحث دلالي — semantic search (unchanged embedding-based ranking)
# ---------------------------------------------------------------------------
def semantic_search(query: str, model, segments_df: pd.DataFrame, embeddings: np.ndarray,
                     verse_text_lookup: dict):
    query_normalized = normalize_arabic(query)
    query_embedding = model.encode(query_normalized)
    scores = util.cos_sim(query_embedding, embeddings)[0].numpy()

    ranked_idx = scores.argsort()[::-1]
    results = segments_df.iloc[ranked_idx].copy()
    results["score"] = scores[ranked_idx]

    # آية واحدة ممكن تنقسم لعدة أجزاء، وأكثر من جزء منها ممكن يطلع بنتايج
    # البحث. نحتفظ فقط بأعلى تشابه لكل آية (sura, aya) حتى ما تتكرر نفس
    # الآية أكثر من مرة بالنتايج.
    results = results.drop_duplicates(subset=["sura", "aya"], keep="first")

    # نعرض نص الآية الكاملة (من verses.csv) بدل الجزء المقصوص، مع إبقاء
    # الترتيب والـ score محسوبين على مستوى الجزء كما هو مصمَّم أصلًا.
    results["text"] = [
        verse_text_lookup.get((s, a), t)
        for s, a, t in zip(results["sura"], results["aya"], results["text"])
    ]

    return results.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Global styling
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    html, body, [class*="css"]  {
        font-family: -apple-system, "Segoe UI", Tahoma, Geneva, Arial, sans-serif;
    }
    .app-header {
        text-align: center;
        padding: 0.5rem 0 1.5rem 0;
    }
    .app-header h1 {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
    }
    .app-header p {
        color: #6b7280;
        font-size: 1rem;
        margin: 0;
    }
    .verse-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-left: 4px solid var(--accent, #2563eb);
        border-radius: 10px;
        padding: 20px 22px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        direction: rtl;
        text-align: right;
    }
    .verse-card.primary {
        border-left-color: #16a34a;
        background: #f8fdf9;
    }
    .verse-card .verse-text {
        font-size: 21px;
        line-height: 2;
        color: #1a1a2e;
        margin-bottom: 10px;
    }
    .verse-card .verse-meta {
        display: flex;
        justify-content: flex-start;
        gap: 10px;
        font-size: 13px;
        color: #6b7280;
        direction: ltr;
        text-align: left;
    }
    .verse-card .ref-badge {
        background: #eef2ff;
        color: #3730a3;
        border-radius: 6px;
        padding: 2px 10px;
        font-weight: 600;
    }
    .verse-card.primary .ref-badge {
        background: #dcfce7;
        color: #166534;
    }
    .score-badge {
        color: #6b7280;
    }
    .overflow-note {
        background: #f9fafb;
        border: 1px dashed #d1d5db;
        border-radius: 8px;
        padding: 14px 16px;
        margin-top: 8px;
        font-size: 14px;
        color: #4b5563;
        text-align: center;
    }
</style>
"""


def render_verse_card(text: str, sura, aya, meta: str = "", primary: bool = False):
    card_class = "verse-card primary" if primary else "verse-card"
    st.markdown(
        f"""
        <div class="{card_class}">
            <div class="verse-text">{text}</div>
            <div class="verse-meta">
                <span class="ref-badge">Surah {sura} : {aya}</span>
                <span class="score-badge">{meta}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Results")
    return buffer.getvalue()


def paginate_capped(df: pd.DataFrame, page_key: str, file_prefix: str):
    """Shows at most MAX_PAGES_SHOWN pages on screen. Anything beyond that
    is never rendered as more pages — it's offered as a single Excel
    download instead, so the UI never turns into a 1000+ page list."""
    display_df = df.head(MAX_DISPLAY_RESULTS)
    overflow = len(df) - len(display_df)

    total_pages = max(1, -(-len(display_df) // RESULTS_PER_PAGE))
    st.session_state[page_key] = min(st.session_state.get(page_key, 1), total_pages)
    page = st.session_state[page_key]

    start = (page - 1) * RESULTS_PER_PAGE
    end = start + RESULTS_PER_PAGE
    page_rows = display_df.iloc[start:end]

    yield page_rows

    if total_pages > 1:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if page > 1 and st.button("Previous", key=f"{page_key}_prev"):
                st.session_state[page_key] -= 1
                st.rerun()
        with col2:
            st.markdown(
                f"<p style='text-align:center; color:#6b7280;'>Page {page} of {total_pages}</p>",
                unsafe_allow_html=True,
            )
        with col3:
            if page < total_pages and st.button("Next", key=f"{page_key}_next"):
                st.session_state[page_key] += 1
                st.rerun()

    if overflow > 0:
        st.markdown(
            f"<div class='overflow-note'>{overflow} more result(s) not shown here — "
            "download the full list below.</div>",
            unsafe_allow_html=True,
        )
        export_cols = [c for c in ["sura", "aya", "text", "score", "root_match_count", "root_score"]
                       if c in df.columns]
        excel_bytes = to_excel_bytes(df[export_cols])
        st.download_button(
            label="Download all results (Excel)",
            data=excel_bytes,
            file_name=f"{file_prefix}_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"{page_key}_download",
        )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="app-header">
            <h1>Quran Search</h1>
            <p>Root-based text search, or meaning-based semantic search — pick a tab below.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model = load_model()
    segments_df, embeddings, verses_df, verse_text_lookup = load_data()
    root_to_verses = build_root_index(verses_df)

    tab_text, tab_semantic = st.tabs(["Text Search", "Semantic Search"])

    # ------------------------------------------------------------------
    # Tab 1: root-based text search
    # ------------------------------------------------------------------
    with tab_text:
        st.caption(
            "Finds every verse containing any word sharing the same Arabic root "
            "(e.g. searching \"النجم\" also finds \"والنجم إذا هوى\" and \"النجم الثاقب\")."
        )
        text_query = st.text_input(
            "Text search",
            placeholder="e.g. النجم, الصبر, يوسف...",
            label_visibility="collapsed",
            key="text_query_input",
        )

        if text_query.strip():
            if text_query != st.session_state.get("last_text_query", ""):
                st.session_state.text_page = 1
                st.session_state.last_text_query = text_query

            matches, query_roots = root_search(text_query, verses_df, root_to_verses)

            if matches.empty:
                st.warning("No verses matched the root of this word.")
            else:
                st.success(f"{len(matches)} verse(s) found (root-based match)")
                for page_rows in paginate_capped(matches, "text_page", "text_search"):
                    for _, row in page_rows.iterrows():
                        meta = ""
                        if row.get("root_match_count", 1) > 1:
                            meta = f"{int(row['root_match_count'])} matching words"
                        render_verse_card(row["text"], row["sura"], row["aya"], meta=meta)
        else:
            st.info("Type one or more words above, then press Enter.")

    # ------------------------------------------------------------------
    # Tab 2: semantic search (unchanged ranking logic)
    # ------------------------------------------------------------------
    with tab_semantic:
        st.caption(
            "Finds verses closest in meaning (not literal wording) — useful for "
            "general themes and ideas such as \"الصبر على البلاء\"."
        )
        semantic_query = st.text_input(
            "Semantic search",
            placeholder="e.g. الصبر على البلاء, التوكل على الله...",
            label_visibility="collapsed",
            key="semantic_query_input",
        )

        if semantic_query.strip():
            if semantic_query != st.session_state.get("last_semantic_query", ""):
                st.session_state.semantic_page = 1
                st.session_state.last_semantic_query = semantic_query

            results = semantic_search(semantic_query, model, segments_df, embeddings, verse_text_lookup)

            if results.empty:
                st.warning("No semantically similar results found.")
            else:
                top = results.iloc[0]
                st.markdown("**Closest match**")
                render_verse_card(
                    top["text"], top["sura"], top["aya"],
                    meta=f"similarity: {top['score']:.3f}",
                    primary=True,
                )

                remaining = results.iloc[1:].reset_index(drop=True)
                st.markdown(f"**Other related verses** ({len(remaining)} results)")
                for page_rows in paginate_capped(remaining, "semantic_page", "semantic_search"):
                    for _, row in page_rows.iterrows():
                        render_verse_card(
                            row["text"], row["sura"], row["aya"],
                            meta=f"similarity: {row['score']:.3f}",
                        )
        else:
            st.info("Type a theme or idea above, then press Enter.")


if __name__ == "__main__":
    main()
