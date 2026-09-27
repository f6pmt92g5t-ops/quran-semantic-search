# -*- coding: utf-8 -*-
"""
Semantic Search for the Holy Quran — Streamlit app (v3)
========================================================
Two search modes, shown as separate tabs:

  1) بحث نصي (root-based text search) — every verse containing a word from
     the same Arabic root as the query word, ranked so that the exact word /
     closest derivatives come first.

  2) بحث دلالي (semantic search) — hybrid ranking:
        final score = semantic similarity (fine-tuned mpnet embeddings)
                      + BETA_LEXICAL x lexical match strength (0..1)

What changed in v3 (and why — every change is backed by a measured problem,
see the comments next to each constant/function):

  * Morphology source: roots/lemmas now come from the Quranic Arabic Corpus
    (Dukes & Habash 2010, morphology v0.4 — human-verified analysis of every
    word in the Quran) instead of the ISRI stemmer. ISRI was measured to both
    SPLIT one family into several roots (الميراث->يرث but ورثه->ورث;
    وبالوالدين->وبالوالد; تحسدوننا->حسدو; الصلاة->صلة which matched ZERO verses)
    and MERGE unrelated words (النفاق and ينفقون -> نفق; التوكل and وكلوا -> وكل).
  * Match strength tiers (exact lemma / close derivative / same root only)
    instead of one flat bonus, so polysemous roots stop flooding the results.
  * Uthmani -> standard spelling bridge (الصلوٰة=الصلاة, الربوٰا=الربا ...).
  * Basmala copies and formulaic phrases no longer lift unrelated verses.
  * Relevance cut-off + "matched words" column + identical verses merged.
  * Built-in evaluation tab (old vs new ranking on a benchmark of queries).

Files required in the SAME folder as this script:
    - segments.csv            (sura, aya, part_num, text)
    - segment_embeddings.npy  (must have the SAME row count as segments.csv)
    - verses.csv              (sura, aya, text)
    - quran-morphology.txt    Quranic Arabic Corpus morphology v0.4 (GNU GPL),
                              Arabic-script edition by github.com/mustafa0x/quran-morphology

Run locally with:
    streamlit run app.py
"""

import re
import io
import os
import math
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from nltk.stem.isri import ISRIStemmer

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Quran Search",
    layout="centered",
)

MODEL_NAME = "Amer-Surur1/quran-finetuned-mpnet"
# كل ملفات البيانات تُقرأ من مجلد app.py نفسه، فيعمل التطبيق مهما كان مجلد التشغيل.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MORPH_FILE = os.path.join(BASE_DIR, "quran-morphology.txt")

RESULTS_SHOWN = 20   # عدد النتائج في الجدول (الباقي في ملف Excel)
MIN_WORD_LEN = 2

# الحد الأدنى لعدد كلمات "الجزء" (segment) عشان يدخل الفهرس الدلالي: الأجزاء
# القصيرة جدًا (كلمة أو كلمتين) يطلع لها تشابه عالٍ زائف مع أي استعلام.
MIN_SEGMENT_WORDS = 3

# "الجزء المتكرر": نص يتكرر حرفيًا في 3 آيات أو أكثر (مثل "والله غفور رحيم" ×13،
# "هم فيها خالدون" ×14، "وهو العزيز الحكيم" ×12). وجدنا 492 نصًا متكررًا تغطي
# 1512 صفًا (14% من الفهرس). المشكلة: الآية تُرتَّب بأعلى أجزائها تشابهًا، فآية
# طويلة عن الطلاق مثلًا تطلع في استعلام "المغفرة" فقط لأنها تنتهي بـ"والله غفور
# رحيم". الحل: الجزء المتكرر لا يُستخدم لترتيب الآية إلا إذا لم يكن لها غيره.
FORMULA_MIN_VERSES = 3

# وزن إشارة المطابقة اللفظية في الدمج الهجين. قيمته عالية نسبيًا لأننا قسنا
# أن النموذج الدلالي وحده ضعيف في الاستعلامات القصيرة: مثلًا تشابه "الحسد" مع
# "أم يحسدون الناس على ما آتاهم الله من فضله" = 0.07 فقط، بينما جزء لا علاقة له
# مثل "يتيما ذا مقربة" = 0.44. وأقرب جار لـ"ومن شر حاسد إذا حسد" في فضاء
# التمثيل هو "ومن شر غاسق إذا وقب" (نفس القالب النحوي، معنى مختلف) — أي أن
# التمثيلات تتأثر ببنية الجملة أكثر من معناها. المطابقة اللفظية الدقيقة (بعد
# إصلاح الصرف) إشارة موثوقة جدًا، فتأخذ وزنًا يضمن ظهور الآيات التي فيها الكلمة.
BETA_LEXICAL = 0.7

# درجات قوة المطابقة اللفظية لكل كلمة في الآية مقارنة بكلمة الاستعلام:
TIER_STRONG = 1.0    # نفس المدخل المعجمي (lemma)، أو نفس الجذر ونفس الوزن الفعلي المزيد (II-X)
TIER_MEDIUM = 0.65   # اشتقاق قريب: فعل/مشتق من الوزن المجرد (I) لاسم مجرد (توبة ← تاب، حسد ← يحسدون)
TIER_WEAK = 0.15     # نفس الجذر فقط بمعنى آخر غالبًا (نفاق ↔ إنفاق، جنة ↔ جِنّة، جهاد ↔ جَهد الأيمان)

# كلمة استعلام يطابق مدخلها أكثر من 12% من آيات القرآن كلمة عامة جدًا (الله 29%،
# قال 21%، كان 18%، رب 14%) لا تميّز شيئًا، فلا تُعطى وزنًا لفظيًا.
LEMMA_DF_MAX = 0.12

# كلمة غير موجودة بلفظها في القرآن: يُعطى جذرها مطابقة متوسطة إذا كان جذرًا
# محددًا (عدد آياته <= 100)، وضعيفة إذا كان واسعًا (غالبًا متعدد المعاني).
FALLBACK_ROOT_MAX_VERSES = 100

# عتبات "الصلة": تُعرض النتيجة كنتيجة ذات صلة إذا طابقت كلمات الاستعلام بما
# يكفي، أو كان تشابهها الدلالي قريبًا من أفضل تشابه. ما عدا ذلك يبقى متاحًا في
# ملف Excel، بدل عرض "6063 نتيجة" أغلبها بلا علاقة.
REL_LEXICAL_MIN = 0.30
REL_SEMANTIC_ABS = 0.55
REL_SEMANTIC_GAP = 0.15

# كلمات وظيفية/نحوية شائعة جدًا — تُستبعد من الاستعلام قبل التحليل الصرفي.
_RAW_STOPWORDS = {
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
    "انا", "أنا", "اني", "إني", "الله",
    # أدوات السؤال والصياغة (فصحى وعامية): "كيف أتعامل مع الظلم" — كلمة "كيف"
    # لها جذر في المعجم فكانت تُحتسب موضوعًا بوزن أعلى من "الظلم" نفسها.
    "كيف", "لماذا", "ماذا", "متى", "اين", "أين", "هل", "كم", "اي", "أي", "ماهو",
    "ماهي", "وش", "ايش", "ليش", "شنو", "حول", "بخصوص", "بشأن", "عند", "يعني",
}
# ملاحظة: "الرحمن/الرحيم/بسم" أُزيلت من القائمة؛ كانت موجودة لأن البسملة ملصقة
# بأول كل سورة فتُغرق النتائج، والآن البسملة تُحذف من النص (انظر load_data)،
# فصار البحث عن "الرحمن" ممكنًا (كان يرجع 0 نتيجة).

# كلمات "إطار" تصف نوع السؤال وليس موضوعه ("أحكام الميراث"، "آيات عن الصبر").
# تُحذف فقط إذا بقي في الاستعلام كلمة موضوع أخرى. مثال مقاس: "أحكام" أعطت
# الجذر "حكم" (161 آية: الحكيم، الحكم لله...) فملأت نتائج "أحكام الميراث".
_RAW_META_WORDS = {
    "احكام", "أحكام", "حكم", "آيات", "ايات", "آية", "اية", "سورة", "سوره",
    "قرآن", "القرآن", "قران", "القران", "موضوع", "معنى", "معني", "تفسير",
    "قصة", "قصه", "قصص", "فضل", "عقوبة", "عقوبه", "حكمة",
    # ألفاظ "الحكم الشرعي" تصف نوع السؤال: في "تحريم الربا" الموضوع هو الربا،
    # و"تحريم" (← حرّم) كانت تجلب "حرّم عليكم الميتة..." وتزاحم آيات الربا.
    "تحريم", "تحليل", "حرمة", "حرمه", "وجوب", "جواز", "مشروعية", "مشروعيه",
}

# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------
# (1) normalize_arabic: MUST stay exactly as the pipeline used to build the saved
#     embeddings (dagger-Alef fix from Chapter Four) — used for the semantic query.
ARABIC_DIACRITICS = re.compile(r"[ً-ٰٟۖ-ۭ]")


def remove_diacritics(text: str) -> str:
    text = re.sub(r"ٰ", "ا", text)
    return ARABIC_DIACRITICS.sub("", text)


def normalize_arabic(text: str) -> str:
    text = remove_diacritics(text)
    text = re.sub(r"[ؐ-ؚۖ-ۜ۟-۪ۨ-ۭ]", "", text)
    text = re.sub(r"[إأآٱ]", "ا", text)
    text = re.sub(r"ـ", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# (2) Uthmani -> standard spelling bridge, used ONLY for lexical matching.
# الرسم العثماني يكتب كلمات كثيرة بغير الإملاء المعتاد، فكانت المطابقة الحرفية
# تفشل: "ٱلصَّلَوٰةَ" كانت تتحول إلى "الصلواة"، فبحث "الصلاة" لا يطابق شيئًا
# (قسناه: الجذر المستخرج "صلة" طابق 0 آية). القواعد:
#   - واو بعدها ألف خنجرية مباشرة = ألف (صلوٰة←صلاة، زكوٰة←زكاة، حيوٰة←حياة، ربوٰا←ربا)
#   - ألف مقصورة بعدها ألف خنجرية = ألف مقصورة (هدىٰ←هدى)
#   - ياء صغيرة ۧ = ياء (إبراهـۧم←إبراهيم)
#   - الألف الخنجرية في غير ذلك: نولّد الصيغتين (بألف وبدونها) لأن الإملاء
#     المعتاد يكتبها أحيانًا (الكتاب، العالمين) ويحذفها أحيانًا (الرحمن، إله، هذا).
_QURAN_MARKS = re.compile(r"[ؐ-ًؚ-ٟۖ-ۭـ]")
_NON_LETTERS = re.compile(r"[^ء-ي]")


def uthmani_plain(text: str, dagger_as_alef: bool = True) -> str:
    text = text.replace("ٱ", "ا")
    text = text.replace("وٰ", "ا")
    text = text.replace("ىٰ", "ى")
    text = text.replace("ۧ", "ي")
    text = text.replace("ٰ", "ا" if dagger_as_alef else "")
    return _QURAN_MARKS.sub("", text)


def lex_key(text: str, dagger_as_alef: bool = True) -> str:
    """مفتاح مطابقة موحّد: بدون تشكيل، همزات الألف = ا، ى = ي، ة = ه."""
    text = uthmani_plain(text, dagger_as_alef)
    text = re.sub(r"[أإآ]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = re.sub(r"ا{2,}", "ا", text)
    return _NON_LETTERS.sub("", text)


STOP_KEYS = {lex_key(w) for w in _RAW_STOPWORDS}
META_KEYS = {lex_key(w) for w in _RAW_META_WORDS}

_PREFIXES = ("وبال", "فبال", "وال", "فال", "بال", "كال", "لل", "ال", "و", "ف", "ب", "ك", "ل")
_SUFFIXES = ("هما", "كما", "هم", "هن", "كم", "كن", "نا", "ها", "ون", "ين", "ان", "ات", "ه", "ي", "ك")

_isri = ISRIStemmer()

# ---------------------------------------------------------------------------
# قاموس مفاهيم مختصر: مصطلحات شائعة يكتبها المستخدم بلفظ غير قرآني أو ملتبس،
# فنربطها بالمفردة القرآنية التي تعبّر عنها. قائمة صغيرة موثقة قابلة للتوسعة.
#   ("lemma", (root, lemma), tier) | ("root_vf", (root, vf), tier) | ("root", root, tier)
#   ("near", (root1, root2, window), tier): كلمتان من الجذرين متقاربتان في الآية
#   ("and", (root1, root2), tier): الجذران موجودان في نفس الآية
# الوضع "replace": التحليل الآلي للكلمة خاطئ فيُستبدل كليًا بالقاعدة.
# الوضع "add": التحليل الآلي صحيح ونضيف له لفظًا قرآنيًا مرادفًا.
# ---------------------------------------------------------------------------
# الميراث: كلمة "ميراث" في القرآن جاءت بمعنى "ميراث السماوات والأرض" (3:180،
# 57:10) لا بمعنى الفرائض. آيات الفرائض تُعرف بأنصبتها: السُّدُس والثُّمُن
# والرُّبُع لا تأتي إلا في 4:11-4:12، والثُّلُث في 4:11، 4:12، 4:176؛ و"نصيب
# مما ترك" (4:7، 4:33).
_INHERITANCE = [
    ("lemma", ("سدس", "سُدُس"), TIER_STRONG), ("lemma", ("ثمن", "ثُمُن"), TIER_STRONG),
    ("lemma", ("ربع", "رُبُع"), TIER_STRONG), ("and", ("نصب", "ترك"), TIER_STRONG),
    ("and", ("ورث", "ترك"), TIER_STRONG),
    # الثلث والنصف يأتيان أيضًا في "ثلثي الليل ونصفه" (73:3، 73:20) فوزنهما أقل
    ("lemma", ("ثلث", "ثُلُث"), TIER_MEDIUM), ("lemma", ("نصف", "نِصْف"), TIER_WEAK),
]
_CONCEPTS_RAW = {
    # "عقوق" ليست لفظًا قرآنيًا؛ النهي عنه في القرآن بلفظ "أُفّ" (17:23، 46:17)
    "عقوق": ("replace", [("lemma", ("أفف", "أُفّ"), TIER_STRONG)]),
    # "الغيبة" تطابق آليًا "غَيْبِهِ" (72:26) فتجلب آيات "عالم الغيب" — خطأ جسيم؛
    # لفظها القرآني "ولا يغتب" (49:12)، ومعها الهمز واللمز.
    "غيبه": ("replace", [("root_vf", ("غيب", "8"), TIER_STRONG), ("root", "همز", TIER_MEDIUM),
                         ("root", "لمز", TIER_MEDIUM)]),
    # "الأخلاق" تتحول آليًا إلى الجذر "خلق" (218 آية عن الخَلْق) — لفظها القرآني "خُلُق"
    "اخلاق": ("replace", [("lemma", ("خلق", "خُلُق"), TIER_STRONG)]),
    # السفر في القرآن غالبًا "الضرب في الأرض" (4:101 قصر الصلاة، 5:106، 73:20).
    # مطابقة قوية: في التقييم الفعلي كانت 4:101 سابعة خلف آيات فيها "الصلاة" فقط.
    "سفر": ("add", [("near", ("ضرب", "أرض", 3), TIER_STRONG)]),
    "مسافر": ("replace", [("lemma", ("سفر", "سَفَر"), TIER_STRONG),
                          ("near", ("ضرب", "أرض", 3), TIER_STRONG)]),
    # الميراث: "replace" لأن لفظ "ميراث" نفسه في القرآن = "ميراث السماوات والأرض"
    # (3:180، 57:10)، ومشتقات ورث غالبًا بمعنى آخر ("أولئك هم الوارثون" 23:10).
    # فآيات الفرائض تُعرف بأنصبتها (انظر _INHERITANCE) وجذر ورث مطابقة ضعيفة فقط.
    "ميراث": ("replace", [("root", "ورث", TIER_WEAK)] + _INHERITANCE),
    "مواريث": ("replace", [("root", "ورث", TIER_WEAK)] + _INHERITANCE),
    "ارث": ("replace", [("root", "ورث", TIER_WEAK)] + _INHERITANCE),
    "فرائض": ("replace", [("root", "ورث", TIER_WEAK)] + _INHERITANCE),
    # جذور يلتبس فيها الاسم بمشتقات فعلية بمعنى آخر (قيس في التقييم الفعلي):
    # "الجنة" جلبت "والجانّ خلقناه من قبل من نار السموم" (15:27، 55:15)،
    # و"الربا" جلبت "أخذة رابية" و"اهتزت وربت" (الانتفاخ). المطابقة القوية
    # هنا للمدخل نفسه فقط، وبقية الجذر ضعيفة.
    "جنه": ("replace", [("lemma", ("جنن", "جَنَّة"), TIER_STRONG), ("root", "جنن", TIER_WEAK)]),
    "ربا": ("replace", [("lemma", ("ربو", "رِبا"), TIER_STRONG), ("root", "ربو", TIER_WEAK)]),
    # الوضوء: لفظه القرآني "فاغسلوا" (5:6)
    "وضوء": ("replace", [("root", "غسل", TIER_STRONG)]),
    "قمار": ("replace", [("lemma", ("يسر", "مَيْسِر"), TIER_STRONG)]),
}
CONCEPTS = {lex_key(k): v for k, v in _CONCEPTS_RAW.items()}


# ---------------------------------------------------------------------------
# Quranic Arabic Corpus morphology -> word-level lexicon
# ---------------------------------------------------------------------------
class Lexicon:
    """Word-level morphology for the whole Quran + inverted indexes."""
    pass


def _nfc(text):
    """توحيد ترتيب علامات التشكيل (Unicode NFC): "جَنَّة" قد تُكتب شدة ثم فتحة أو
    فتحة ثم شدة؛ بدون التوحيد لا تتطابق القاعدة المكتوبة يدويًا مع بيانات المعجم."""
    return unicodedata.normalize("NFC", text) if text else text


def _parse_morphology(path: str) -> dict:
    words = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line[0] == "#":
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                continue
            loc, form, pos, feats = parts
            s, a, w, _seg = loc.split(":")
            key = (int(s), int(a), int(w))
            d = words.get(key)
            if d is None:
                d = words[key] = {"full": "", "root": None, "lemma": None, "vf": None, "stem": None,
                                  "noun": False}
            d["full"] += form
            f = feats.split("|")
            root = lem = vf = None
            for x in f:
                if x.startswith("ROOT:"):
                    root = x[5:]
                elif x.startswith("LEM:"):
                    lem = x[4:]
                elif x.startswith("VF:"):
                    vf = x[3:]
            is_affix = "PREF" in f or "SUFF" in f
            if root is None and "PN" in f and lem:
                root = "PN:" + lem          # أسماء الأعلام بلا جذر: كل اسم "جذر" نفسه
            if root and d["root"] is None:
                d["root"], d["lemma"], d["vf"], d["stem"] = root, lem, vf, form
                d["noun"] = pos == "N"
            elif d["stem"] is None and not is_affix:
                d["stem"] = form
    return words


def _guess_form(word_key: str, root: str, root_forms: dict):
    """يستنتج الوزن الفعلي (II-X) للمصدر من وزنه الصرفي القياسي: تفعيل=II،
    إفعال=IV، تفعُّل=V، تفاعُل=VI، انفعال=VII، افتعال=VIII، استفعال=X،
    وفِعال/مُفاعَلة=III. يُقبل فقط إذا كان هذا الوزن موجودًا فعلًا لهذا الجذر في
    القرآن. مثال: "تحريم" ← حرّم (II) فتطابق "حرّم الربا" ولا تطابق "المسجد
    الحرام"؛ "الجهاد" ← جاهد (III) فتطابق "جاهدوا" ولا تطابق "جَهد أيمانهم"."""
    rk = lex_key(root)
    if len(rk) != 3:
        return None
    a, b, c = rk
    patterns = [
        ("اي" + b + "ا" + c, "4"),        # إفعال من جذر مهموز الأول: إيثار (أثر)، إيمان (أمن)
        ("ت" + a + b + "ي" + c, "2"),
        ("است" + a + b + "ا" + c, "10"),
        ("ا" + a + "ت" + b + "ا" + c, "8"),
        ("ان" + a + b + "ا" + c, "7"),
        ("ا" + a + b + "ا" + c, "4"),
        ("ت" + a + "ا" + b + c, "6"),
        ("ت" + a + b + c, "5"),
        ("م" + a + "ا" + b + c + "ه", "3"),
        (a + b + "ا" + c, "3"),
    ]
    forms = root_forms.get(root, set())
    for pat, form in patterns:
        if word_key == pat and form in forms:
            return form
    return None


def build_lexicon(verses_df: pd.DataFrame, path: str = MORPH_FILE) -> Lexicon:
    raw = _parse_morphology(path)
    vidx = {(int(s), int(a)): i for i, (s, a) in enumerate(zip(verses_df["sura"], verses_df["aya"]))}

    # توحيد مداخل معجمية متكررة بصيغتين في البيانات (مثال: "والِدَي" و"والِد").
    lemmas_by_root = {}
    for d in raw.values():
        if d["root"]:
            lemmas_by_root.setdefault(d["root"], set()).add(d["lemma"])
    lemma_fix = {}
    for r, lems in lemmas_by_root.items():
        for l in lems:
            if l and l.endswith("َي") and l[:-2] in lems:
                lemma_fix[(r, l)] = l[:-2]

    w_verse, w_pos, w_root, w_lemma, w_vf, w_form = [], [], [], [], [], []
    surface = {}
    root_forms = {}
    for (s, a, w), d in raw.items():
        if not d["root"] or (s, a) not in vidx:
            continue
        r = d["root"]
        l = lemma_fix.get((r, d["lemma"]), d["lemma"])
        l = _nfc(l)
        vf = d["vf"]
        w_verse.append(vidx[(s, a)])
        w_pos.append(w)
        w_root.append(r)
        w_lemma.append(l)
        w_vf.append(vf)
        w_form.append(d["full"])
        if vf:
            root_forms.setdefault(r, set()).add(vf)
        analysis = (r, l, vf, d["noun"])
        keys = set()
        for t in (d["full"], d["stem"] or "", l or ""):
            if t:
                keys.add(lex_key(t, True))
                keys.add(lex_key(t, False))
        for k in keys:
            if len(k) >= 2:
                surface.setdefault(k, Counter())[analysis] += 1

    lex = Lexicon()
    lex.n_verses = len(verses_df)
    lex.w_verse = np.array(w_verse, dtype=np.int32)
    lex.w_pos = np.array(w_pos, dtype=np.int32)
    lex.w_root = w_root
    lex.w_lemma = w_lemma
    lex.w_vf = w_vf
    lex.w_form = w_form
    lex.surface = surface
    lex.root_forms = root_forms

    by_root, by_lemma, by_root_vf = {}, {}, {}
    for i, (r, l, vf) in enumerate(zip(w_root, w_lemma, w_vf)):
        by_root.setdefault(r, []).append(i)
        by_lemma.setdefault((r, l), []).append(i)
        by_root_vf.setdefault((r, vf), []).append(i)
    to_arr = lambda d: {k: np.array(v, dtype=np.int32) for k, v in d.items()}
    lex.by_root, lex.by_lemma, lex.by_root_vf = to_arr(by_root), to_arr(by_lemma), to_arr(by_root_vf)
    lex.root_by_key = {}
    for r in by_root:
        if not r.startswith("PN:"):
            lex.root_by_key.setdefault(lex_key(r), r)
    return lex


# ---------------------------------------------------------------------------
# Query analysis
# ---------------------------------------------------------------------------
def _lookup_surface(k: str, lex: Lexicon, allow_h_suffix: bool):
    cands = [k]
    for p in _PREFIXES:
        if k.startswith(p) and len(k) - len(p) >= 2:
            cands.append(k[len(p):])
    extra = []
    for c in cands:
        if c.endswith("ا"):
            extra.append(c[:-1] + "ي")
        elif c.endswith("ي"):
            extra.append(c[:-1] + "ا")
    cands += extra
    for c in cands:
        if c in lex.surface:
            return lex.surface[c]
    for c in cands:
        for suf in _SUFFIXES:
            if suf == "ه" and not allow_h_suffix:
                continue
            if c.endswith(suf) and len(c) - len(suf) >= 2 and c[:-len(suf)] in lex.surface:
                return lex.surface[c[:-len(suf)]]
    return None


def _strip_article(k: str) -> str:
    for p in _PREFIXES:
        if k.startswith(p) and len(k) - len(p) >= 3:
            return k[len(p):]
    return k


def _fallback_root(k: str, base: str, lex: Lexicon):
    """جذر ISRI مع تصحيح أخطائه المعروفة في الجذور المهموزة والمعتلة والمضعّفة،
    ثم نقبله فقط إذا كان جذرًا قرآنيًا. أمثلة مقاسة: الإيثار←"يثر" (الصحيح أثر)،
    الوفاء←"وفء" (الصحيح وفي)، العفة←"عفه" (الصحيح عفف)."""
    for word in (k, base):
        stem = lex_key(_isri.stem(word))
        cands = [stem]
        if len(stem) == 3:
            if stem[0] in "يوا":
                cands.append("ا" + stem[1:])          # همزة أولى: يثر ← أثر
            if stem[2] in "ءاوي":
                cands += [stem[:2] + "ي", stem[:2] + "و"]   # لام معتلة: وفء ← وفي
        if stem.endswith("ه") and len(stem) == 3:
            cands.append(stem[:2] + stem[1])          # مضعّف: عفه ← عفف
        if len(stem) == 2:
            cands.append(stem + stem[1])              # مضعّف: عف ← عفف
        for c in cands:
            r = lex.root_by_key.get(c)
            if r:
                return r
    return None


def analyze_query_word(word: str, lex: Lexicon, ignore_stop: bool = False):
    """Returns a list of matching rules (kind, key, tier) for one query word."""
    k = lex_key(word)
    if len(k) < MIN_WORD_LEN or (k in STOP_KEYS and not ignore_stop):
        return []
    # حرف عطف ملتصق بكلمة وظيفية ("ولكم"، "فيهم" ← "لكم"، "هم"): كلمة وظيفية أيضًا.
    # بدونه حُلّلت "ولكم" كالفعل "ولّى" فأخذت وزن كلمة موضوع (اكتشفه بوت الاختبار).
    if k[:1] in ("و", "ف") and k[1:] in STOP_KEYS:
        return []
    base = _strip_article(k)
    concept = CONCEPTS.get(base) or CONCEPTS.get(k)
    if concept and concept[0] == "replace":
        return list(concept[1])
    rules = []
    hits = _lookup_surface(k, lex, allow_h_suffix=not word.strip().endswith("ة"))
    if hits:
        # "ال" لا تدخل إلا على الأسماء: "الظلم" = الاسم ظُلْم لا الفعل ظَلَمَ. بدون هذا
        # كان الفعل (الأكثر تكرارًا) يُختار، فتأخذ آيات "بظلم/ظلمًا" مطابقة ضعيفة
        # فقط — وهذا سبب تراجع "الظلم" في التقييم الفعلي.
        if k.startswith(("ال", "وال", "فال", "بال", "كال", "لل")):
            nouns = {a: n for a, n in hits.items() if a[3]}
            if nouns:
                hits = nouns
        top = max(hits.values())
        for (r, l, vf, _noun), n in hits.items():
            if n < 0.25 * top:
                # تحليل نادر لنفس الحروف (جِنّة مقابل جَنّة): لا نعطيه مطابقة قوية، لكنه
                # نفس الكلمة المكتوبة حرفيًا فيأخذ مطابقة متوسطة بدل تجاهله كليًا.
                # مثال: "شرب" تُحلَّل فعلًا (شَرِبَ)، و"لها شِرْبٌ" (26:155) اسم نادر.
                rules.append(("lemma", (r, l), TIER_MEDIUM))
                continue
            form = vf or _guess_form(lex_key(l or ""), r, lex.root_forms)
            rules.append(("lemma", (r, l), TIER_STRONG))
            if form and form != "1":
                rules.append(("root_vf", (r, form), TIER_STRONG))
            else:
                rules.append(("root_vf", (r, "1"), TIER_MEDIUM))
            rules.append(("root", r, TIER_WEAK))
    elif not concept:
        # الكلمة غير موجودة في القرآن بلفظها (مثل "الطهارة"، والقرآن فيه تطهروا/
        # طهورًا/المطهرون): نستخرج جذرها بـ ISRI ونقبله إذا كان جذرًا قرآنيًا.
        # - وزن مزيد مؤكد (تحريم=II، تفكّر=V) ← مطابقة قوية لنفس الوزن.
        # - جذر محدد (<= FALLBACK_ROOT_MAX_VERSES آية، مثل طهر: 26) ← مطابقة متوسطة
        #   لكل مشتقاته. قسنا: إعطاؤه مطابقة ضعيفة فقط أنزل "الطهارة" من المرتبة 5
        #   إلى 45 في التقييم الفعلي.
        # - جذر واسع (مثل خلق: 250 آية) ← مطابقة ضعيفة فقط، لأنه غالبًا متعدد المعاني.
        r = _fallback_root(k, base, lex)
        if r:
            form = _guess_form(base, r, lex.root_forms)
            if form and form != "1":
                rules.append(("root_vf", (r, form), TIER_STRONG))
            n_root = len(np.unique(lex.w_verse[lex.by_root[r]]))
            rules.append(("root", r, TIER_MEDIUM if n_root <= FALLBACK_ROOT_MAX_VERSES else TIER_WEAK))
    if concept:
        rules.extend(concept[1])
    return rules


def _rule_word_ids(kind, key, lex: Lexicon):
    if kind == "lemma":
        return lex.by_lemma.get((key[0], _nfc(key[1])))
    if kind == "root_vf":
        return lex.by_root_vf.get(key)
    if kind == "root":
        return lex.by_root.get(key)
    if kind in ("near", "and"):
        r1, r2 = key[0], key[1]
        a, b = lex.by_root.get(r1), lex.by_root.get(r2)
        if a is None or b is None:
            return None
        common = np.intersect1d(lex.w_verse[a], lex.w_verse[b])
        if len(common) == 0:
            return None
        a = a[np.isin(lex.w_verse[a], common)]
        b = b[np.isin(lex.w_verse[b], common)]
        if kind == "and":
            return np.concatenate([a, b])
        window = key[2]
        keep = []
        for i in a:
            same = b[lex.w_verse[b] == lex.w_verse[i]]
            close = same[np.abs(lex.w_pos[same] - lex.w_pos[i]) <= window]
            if len(close):
                keep.append(i)
                keep.extend(close.tolist())
        return np.array(sorted(set(keep)), dtype=np.int32) if keep else None
    return None


def analyze_query(query: str, lex: Lexicon):
    """Splits the query into content words and scores every verse lexically.

    Returns (lexical score per verse in [0,1], per-word tier matrix, word-level
    tiers for highlighting, list of (word, idf))."""
    n = lex.n_verses
    raw_words = [w for w in re.split(r"\s+", normalize_arabic(query)) if len(lex_key(w)) >= MIN_WORD_LEN]
    content = [w for w in raw_words if lex_key(w) not in STOP_KEYS]
    if not content:
        # الاستعلام كله كلمات شائعة (مثل "الله" وحدها) — نبحث بها كما هي
        content = raw_words
    non_meta = [w for w in content if lex_key(w) not in META_KEYS]
    if non_meta:
        content = non_meta

    # كلمة مكررة في الاستعلام ("لها شرب ولكم شرب") تُحتسب مرة واحدة، وإلا تضاعف وزنها.
    seen = set()
    content = [w for w in content if not (lex_key(w) in seen or seen.add(lex_key(w)))]

    candidates = []
    for w in content:
        rules = analyze_query_word(w, lex, ignore_stop=True)
        if not rules:
            continue
        wt = np.zeros(len(lex.w_verse), dtype=np.float32)
        for kind, key, tier in rules:
            ids = _rule_word_ids(kind, key, lex)
            if ids is not None and len(ids):
                wt[ids] = np.maximum(wt[ids], tier)
        vt = np.zeros(n, dtype=np.float32)
        np.maximum.at(vt, lex.w_verse, wt)
        df = int((vt >= TIER_MEDIUM).sum()) or int((vt > 0).sum())
        if df == 0:
            continue
        candidates.append((w, wt, vt, df))
    # الكلمات العامة جدًا (df > 12%) لا تُحتسب إذا وُجدت في الاستعلام كلمات أدق
    specific = [c for c in candidates if c[3] / n <= LEMMA_DF_MAX]
    if specific:
        candidates = specific

    word_tier_all = np.zeros(len(lex.w_verse), dtype=np.float32)
    verse_tiers, weights, used = [], [], []
    for w, wt, vt, df in candidates:
        # وزن الكلمة = IDF × أقوى مطابقة ممكنة لها. كلمة ليس لها في القرآن إلا
        # جذر بمعنى آخر (مثل "أتعامل" ← عمل) تأخذ وزنًا صغيرًا، فلا تُضعف وزن
        # كلمات الموضوع الحقيقية في نفس الاستعلام.
        idf = (math.log((n + 1) / (df + 1)) + 1.0) * float(vt.max())
        verse_tiers.append(vt)
        weights.append(idf)
        used.append((w, round(idf, 2)))
        word_tier_all = np.maximum(word_tier_all, wt)

    if not weights:
        return np.zeros(n, dtype=np.float32), None, word_tier_all, used
    tiers = np.vstack(verse_tiers)
    wv = np.array(weights, dtype=np.float32)
    lexical = (tiers * wv[:, None]).sum(0) / wv.sum()
    coverable = tiers.max(1) >= TIER_MEDIUM
    if coverable.sum() > 1:
        # معامل التغطية (coordination factor، كما في تشابه Lucene الكلاسيكي): الآية
        # التي تطابق كل كلمات الاستعلام أولى من آية تطابق كلمة واحدة بقوة.
        # مثال مقاس: في "عقوق الوالدين" كانت "ووالد وما ولد" (كلمة واحدة، آية
        # قصيرة) تسبق 17:23 و46:17 اللتين فيهما "الوالدين" و"أُفّ" معًا.
        coverage = (tiers[coverable] >= TIER_MEDIUM).sum(0) / coverable.sum()
        lexical = lexical * (0.5 + 0.5 * coverage)
    return lexical, tiers, word_tier_all, used


def matched_words_by_verse(word_tier: np.ndarray, lex: Lexicon, min_tier: float):
    out = {}
    for i in np.nonzero(word_tier >= min_tier)[0]:
        out.setdefault(int(lex.w_verse[i]), []).append(lex.w_form[i])
    return {k: "، ".join(dict.fromkeys(v)) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Legacy ISRI pipeline (v2) — kept ONLY for the before/after evaluation tab
# ---------------------------------------------------------------------------
def normalize_for_text_search(text: str) -> str:
    return normalize_arabic(text).replace("ى", "ي")


LEGACY_STOPWORDS = {normalize_for_text_search(w) for w in _RAW_STOPWORDS}
_LEGACY_PN = {normalize_for_text_search(w) for w in [
    "يوسف", "موسى", "عيسى", "نوح", "لوط", "هود", "صالح", "شعيب", "يعقوب",
    "اسماعيل", "اسحاق", "ايوب", "يونس", "الياس", "اليسع", "داود", "سليمان",
    "زكريا", "يحيى", "ادم", "ابراهيم", "هارون", "محمد", "احمد", "عمران",
    "مريم", "طالوت", "جالوت", "فرعون", "قارون", "هامان", "ابليس", "ادريس",
    "بلقيس", "لقمان", "عزير", "ذوالكفل", "ذوالقرنين"]}
_LEGACY_PREFIXES = ["وال", "فال", "بال", "كال", "لل", "ال", "و", "ف", "ب", "ك", "ل"]


def legacy_get_root(word: str) -> str:
    if word in _LEGACY_PN:
        return "PN:" + word
    for p in _LEGACY_PREFIXES:
        if word.startswith(p) and word[len(p):] in _LEGACY_PN:
            return "PN:" + word[len(p):]
    try:
        return _isri.stem(word)
    except Exception:
        return word


def legacy_build_root_index(verse_texts) -> dict:
    root_to_verses = {}
    for idx, text in enumerate(verse_texts):
        roots = set()
        for w in normalize_for_text_search(text).split():
            if len(w) < MIN_WORD_LEN or w in LEGACY_STOPWORDS:
                continue
            r = legacy_get_root(w)
            if r and len(r) >= 2:
                roots.add(r)
        for r in roots:
            root_to_verses.setdefault(r, set()).add(idx)
    return root_to_verses


def legacy_bonus(query: str, root_to_verses: dict, n: int) -> np.ndarray:
    words = [w for w in normalize_for_text_search(query).split()
             if len(w) >= MIN_WORD_LEN and w not in LEGACY_STOPWORDS]
    roots = list(dict.fromkeys(legacy_get_root(w) for w in words))
    gmax = math.log(n + 1) + 1.0
    bonus = np.zeros(n)
    for r in roots:
        df = len(root_to_verses.get(r, ()))
        w = math.log((n + 1) / (df + 1)) + 1.0
        b = 0.5 * min(1.0, w / gmax)
        for v in root_to_verses.get(r, ()):
            bonus[v] += b
    return np.minimum(bonus, 0.5)


# ---------------------------------------------------------------------------
# Data loading (cached once per server process)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="جاري تحميل نموذج البحث الدلالي (قد يأخذ دقيقة أول مرة)...")
def load_model():
    # نموذج mpnet متعدد اللغات بعد تدريب إضافي (fine-tuning) على 16929 زوج
    # (تفسير الآية ↔ نص الآية). انظر فصل التقييم في التقرير.
    return SentenceTransformer(MODEL_NAME)


class SearchData:
    pass


@st.cache_resource(show_spinner="جاري تحميل بيانات القرآن والمعجم الصرفي (مرة واحدة)...")
def load_data() -> SearchData:
    D = SearchData()
    segments_df = pd.read_csv(os.path.join(BASE_DIR, "segments.csv"))
    embeddings = np.load(os.path.join(BASE_DIR, "segment_embeddings.npy")).astype(np.float32)
    if len(segments_df) != embeddings.shape[0]:
        raise ValueError(
            f"segments.csv has {len(segments_df)} rows but segment_embeddings.npy has "
            f"{embeddings.shape[0]} — these files must be regenerated together.")
    verses_df = pd.read_csv(os.path.join(BASE_DIR, "verses.csv"))

    # البسملة ملصقة في بداية الآية الأولى من 112 سورة في verses.csv، وليست من
    # الآية (إلا الفاتحة). نحذفها من نص العرض ومن الفهرسة، ونُبقي النص الأصلي
    # للمقارنة مع النسخة السابقة فقط.
    basmala = verses_df.loc[(verses_df.sura == 1) & (verses_df.aya == 1), "text"].iloc[0].strip()
    verses_df["text_original"] = verses_df["text"]
    first = (verses_df.aya == 1) & (verses_df.sura != 1) & verses_df.text.str.startswith(basmala)
    verses_df.loc[first, "text"] = verses_df.loc[first, "text"].str[len(basmala):].str.strip()
    D.verses = verses_df.reset_index(drop=True)
    n = len(D.verses)
    vidx = {(int(s), int(a)): i for i, (s, a) in enumerate(zip(D.verses.sura, D.verses.aya))}

    seg_norm = segments_df["text"].apply(normalize_arabic)
    word_counts = seg_norm.str.split().str.len().to_numpy()
    seg_v = np.array([vidx[(int(s), int(a))] for s, a in zip(segments_df.sura, segments_df.aya)])
    is_basmala = (segments_df["text"].str.strip() == basmala).to_numpy() & ~(
        (segments_df.sura == 1) & (segments_df.aya == 1)).to_numpy()
    verses_per_text = pd.Series(seg_v).groupby(seg_norm.values).transform("nunique").to_numpy()
    is_formula = verses_per_text >= FORMULA_MIN_VERSES
    long_enough = word_counts >= MIN_SEGMENT_WORDS

    D.emb = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    D.seg_v = seg_v
    D.prim = long_enough & ~is_basmala & ~is_formula
    D.second = long_enough & ~is_basmala & is_formula
    D.legacy_mask = long_enough            # v2 behaviour (for the evaluation tab)
    D.n = n

    D.lex = build_lexicon(D.verses)
    D.legacy_roots = legacy_build_root_index(D.verses["text_original"])
    D.text_key = D.verses["text"].apply(lex_key).to_numpy()   # لدمج الآيات المتطابقة نصًا
    # كلمات كل آية بعد التوحيد وحذف السوابق: لترتيب "الكلمة نفسها أولًا" في البحث النصي
    D.word_sets = [{_strip_article(lex_key(w)) for w in t.split()} for t in D.verses["text"]]
    return D


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
def _verse_max(values: np.ndarray, seg_v: np.ndarray, mask: np.ndarray, n: int) -> np.ndarray:
    out = np.full(n, -np.inf, dtype=np.float32)
    np.maximum.at(out, seg_v[mask], values[mask])
    return out


def semantic_components(query: str, model, D: SearchData):
    q = model.encode(normalize_arabic(query)).astype(np.float32)
    q /= np.linalg.norm(q) + 1e-12
    cos = D.emb @ q
    sem = _verse_max(cos, D.seg_v, D.prim, D.n)
    sem2 = _verse_max(cos, D.seg_v, D.second, D.n)
    sem = np.where(np.isfinite(sem), sem, sem2)
    sem = np.where(np.isfinite(sem), sem, 0.0).astype(np.float32)
    return sem, cos


def _merge_identical(df: pd.DataFrame, D: SearchData) -> pd.DataFrame:
    """آيات متطابقة نصًا (مثل "فبأي آلاء ربكما تكذبان" ×31) تُعرض مرة واحدة."""
    keys = D.text_key[df["vidx"].to_numpy()]
    first_row, others = {}, {}
    for pos, (k, s, a) in enumerate(zip(keys, df["sura"], df["aya"])):
        if k in first_row:
            others[k].append(f"{s}:{a}")
        else:
            first_row[k] = pos
            others[k] = []
    keep = sorted(first_row.values())
    out = df.iloc[keep].copy()
    out["also"] = [", ".join(others[k]) for k in keys[keep]]
    return out


def semantic_search(query: str, model, D: SearchData, sem_cos=None) -> pd.DataFrame:
    if sem_cos is None:
        sem, _cos = semantic_components(query, model, D)
    else:
        sem = sem_cos
    lexical, _tiers, word_tier, _used = analyze_query(query, D.lex)
    score = sem + BETA_LEXICAL * lexical
    order = np.argsort(-score, kind="stable")
    sem_cut = max(REL_SEMANTIC_ABS, float(sem.max()) - REL_SEMANTIC_GAP)
    relevant = (lexical >= REL_LEXICAL_MIN) | (sem >= sem_cut)
    matched = matched_words_by_verse(word_tier, D.lex, TIER_MEDIUM)
    df = pd.DataFrame({
        "vidx": order,
        "sura": D.verses.sura.to_numpy()[order],
        "aya": D.verses.aya.to_numpy()[order],
        "text": D.verses.text.to_numpy()[order],
        "score": score[order],
        "semantic": sem[order],
        "lexical": lexical[order],
        "relevant": relevant[order],
    })
    df["matched"] = [matched.get(int(i), "") for i in order]
    return _merge_identical(df, D).reset_index(drop=True)


def text_search(query: str, D: SearchData) -> pd.DataFrame:
    lexical, tiers, word_tier, _used = analyze_query(query, D.lex)
    if tiers is None:
        return pd.DataFrame(columns=["vidx", "sura", "aya", "text", "score", "matched", "also"])
    hits = np.nonzero(lexical > 0)[0]
    words_matched = (tiers[:, hits] > 0).sum(0)
    # عند تساوي الدرجة (مثلًا كل مشتقات "استوى" قوية) تتقدم الآية التي فيها الكلمة
    # بلفظها كما كُتبت ("استويت" 23:28) بدل ترتيب المصحف فقط.
    qkeys = {_strip_article(lex_key(w)) for w in normalize_arabic(query).split()} - STOP_KEYS
    exact = np.array([len(qkeys & D.word_sets[i]) for i in hits])
    order = sorted(range(len(hits)), key=lambda j: (-round(float(lexical[hits[j]]), 6), -exact[j], hits[j]))
    idx = hits[order]
    matched = matched_words_by_verse(word_tier, D.lex, TIER_WEAK)
    df = pd.DataFrame({
        "vidx": idx,
        "sura": D.verses.sura.to_numpy()[idx],
        "aya": D.verses.aya.to_numpy()[idx],
        "text": D.verses.text.to_numpy()[idx],
        "score": lexical[idx],
        "words_matched": words_matched[order],
    })
    df["matched"] = [matched.get(int(i), "") for i in idx]
    return _merge_identical(df, D).reset_index(drop=True)


def legacy_ranking(query: str, cos: np.ndarray, D: SearchData) -> list:
    """ترتيب النسخة السابقة (v2) حرفيًا: أعلى جزء + مكافأة جذر ISRI حتى 0.5."""
    sem = _verse_max(cos, D.seg_v, D.legacy_mask, D.n)
    score = np.where(np.isfinite(sem), sem + legacy_bonus(query, D.legacy_roots, D.n), -np.inf)
    order = np.argsort(-score, kind="stable")
    return [(int(D.verses.sura[i]), int(D.verses.aya[i])) for i in order if np.isfinite(score[i])]


# ---------------------------------------------------------------------------
# Evaluation benchmark (gold = key verses that clearly belong to each topic)
# القائمة بُنيت بطريقة "التجميع" (pooling) المعتمدة في تقييم محركات البحث (TREC):
# قائمة أولية بالآيات الأساسية، ثم حُكم يدويًا على كل آية ظهرت في أول 10 نتائج
# للنظامين القديم والجديد، وأُضيف منها ما يتعلق بالموضوع — حتى لا يُظلم أي نظام.
# جولة تجميع ثانية شملت نتائج نماذج e5 وBGE-M3 (انظر colab_compare_models.py).
# ---------------------------------------------------------------------------
GOLD = {
    "الجنة والنار": "59:20 7:44 7:46 7:50 13:35 47:15 3:185 2:221 5:72 42:7 41:40 2:81 2:82 11:106 11:108 104:6 3:10 47:12",
    "الحياة الدنيا": "57:20 6:32 29:64 3:185 18:45 18:46 10:24 47:36 3:14 13:26 43:35 28:60 87:16 79:38 35:5 31:33 40:39 2:212 9:38 42:36 11:15 10:7 2:86 10:23 20:72",
    "تحريم الربا": "2:275 2:276 2:278 2:279 2:280 3:130 30:39 4:161",
    "الصلاة في السفر": "4:101 4:102 4:103 4:43 5:6 2:239 73:20",
    "أحكام الميراث": "4:7 4:8 4:11 4:12 4:13 4:33 4:176 2:180 2:240 5:106 8:75 33:6 89:19 2:233",
    "الجهاد": "2:216 2:218 4:95 8:72 8:74 9:20 9:24 9:41 9:44 9:73 9:81 9:86 9:88 22:78 25:52 29:6 29:69 47:31 49:15 60:1 61:11 66:9 5:35 5:54 3:142 9:111 9:19 16:110 8:75 9:16",
    "التوبة": "66:8 4:17 4:18 9:104 42:25 25:70 25:71 20:82 39:53 2:37 2:160 9:118 5:39 6:54 24:31 11:3 11:52 11:90 40:3 110:3 2:222 7:153 16:119 3:89 4:146 4:16 4:92 66:4 9:15 9:74 5:74 3:90",
    "الرزق": "11:6 51:22 51:58 65:2 65:3 34:36 34:39 13:26 17:30 29:62 42:12 42:27 30:37 39:52 28:82 16:71 20:132 67:15 35:3 10:31 27:64 2:212 3:37 24:38 29:60 67:21 30:40 37:41 65:11 24:26 8:74 34:4 16:75 2:233 6:151",
    "الحسد": "113:5 2:109 4:54 48:15 4:32 12:8 12:5 5:27 3:120",
    "عقوق الوالدين": "17:23 17:24 46:17 46:18 46:15 31:14 31:15 29:8 2:83 4:36 6:151 19:14 19:32 18:80 14:41 71:28 27:19",
    "التوكل على الله": "65:3 3:159 3:160 3:173 8:2 8:49 9:51 9:129 11:56 11:88 11:123 12:67 14:11 14:12 16:42 16:99 25:58 26:217 27:79 29:59 33:3 33:48 39:38 42:10 42:36 58:10 60:4 64:13 67:29 73:9 5:11 5:23 10:71 10:84 10:85 13:30 4:81 7:89",
    "الظلم": "31:13 4:40 10:44 18:49 40:17 3:57 3:140 14:42 42:42 20:111 4:10 6:82 11:113 3:108 3:182 50:29 41:46 16:118 3:117 30:9 9:70 16:33 23:62 2:57 7:160 4:110 26:227 40:31 6:131 27:14 4:160 42:41 2:140 2:231 22:25 25:4 16:61 4:30",
    "النفاق": "63:1 63:2 63:3 63:4 63:7 63:8 4:61 4:88 4:138 4:140 4:142 4:143 4:145 9:64 9:67 9:68 9:73 9:77 9:101 33:1 33:12 33:48 33:60 33:73 48:6 57:13 59:11 66:9 29:11 2:8 2:9 2:10 2:14 3:167 8:49 9:97",
    "التفكر في خلق الكون": "3:190 3:191 2:164 30:8 45:13 13:3 10:5 10:6 16:11 16:12 88:17 50:6 67:3 7:185 29:20 31:10 21:30 41:53 51:20 51:21 45:3 45:4 45:5 36:37 16:69 30:21 6:101 16:3 39:5 29:44 14:19 56:59 39:62 10:24 86:5 34:46 7:176 24:45",
    "الصبر على البلاء": "2:155 2:156 2:157 2:153 2:177 3:186 22:35 39:10 12:18 12:83 31:17 16:127 42:43 47:31 3:142 29:2 29:3 21:83 38:44 76:12 13:22 13:24 11:11 3:200 2:45 70:5 46:35 8:46 2:249 16:126 14:12 16:42 52:48",
    "الطهارة": "5:6 2:222 4:43 9:108 74:4 8:11 2:125 22:26 56:79 25:48 33:33 9:103 7:82 27:56 5:41 2:232 58:12",
}


def _gold(q):
    return {tuple(map(int, x.split(":"))) for x in GOLD[q].split()}


def _metrics(ranking: list, gold: set) -> dict:
    first = next((i for i, k in enumerate(ranking, 1) if k in gold), None)
    return {
        "MRR": 1.0 / first if first else 0.0,
        "P@10": sum(k in gold for k in ranking[:10]) / 10,
        "R@20": sum(k in gold for k in ranking[:20]) / len(gold),
        "first hit": first or "-",
    }


def _top_refs(ranking: list, gold: set, k: int = 10) -> str:
    return " ".join(f"{s}:{a}{'✓' if (s, a) in gold else '✗'}" for s, a in ranking[:k])


def run_benchmark(model, D: SearchData) -> pd.DataFrame:
    rows = []
    for q in GOLD:
        gold = _gold(q)
        sem, cos = semantic_components(q, model, D)
        old = legacy_ranking(q, cos, D)
        new_df = semantic_search(q, model, D, sem_cos=sem)
        new = list(zip(new_df.sura.astype(int), new_df.aya.astype(int)))
        mo, mn = _metrics(old, gold), _metrics(new, gold)
        rows.append({"query": q, "gold": len(gold),
                     "old MRR": mo["MRR"], "new MRR": mn["MRR"],
                     "old P@10": mo["P@10"], "new P@10": mn["P@10"],
                     "old R@20": mo["R@20"], "new R@20": mn["R@20"],
                     "old first hit": mo["first hit"], "new first hit": mn["first hit"],
                     "new top10": _top_refs(new, gold), "old top10": _top_refs(old, gold)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    html, body, [class*="css"]  {
        font-family: -apple-system, "Segoe UI", Tahoma, Geneva, Arial, sans-serif;
    }
    .app-header { text-align: center; padding: 0.5rem 0 1.5rem 0; }
    .app-header h1 { font-size: 2.1rem; font-weight: 700; color: #1a1a2e;
                     margin-bottom: 0.25rem; letter-spacing: -0.02em; }
    .app-header p { color: #6b7280; font-size: 1rem; margin: 0; }
</style>
"""


# ملف Excel لكل النتائج (~6000 صف) يأخذ حوالي ثانية؛ نخزّنه حتى لا يُعاد بناؤه
# مع كل تفاعل في الصفحة (تبديل تبويب، ضغط مربع اختيار...).
@st.cache_data(max_entries=64, show_spinner=False)
def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Results")
    return buffer.getvalue()


def render_results_table(df: pd.DataFrame, score_label: str, height: int = 700):
    view = df.head(RESULTS_SHOWN).copy()
    view.insert(0, "Ref", [f"{s}:{a}" for s, a in zip(view["sura"], view["aya"])])
    cols = ["Ref", "text", "score", "matched"]
    config = {
        "Ref": st.column_config.TextColumn("Surah:Aya", width="small"),
        "text": st.column_config.TextColumn("Verse", width="large"),
        "score": st.column_config.NumberColumn(score_label, format="%.3f", width="small"),
        "matched": st.column_config.TextColumn("Matched words", width="medium"),
    }
    if "also" in view.columns and view["also"].astype(bool).any():
        cols.append("also")
        config["also"] = st.column_config.TextColumn("Same text also in", width="small")
    st.dataframe(view[cols], column_config=config, hide_index=True,
                 width="stretch", height=height)


def download_button(df: pd.DataFrame, cols: list, name: str, key: str):
    export = df.copy()
    export.insert(0, "Ref", [f"{s}:{a}" for s, a in zip(export["sura"], export["aya"])])
    st.download_button(
        label=f"Download all {len(export)} results (Excel)",
        data=to_excel_bytes(export[["Ref"] + [c for c in cols if c in export.columns]]),
        file_name=f"{name}_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=key,
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def has_arabic(text: str) -> bool:
    """استعلام بلا أي حرف عربي (hello، 123، رموز) لا معنى للبحث به في النص القرآني."""
    return re.search(r"[\u0621-\u064A]", text) is not None


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
    D = load_data()

    show_eval = st.sidebar.checkbox("Evaluation tab (developers)", value=False)
    names = ["Text Search", "Semantic Search"] + (["Evaluation"] if show_eval else [])
    tabs = st.tabs(names)

    # ------------------------------------------------------------------
    # Tab 1: root-based text search
    # ------------------------------------------------------------------
    with tabs[0]:
        st.caption(
            "Finds every verse containing any word sharing the same Arabic root "
            "(e.g. searching \"النجم\" also finds \"والنجم إذا هوى\" and \"النجم الثاقب\"). "
            "Exact words and close derivatives are listed first.")
        text_query = st.text_input("Text search", placeholder="e.g. النجم, الصبر, يوسف...",
                                   label_visibility="collapsed", key="text_query_input", max_chars=200)
        if text_query.strip() and not has_arabic(text_query):
            st.warning("Please type the query in Arabic letters (e.g. الصبر).")
        elif text_query.strip():
            matches = text_search(text_query, D)
            if matches.empty:
                st.warning("No verses matched the root of this word.")
            else:
                st.success(f"{len(matches)} verse(s) found (root-based match)")
                render_results_table(matches, "Match")
                if len(matches) > RESULTS_SHOWN:
                    download_button(matches, ["text", "score", "words_matched", "matched", "also"],
                                    "text_search", "text_dl")
        else:
            st.info("Type one or more words above, then press Enter.")

    # ------------------------------------------------------------------
    # Tab 2: semantic search (hybrid)
    # ------------------------------------------------------------------
    with tabs[1]:
        st.caption(
            "Finds verses closest in meaning (not literal wording) — useful for "
            "general themes and ideas such as \"الصبر على البلاء\".")
        semantic_query = st.text_input("Semantic search",
                                       placeholder="e.g. الصبر على البلاء, التوكل على الله...",
                                       label_visibility="collapsed", key="semantic_query_input", max_chars=200)
        if semantic_query.strip() and not has_arabic(semantic_query):
            st.warning("Please type the query in Arabic letters (e.g. الصبر على البلاء).")
        elif semantic_query.strip():
            results = semantic_search(semantic_query, model, D)
            relevant = results[results["relevant"]]
            show_all = st.checkbox("Include loosely related verses", value=False, key="sem_all")
            shown = results if (show_all or relevant.empty) else relevant
            if relevant.empty:
                st.warning("No strongly related verses found — showing the closest ones.")
            else:
                st.markdown(f"**{len(relevant)} related verse(s)** — top {min(RESULTS_SHOWN, len(shown))} shown")
            render_results_table(shown, "Score")
            download_button(results, ["text", "score", "semantic", "lexical", "relevant", "matched", "also"],
                            "semantic_search", "sem_dl")
            with st.expander("How is the score computed?"):
                st.markdown(
                    f"- **Score** = meaning similarity (fine-tuned model) + {BETA_LEXICAL} × word-match strength.\n"
                    "- **Word match**: exact Quranic word/lemma = 1.0, close derivative = 0.65, "
                    "same root with a different meaning = 0.15 (roots and lemmas from the Quranic "
                    "Arabic Corpus).\n"
                    "- **Matched words** shows the words in the verse that matched your query.")
        else:
            st.info("Type a theme or idea above, then press Enter.")

    # ------------------------------------------------------------------
    # Tab 3: evaluation (developers)
    # ------------------------------------------------------------------
    if show_eval:
        with tabs[2]:
            st.caption("Old ranking (v2: ISRI root bonus) vs new ranking (v3) on a fixed benchmark. "
                       "Gold = key verses of each topic. MRR = 1/rank of the first gold verse; "
                       "P@10 = share of gold verses in the top 10; R@20 = share of gold found in the top 20.")
            if st.button("Run benchmark", key="run_bench"):
                with st.spinner("Running..."):
                    bench = run_benchmark(model, D)
                metric_cols = [f"{v} {m}" for v in ("old", "new") for m in ("MRR", "P@10", "R@20")]
                avg = bench[metric_cols].astype(float).mean()
                c1, c2, c3 = st.columns(3)
                c1.metric("MRR", f"{avg['new MRR']:.3f}", f"{avg['new MRR'] - avg['old MRR']:+.3f}")
                c2.metric("P@10", f"{avg['new P@10']:.3f}", f"{avg['new P@10'] - avg['old P@10']:+.3f}")
                c3.metric("R@20", f"{avg['new R@20']:.3f}", f"{avg['new R@20'] - avg['old R@20']:+.3f}")
                st.dataframe(bench, hide_index=True, width="stretch")
                st.download_button("Download benchmark (CSV)", bench.to_csv(index=False).encode("utf-8-sig"),
                                   "benchmark.csv", "text/csv", key="bench_dl")


if __name__ == "__main__":
    main()
