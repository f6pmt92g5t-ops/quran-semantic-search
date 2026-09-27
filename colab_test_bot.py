# -*- coding: utf-8 -*-
"""
بوت اختبار الموقع: يشغّل نفس كود الموقع (app.py) بالنموذج الحقيقي على مئات الاستعلامات،
ويتحقق آليًا من النتائج، ويطلع تقرير Excel تقرأه بدقائق بدل ما تجرب يدويًا.

الخطوات في Google Colab (يفضّل Runtime > Change runtime type > T4 GPU، ويشتغل بدونه لكن أبطأ):
    !git clone https://github.com/f6pmt92g5t-ops/quran-semantic-search
    %cd quran-semantic-search
    !pip -q install sentence-transformers nltk streamlit openpyxl
    !python colab_test_bot.py
    from google.colab import files; files.download("bot_report.xlsx")

وش يختبر:
  1) مواضيع معروفة: لكل موضوع آية أو آيات يجب أن تظهر في أول 10 نتائج (مثل عقوق الوالدين ← 17:23).
  2) مقاطع آيات: يأخذ 5 كلمات من وسط آية عشوائية ويتأكد أن الآية نفسها تطلع في أول 3.
  3) مقياس التقييم (16 موضوعًا) نفس تبويب Evaluation.
  4) مدخلات غريبة (إنجليزي، أرقام، رموز، نص طويل، عامية) — المهم ألا يحدث خطأ.
  5) السرعة: زمن كل بحث.
ثم ورقة "Browse" فيها أول 5 نتائج لكل استعلام لتتصفحها بعينك.
"""
import random
import sys
import time
import logging
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import app  # noqa: E402  نفس كود الموقع حرفيًا

# ---------------------------------------------------------------------------
# 1) مواضيع لها آيات معروفة: يكفي أن تظهر آية واحدة من القائمة في أول 10 نتائج
# ---------------------------------------------------------------------------
EXPECTED = {
    "عقوق الوالدين": "17:23 46:17",
    "بر الوالدين": "17:23 17:24 31:14 46:15 29:8",
    "أحكام الميراث": "4:11 4:12 4:176",
    "الصلاة في السفر": "4:101",
    "الغيبة": "49:12",
    "الوضوء": "5:6",
    "القمار": "5:90 5:91 2:219",
    "الخمر": "5:90 5:91 2:219",
    "تحريم الربا": "2:275 2:276 2:278 3:130",
    "ليلة القدر": "97:1 97:2 97:3",
    "صيام رمضان": "2:183 2:185",
    "الحج": "2:196 3:97 22:27 2:197",
    "الزنا": "17:32 24:2",
    "قتل النفس": "17:33 5:32 4:93 4:29 6:151",
    "كفالة اليتيم": "4:10 93:9 17:34 2:220 4:2 6:152",
    "الشورى": "42:38 3:159",
    "أداء الأمانة": "4:58 23:8 70:32 33:72",
    "الصدق": "9:119 33:70 33:24",
    "الإسراف": "7:31 17:26 17:27 25:67",
    "الدعاء": "2:186 40:60 7:55",
    "الموت": "3:185 21:35 29:57 62:8",
    "الشهداء": "3:169 2:154",
    "الحجاب": "24:31 33:59 33:53",
    "الوصية": "2:180 2:181 2:240 5:106",
    "الحسد": "113:5 2:109 4:54",
    "النفاق": "63:1 4:145 9:67",
    "التوبة": "66:8 39:53 4:17 25:70",
    "الرزق": "11:6 51:58 51:22 65:3",
    "الطلاق": "65:1 2:229 2:230 2:231",
    "الزكاة": "9:60 2:43 2:110 9:103",
    "الكبر والتكبر": "31:18 17:37 16:23 40:35",
    "كظم الغيظ والعفو": "3:134 42:37 42:40 24:22",
    "الشكر": "14:7 2:152 31:12",
    "الله لا إله إلا هو الحي القيوم": "2:255 3:2",
    "خلق الإنسان من طين": "23:12 32:7 6:2 38:71",
    "التوكل على الله": "65:3 3:159 8:2",
    "الصبر على البلاء": "2:155 2:156 2:153",
    "الظلم": "31:13 4:40 10:44",
}
EXPECTED_SURA = {"قصة يوسف": 12, "أصحاب الكهف": 18, "قصة مريم وعيسى": 19}

COLLOQUIAL = ["وش حكم الربا", "كيف أتعامل مع الظلم", "ايش يقول القران عن الصبر", "وش جزاء اللي يعق والديه",
              "كيف اتوب من الذنوب", "هل الخمر حرام", "ابي ايات عن الرزق", "وش قصة اصحاب الفيل",
              "ايات عن الام", "كيف ادعي ربي", "ايات تريح القلب", "ايات عن الموت والقبر"]
EDGE = ["hello", "123", "١٢٣", "؟؟", "!!!", "ما هو", "الله", "و", "في", "ال", "😀", "الصبر😀",
        "<script>alert(1)</script>", "' OR 1=1 --", "الصَّبْرِ", "ٱلصَّلَوٰةَ", "الصلاه", "الـصـبـر",
        "Patience", "الصبر patience", "ب" * 190, "الصبر " * 30, "ﷲ", "بسم الله الرحمن الرحيم"]


def refs_of(df, k):
    """مراجع أول k نتيجة، مع الآيات المطابقة نصًا المدمجة في عمود also."""
    out = []
    for s, a, also in zip(df.sura.head(k), df.aya.head(k), df.get("also", pd.Series([""] * len(df))).head(k)):
        out.append(f"{int(s)}:{int(a)}")
        out += [x.strip() for x in str(also or "").split(",") if x.strip()]
    return out


def timed(fn, *args):
    t = time.time()
    try:
        return fn(*args), time.time() - t, ""
    except Exception as e:  # نسجّل الخطأ ونكمل
        return None, time.time() - t, f"{type(e).__name__}: {e}"


def main():
    print("Loading model and data (first time downloads the model)...", flush=True)
    model = app.SentenceTransformer(app.MODEL_NAME)
    D = app.load_data()
    random.seed(7)

    checks, browse, errors, times = [], [], [], []

    def run(q, kind):
        if kind == "text":
            df, dt, err = timed(app.text_search, q, D)
        else:
            df, dt, err = timed(app.semantic_search, q, model, D)
        times.append((kind, dt))
        if err:
            errors.append({"query": q, "search": kind, "error": err})
        return df

    def add_browse(q, group):
        for kind in ("semantic", "text"):
            df = run(q, kind)
            row = {"group": group, "search": kind, "query": q,
                   "results": 0 if df is None else len(df)}
            if df is not None:
                if kind == "semantic":
                    row["relevant"] = int(df["relevant"].sum())
                for i in range(5):
                    if i < len(df):
                        r = df.iloc[i]
                        row[f"#{i + 1}"] = f"{int(r.sura)}:{int(r.aya)}  {r.text[:90]}"
            browse.append(row)
            if kind == "semantic":
                yield df

    # 1) known topics
    print("1/5 known topics...", flush=True)
    for q, exp in EXPECTED.items():
        df = next(add_browse(q, "topic"))
        exp_set = set(exp.split())
        top = refs_of(df, 10) if df is not None else []
        hit = next((i for i, r in enumerate(top, 1) if r in exp_set), None)
        checks.append({"test": "topic", "query": q, "expected (any of)": exp,
                       "PASS": hit is not None, "found at": hit or "-", "top 10": " ".join(top[:10])})
    for q, sura in EXPECTED_SURA.items():
        df = next(add_browse(q, "topic"))
        top = df.sura.head(10).astype(int).tolist() if df is not None else []
        n = sum(s == sura for s in top)
        checks.append({"test": "story", "query": q, "expected (any of)": f"surah {sura}",
                       "PASS": n >= 3, "found at": f"{n}/10 from surah {sura}", "top 10": " ".join(map(str, top))})

    # 2) verse fragments: the verse itself should come back in the top 3
    print("2/5 verse fragments...", flush=True)
    # بالإملاء المعتاد كما يكتبه المستخدم (الصلاة لا الصلوة)
    plain = D.verses.text.apply(lambda t: " ".join(app.uthmani_plain(t).split()))
    long_ones = [i for i, t in enumerate(plain) if len(t.split()) >= 9]
    frags = []
    for i in random.sample(long_ones, 400):
        w = plain[i].split()
        start = random.randrange(1, len(w) - 5)
        q = " ".join(w[start:start + 5])
        # مقطع أغلبه حروف وأدوات ("ان ذالك فى كتاب ان") يصلح لآيات كثيرة — نتخطاه
        if sum(app.lex_key(x) not in app.STOP_KEYS for x in q.split()) >= 3:
            frags.append((i, q))
    for i, q in frags[:60]:
        ref = f"{int(D.verses.sura[i])}:{int(D.verses.aya[i])}"
        for kind in ("semantic", "text"):
            df = run(q, kind)
            top = refs_of(df, 3 if kind == "semantic" else 10) if df is not None else []
            checks.append({"test": f"fragment ({kind}, top {3 if kind == 'semantic' else 10})", "query": q,
                           "expected (any of)": ref, "PASS": ref in top,
                           "found at": (top.index(ref) + 1) if ref in top else "-", "top 10": " ".join(top)})

    # 3) colloquial questions and 4) odd inputs — only need no errors (plus a look in Browse)
    print("3/5 colloquial questions...", flush=True)
    for q in COLLOQUIAL:
        list(add_browse(q, "colloquial"))
    print("4/5 odd inputs...", flush=True)
    for q in EDGE:
        list(add_browse(q, "odd input"))

    # 5) benchmark (same as the Evaluation tab)
    print("5/5 benchmark...", flush=True)
    bench = app.run_benchmark(model, D)

    checks = pd.DataFrame(checks)
    t = pd.DataFrame(times, columns=["search", "seconds"])
    summary = []
    for name, g in checks.groupby("test", sort=False):
        summary.append({"item": name, "value": f"{int(g.PASS.sum())}/{len(g)} passed"})
    metric_cols = [f"new {m}" for m in ("MRR", "P@10", "R@20")]
    for m in metric_cols:
        summary.append({"item": f"benchmark {m}", "value": f"{bench[m].astype(float).mean():.3f}"})
    summary.append({"item": "errors", "value": str(len(errors))})
    for kind, g in t.groupby("search"):
        summary.append({"item": f"{kind} search time (avg / max)",
                        "value": f"{g.seconds.mean():.2f}s / {g.seconds.max():.2f}s over {len(g)} searches"})
    summary = pd.DataFrame(summary)

    print("\n" + summary.to_string(index=False))
    failed = checks[~checks.PASS]
    if len(failed):
        print("\nFAILED checks (look at these on the site):")
        print(failed[["test", "query", "expected (any of)", "top 10"]].head(40).to_string(index=False))

    with pd.ExcelWriter("bot_report.xlsx", engine="openpyxl") as xw:
        summary.to_excel(xw, sheet_name="Summary", index=False)
        checks.sort_values("PASS").to_excel(xw, sheet_name="Checks", index=False)
        pd.DataFrame(browse).to_excel(xw, sheet_name="Browse", index=False)
        bench.to_excel(xw, sheet_name="Benchmark", index=False)
        pd.DataFrame(errors or [{"query": "", "search": "", "error": "no errors"}]).to_excel(xw, sheet_name="Errors", index=False)
    print("\nSaved bot_report.xlsx")


if __name__ == "__main__":
    main()
