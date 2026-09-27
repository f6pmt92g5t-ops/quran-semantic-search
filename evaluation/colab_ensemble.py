# -*- coding: utf-8 -*-
"""
دمج نموذجين دلاليين (النموذج الحالي + نموذج ثانٍ) وقياس أثر الدمج على نفس مقياس التقييم.
يُشغَّل في Colab بعد colab_compare_models.py (يعيد استخدام ملف embeddings_<model>.npy المحفوظ).

    !python evaluation/colab_ensemble.py bge-m3            # أو: multilingual-e5-large

لماذا: في المقارنة لم يتفوق أي نموذج جديد على الحالي إجمالًا، لكن BGE-M3 وجد آيات
"صعبة" لا يجدها الحالي (2:155، 47:31 في الصبر على البلاء؛ 19:14، 31:14 في عقوق
الوالدين)، والحالي أدق في الظلم والتفكر — أي أنهما يكمّلان بعضهما.
طريقة الدمج: لكل استعلام تُوحَّد درجات النموذج الثاني إلى نفس متوسط وانحراف درجات
النموذج الحالي عبر كل الآيات (z-score)، ثم sem = (1-w)*الحالي + w*الثاني.
التوحيد ضروري لأن مقاييس التشابه تختلف بين النماذج (e5 مثلًا درجاته كلها 0.7-0.9).
"""
import os
import sys

# يُشغَّل من مجلد المستودع الرئيسي: python evaluation/<script>.py
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
TEST_DATA = os.path.join("evaluation", "test_data")

import sys
import logging
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import app  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

SECOND = {
    "bge-m3": ("BAAI/bge-m3", ""),
    "multilingual-e5-large": ("intfloat/multilingual-e5-large", "query: "),
    "multilingual-e5-base": ("intfloat/multilingual-e5-base", "query: "),
}
WEIGHTS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]


def verse_scores(D, emb, qv):
    cos = emb @ qv
    s = app._verse_max(cos, D.seg_v, D.prim, D.n)
    s2 = app._verse_max(cos, D.seg_v, D.second, D.n)
    s = np.where(np.isfinite(s), s, s2)
    return np.where(np.isfinite(s), s, np.nan)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "bge-m3"
    hub, qp = SECOND[name]
    D = app.load_data()
    e1 = np.load("segment_embeddings.npy").astype(np.float32)
    e2 = np.load(f"embeddings_{name}.npy").astype(np.float32)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 /= np.linalg.norm(e2, axis=1, keepdims=True)
    m1 = SentenceTransformer(app.MODEL_NAME)
    m2 = SentenceTransformer(hub)

    per_query = {}
    for q in app.GOLD:
        nq = app.normalize_arabic(q)
        v1 = m1.encode(nq).astype(np.float32); v1 /= np.linalg.norm(v1)
        v2 = m2.encode(qp + nq).astype(np.float32); v2 /= np.linalg.norm(v2)
        per_query[q] = (verse_scores(D, e1, v1), verse_scores(D, e2, v2))

    summary, details = [], []
    for w in WEIGHTS:
        rows = []
        for q, (s1, s2) in per_query.items():
            ok = np.isfinite(s1) & np.isfinite(s2)
            z2 = np.zeros_like(s1)
            z2[ok] = (s2[ok] - s2[ok].mean()) / (s2[ok].std() + 1e-9) * s1[ok].std() + s1[ok].mean()
            sem = np.where(ok, (1 - w) * s1 + w * z2, 0.0).astype(np.float32)
            gold = app._gold(q)
            hyb = app.semantic_search(q, None, D, sem_cos=sem)
            rank = list(zip(hyb.sura.astype(int), hyb.aya.astype(int)))
            m = app._metrics(rank, gold)
            rows.append({"weight": w, "query": q, "MRR": m["MRR"], "P@10": m["P@10"], "R@20": m["R@20"],
                         "top10": app._top_refs(rank, gold)})
        df = pd.DataFrame(rows)
        details.append(df)
        summary.append({"weight of " + name: w, **df[["MRR", "P@10", "R@20"]].mean().round(3).to_dict()})
        print(summary[-1], flush=True)

    print("\n######## ENSEMBLE SUMMARY (0 = current model only, 1 =", name, "only) ########")
    print(pd.DataFrame(summary).to_string(index=False))
    pd.DataFrame(summary).to_csv("ensemble_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(details).to_csv("ensemble_details.csv", index=False, encoding="utf-8-sig")
    print("\nSaved: ensemble_summary.csv, ensemble_details.csv")


if __name__ == "__main__":
    main()
