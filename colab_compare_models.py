# -*- coding: utf-8 -*-
"""
مقارنة نماذج التمثيل الدلالي على نفس مقياس التقييم في app.py (يُشغَّل في Google Colab مع GPU).

الخطوات في Colab (Runtime > Change runtime type > T4 GPU):
    !git clone https://github.com/f6pmt92g5t-ops/quran-semantic-search
    %cd quran-semantic-search
    !pip -q install sentence-transformers nltk streamlit openpyxl
    !python colab_compare_models.py

الناتج: جدول لكل نموذج (بحث دلالي فقط + البحث الهجين الكامل كما في الموقع) وملف
model_comparison.csv، وملفات embeddings_<model>.npy لكل نموذج (float16) جاهزة للرفع.

لماذا: قسنا أن تمثيلات النموذج الحالي تتأثر ببنية الجملة أكثر من معناها (أقرب جار
لـ"ومن شر حاسد إذا حسد" هو "ومن شر غاسق إذا وقب")، وأن تقنية التغذية الراجعة
(Rocchio) لم تعالج ذلك. نماذج الاسترجاع الأحدث (multilingual-e5، BGE-M3) دُرِّبت
خصيصًا على مطابقة سؤال قصير بنص، وهو بالضبط استخدام الموقع.
"""
import sys
import time
import logging
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import app  # noqa: E402  (نفس الكود المنشور في الموقع: التطبيع، المعجم، الدمج الهجين، التقييم)
from sentence_transformers import SentenceTransformer  # noqa: E402

# (اسم النموذج، بادئة الاستعلام، بادئة النص). نماذج e5 تتطلب بادئتي "query:" و"passage:".
CANDIDATES = [
    ("current (quran-finetuned-mpnet)", app.MODEL_NAME, "", ""),
    ("multilingual-e5-base", "intfloat/multilingual-e5-base", "query: ", "passage: "),
    ("multilingual-e5-large", "intfloat/multilingual-e5-large", "query: ", "passage: "),
    ("bge-m3", "BAAI/bge-m3", "", ""),
]
if len(sys.argv) > 1:   # تشغيل نماذج محددة فقط: python colab_compare_models.py bge-m3 e5-base
    CANDIDATES = [c for c in CANDIDATES if any(a in c[0] for a in sys.argv[1:])]


def evaluate(D, encode_query):
    rows = []
    for q in app.GOLD:
        gold = app._gold(q)
        qv = encode_query(q)
        cos = D.emb @ qv
        sem = app._verse_max(cos, D.seg_v, D.prim, D.n)
        sem2 = app._verse_max(cos, D.seg_v, D.second, D.n)
        sem = np.where(np.isfinite(sem), sem, sem2)
        sem = np.where(np.isfinite(sem), sem, 0.0).astype(np.float32)
        order = np.argsort(-sem, kind="stable")
        sem_rank = [(int(D.verses.sura[i]), int(D.verses.aya[i])) for i in order]
        hyb = app.semantic_search(q, None, D, sem_cos=sem)
        hyb_rank = list(zip(hyb.sura.astype(int), hyb.aya.astype(int)))
        ms, mh = app._metrics(sem_rank, gold), app._metrics(hyb_rank, gold)
        rows.append({"query": q,
                     "sem MRR": ms["MRR"], "sem P@10": ms["P@10"], "sem R@20": ms["R@20"],
                     "hyb MRR": mh["MRR"], "hyb P@10": mh["P@10"], "hyb R@20": mh["R@20"],
                     "hyb top10": app._top_refs(hyb_rank, gold)})
    return pd.DataFrame(rows)


def main():
    D = app.load_data()
    seg_texts = pd.read_csv("segments.csv")["text"].apply(app.normalize_arabic).tolist()
    summary, details = [], []
    for label, name, qp, pp in CANDIDATES:
        print(f"\n=== {label} ({name})", flush=True)
        t = time.time()
        model = SentenceTransformer(name, device="cuda" if _has_cuda() else "cpu")
        if label.startswith("current"):
            emb = np.load("segment_embeddings.npy").astype(np.float32)
        else:
            emb = model.encode([pp + s for s in seg_texts], batch_size=64, show_progress_bar=True,
                               convert_to_numpy=True).astype(np.float32)
            np.save(f"embeddings_{label}.npy", emb.astype(np.float16))
        D.emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

        def encode_query(q, model=model, qp=qp):
            v = model.encode(qp + app.normalize_arabic(q)).astype(np.float32)
            return v / (np.linalg.norm(v) + 1e-12)

        res = evaluate(D, encode_query)
        res.insert(0, "model", label)
        details.append(res)
        avg = res.drop(columns=["model", "query", "hyb top10"]).mean()
        summary.append({"model": label, **avg.round(3).to_dict(), "seconds": round(time.time() - t)})
        print(pd.DataFrame([summary[-1]]).to_string(index=False), flush=True)
        del model

    print("\n\n######## SUMMARY (average over", len(app.GOLD), "queries) ########")
    print(pd.DataFrame(summary).to_string(index=False))
    pd.concat(details).to_csv("model_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary).to_csv("model_summary.csv", index=False, encoding="utf-8-sig")
    print("\nSaved: model_summary.csv, model_comparison.csv, embeddings_<model>.npy")


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
