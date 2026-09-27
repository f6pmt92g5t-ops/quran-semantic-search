# -*- coding: utf-8 -*-
"""
يحوّل قائمة استعلامات الاختبار إلى متجهات بالنموذج الحقيقي، في ملف صغير واحد
(query_vectors.npz، حوالي 2 ميجا). بهذا الملف يقدر المطوّر يشغّل الموقع كاملًا
بنتائجه الحقيقية على جهاز لا يصل إلى Hugging Face، ويختبر ويصلح بدون Colab.

في Google Colab (دقيقة واحدة تقريبًا):
    !git clone https://github.com/f6pmt92g5t-ops/quran-semantic-search
    %cd quran-semantic-search
    !pip -q install sentence-transformers nltk streamlit openpyxl
    !python colab_encode_queries.py
    from google.colab import files; files.download("query_vectors.npz")

الاستعلامات من test_queries.txt (سطر لكل استعلام) و test_fragments.tsv (مقطع<TAB>آية).
"""
import logging
import warnings

import numpy as np

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import app  # noqa: E402  نفس التطبيع ونفس النموذج المستخدم في الموقع
from sentence_transformers import SentenceTransformer  # noqa: E402


def main():
    queries = []
    for name in ("test_queries.txt", "test_queries2.txt", "test_queries3.txt"):
        try:
            queries += [q.strip() for q in open(name, encoding="utf-8") if q.strip()]
        except FileNotFoundError:
            pass
    queries += [line.split("\t")[0].strip() for line in open("test_fragments.tsv", encoding="utf-8") if line.strip()]
    texts = list(dict.fromkeys(app.normalize_arabic(q) for q in queries))   # نفس ما يفعله الموقع قبل الترميز
    model = SentenceTransformer(app.MODEL_NAME)
    vecs = model.encode(texts, batch_size=64, show_progress_bar=True).astype(np.float16)
    np.savez_compressed("query_vectors.npz", texts=np.array(texts), vectors=vecs)
    print(f"Saved query_vectors.npz: {len(texts)} queries, shape {vecs.shape}")


if __name__ == "__main__":
    main()
