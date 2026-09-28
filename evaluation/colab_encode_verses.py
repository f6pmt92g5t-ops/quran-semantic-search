# -*- coding: utf-8 -*-
"""
يحوّل كل آية كاملة (6236 آية) إلى متجه بالنموذج المدرَّب نفسه، ويحفظها في verse_embeddings.npy
(حوالي 9 ميجا). يستخدمها الموقع في وحدة الاسترجاع «آية كاملة» و«الاثنان معًا».

في Google Colab (دقيقتان تقريبًا على T4 GPU):
    !git clone https://github.com/f6pmt92g5t-ops/quran-semantic-search
    %cd quran-semantic-search
    !pip -q install sentence-transformers nltk streamlit openpyxl
    !python evaluation/colab_encode_verses.py
    from google.colab import files; files.download("verse_embeddings.npy")
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import logging  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import app  # noqa: E402  نفس النص (بدون البسملة الملتصقة) ونفس التطبيع ونفس النموذج
from sentence_transformers import SentenceTransformer  # noqa: E402


def main():
    D = app.load_data()
    texts = [app.normalize_arabic(t) for t in D.verses.text]
    model = SentenceTransformer(app.MODEL_NAME)
    vecs = model.encode(texts, batch_size=64, show_progress_bar=True).astype(np.float16)
    assert vecs.shape == (6236, 768)
    np.save("verse_embeddings.npy", vecs)
    print("Saved verse_embeddings.npy", vecs.shape)


if __name__ == "__main__":
    main()
