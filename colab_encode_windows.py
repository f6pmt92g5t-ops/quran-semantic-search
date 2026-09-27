# -*- coding: utf-8 -*-
"""
يرمّز "النوافذ" (مقاطع من 6 كلمات داخل الأجزاء الطويلة، انظر make_windows في app.py)
بالنموذج الحقيقي، ويحفظها في window_embeddings.npy (حوالي 22 ميجا).

في Google Colab مع T4 GPU (دقيقتان تقريبًا):
    !git clone https://github.com/f6pmt92g5t-ops/quran-semantic-search
    %cd quran-semantic-search
    !pip -q install sentence-transformers nltk streamlit openpyxl
    !python colab_encode_windows.py
    from google.colab import files; files.download("window_embeddings.npy")
"""
import logging
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import app  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402


def main():
    windows = app.make_windows(pd.read_csv("segments.csv"))
    texts = windows["text"].apply(app.normalize_arabic).tolist()   # نفس تطبيع الأجزاء
    model = SentenceTransformer(app.MODEL_NAME)
    vecs = model.encode(texts, batch_size=128, show_progress_bar=True).astype(np.float16)
    np.save("window_embeddings.npy", vecs)
    print(f"Saved window_embeddings.npy: {vecs.shape}")


if __name__ == "__main__":
    main()
