# Evaluation — ملفات التقييم والاختبار

These files are **not needed to run the website**. They reproduce the tests reported in Chapters Four and Five.
هذه الملفات **لا يحتاجها الموقع ليعمل**؛ هي لإعادة الاختبارات والنتائج المذكورة في الفصلين الرابع والخامس من التقرير.

## How to run in Google Colab — طريقة التشغيل

```
!git clone https://github.com/f6pmt92g5t-ops/quran-semantic-search
%cd quran-semantic-search
!pip -q install sentence-transformers nltk streamlit openpyxl
!python evaluation/<script>.py
```

Use a T4 GPU runtime (Runtime → Change runtime type) for the model comparison. Outputs are written to the repository folder.
يفضّل تشغيل Colab على T4 GPU. الملفات الناتجة تُحفظ في المجلد الرئيسي للمستودع.

## Scripts — السكربتات

| Script | What it does | Output | Report |
|---|---|---|---|
| `colab_test_bot.py` | Runs the app code with the real model on ~200 automatic checks: known topics, verse fragments, colloquial and odd inputs, the benchmark, and speed. | `bot_report.xlsx` | 4.3 |
| `colab_encode_queries.py` | Encodes all test queries with the real model into one small file, so the full app can be tested offline without downloading the model. | `query_vectors.npz` | 4.2.2 Step 19 |
| `colab_compare_models.py` | Compares the fine-tuned model with multilingual-e5-base, multilingual-e5-large and BGE-M3 on the benchmark. | `model_summary.csv`, `model_comparison.csv` | Table 5.5 |
| `colab_ensemble.py` | Tests mixing the fine-tuned model with a second model (run after `colab_compare_models.py`). | `ensemble_summary.csv` | 5.3.3 |

## Test data — بيانات الاختبار (`test_data/`)

| File | Content |
|---|---|
| `test_queries.txt` | 992 topic queries in several phrasings (first review set). |
| `test_queries2.txt` | 835 unseen queries (second review set). |
| `test_queries3.txt` | 292 further queries prepared for a third round. |
| `test_queries_en.txt` | 60 free English queries (not yet evaluated). |
| `test_fragments.tsv` | 300 verse fragments with the verse they come from (`fragment<TAB>sura:aya`). |
| `test_expected2.tsv` | 80 well-known verse quotations with their reference. |

The 16-query benchmark itself (gold relevant verses) is inside `app.py` (`GOLD`), and runs in the website's developer view: add `?dev=1` to the address.
مقياس التقييم (16 موضوعًا) موجود داخل `app.py` باسم `GOLD`، ويُشغَّل من لوحة المطورين في الموقع بإضافة `?dev=1` لآخر الرابط.
