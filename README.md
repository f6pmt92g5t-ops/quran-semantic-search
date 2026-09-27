# Quran Semantic Search

A search engine for the Holy Quran that finds verses by **meaning**, not only by exact wording.
Graduation project, B.Sc. Data Science and Analytics, Onaizah Colleges (1447–1448 / 2025–2026).

محرك بحث في القرآن الكريم يجد الآيات بالمعنى لا بالكلمة فقط: بحث نصي بالجذر، وبحث دلالي هجين
(نموذج تضمين مدرَّب على بيانات قرآنية + مطابقة صرفية من مدوّنة القرآن الكريم).

## What the app does

| Tab | What it does |
|---|---|
| **Text Search** | Every verse containing a word from the same Arabic root as the query word. Exact words and close derivatives are listed first. |
| **Semantic Search** | Hybrid ranking: `score = S(v) + 0.7 × L(v)` — semantic similarity from a fine-tuned embedding model plus a graded word-match score from the Quranic Arabic Corpus morphology. Shows the matched words for each verse. |
| **Evaluation** (sidebar checkbox) | Runs the 16-query benchmark and compares the previous ranking (v2, ISRI stemmer) with the final one (v3). |

Every tab can export the full ranked list to Excel.

## How it works

1. **Data** — 6,236 verses (Tanzil.net, Uthmani script) split into 10,675 segments at the Tajweed pause marks.
2. **Semantic signal S(v)** — segments are encoded with
   [`Amer-Surur1/quran-finetuned-mpnet`](https://huggingface.co/Amer-Surur1/quran-finetuned-mpnet)
   (`paraphrase-multilingual-mpnet-base-v2` fine-tuned on 16,929 verse–tafsir pairs, 768 dimensions).
   A verse gets the cosine similarity of its best segment.
3. **Lexical signal L(v)** — each query word is matched against the verified root and lemma of every
   word in the Quran (Quranic Arabic Corpus v0.4): same lemma = 1.0, close derivative = 0.65,
   same root only = 0.15, weighted by IDF and by how many query words the verse covers.
4. **Query handling** — stopwords, question words (كيف، وش…), framing words (أحكام، تحريم…),
   Uthmani ↔ standard spelling (الصلوٰة = الصلاة), and a small concept dictionary
   (عقوق → أُفّ، الميراث → السدس/الثمن/الربع، السفر → الضرب في الأرض …).
5. **Clean ranking** — Basmala removed, formulaic phrases (والله غفور رحيم…) down-weighted,
   identical verses merged, relevance cut-off.

## Results (16-query pooled benchmark)

| Ranking | MRR | P@10 | R@20 |
|---|---|---|---|
| Semantic similarity only | 0.622 | 0.362 | 0.180 |
| v2: hybrid + ISRI stemmer | 0.762 | 0.619 | 0.374 |
| **v3: hybrid + Quranic Arabic Corpus (final)** | **0.953** | **0.788** | **0.519** |

Details, per-query results and the comparison with multilingual-e5 and BGE-M3 are in Chapter Five of the report.

## Files

| File | Purpose |
|---|---|
| `app.py` | The whole application (Streamlit). |
| `verses.csv` | 6,236 verses: `sura, aya, text`. |
| `segments.csv` | 10,675 Tajweed segments: `sura, aya, part_num, text`. |
| `segment_embeddings.npy` | Embeddings of the segments, shape (10675, 768), float16. Must match `segments.csv` row for row. |
| `quran-morphology.txt` | Quranic Arabic Corpus morphology v0.4 (Dukes & Habash 2010, GNU GPL; Arabic-script edition by [mustafa0x/quran-morphology](https://github.com/mustafa0x/quran-morphology)). |
| `requirements.txt` | Python packages. |
| `colab_compare_models.py` | Colab script: compares embedding models on the benchmark. |
| `colab_ensemble.py` | Colab script: tests mixing the current model with a second model. |
| `colab_test_bot.py` | Colab test bot: runs the app code with the real model on ~200 checks (known topics, verse fragments, colloquial and odd inputs, benchmark, speed) and writes `bot_report.xlsx`. |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first start downloads the model from Hugging Face (about 1 GB) and takes a minute; later queries
take a fraction of a second.

## Deploy (Streamlit Community Cloud)

Push the repository to GitHub, then on share.streamlit.io choose **New app** → this repository →
branch `main` → main file `app.py`. No secrets or extra settings are needed.

## Sources

- Quranic text: [Tanzil.net](https://tanzil.net) (Uthmani).
- K. Dukes and N. Habash, "Morphological Annotation of Quranic Arabic," LREC 2010.
- N. Reimers and I. Gurevych, "Sentence-BERT," EMNLP 2019.
