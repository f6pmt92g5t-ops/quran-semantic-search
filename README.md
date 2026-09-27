# Quran Semantic Search

A search engine for the Holy Quran that finds verses by **meaning**, not only by exact wording.

**Open it:** https://f6pmt92g5t-ops.github.io/quran-semantic-search/
Graduation project, B.Sc. Data Science and Analytics, Onaizah Colleges (1447–1448 / 2025–2026).

محرك بحث في القرآن الكريم يجد الآيات بالمعنى لا بالكلمة فقط: بحث نصي بالجذر، وبحث دلالي هجين
(نموذج تضمين مدرَّب على بيانات قرآنية + مطابقة صرفية من مدوّنة القرآن الكريم).

## What the app does

One search box with suggestions as you type, ready-made topic chips, and two modes:

| Mode | What it does |
|---|---|
| **By meaning and topic** (default) | Hybrid ranking: `score = S(v) + 0.7 × L(v)` — semantic similarity from a fine-tuned embedding model plus a graded word-match score from the Quranic Arabic Corpus morphology, plus a topical index of core verses. |
| **By word and root** | Every verse containing a word from the same Arabic root as the query word. Exact words and close derivatives are listed first. |

Each verse card has: recitation (Mishary Alafasy, EveryAyah), copy, share, Tafsir al-Muyassar, the surrounding
verses, the three most similar verses (by embedding), a link to the mushaf page on quran.com, and — in meaning
mode — a line from the tafsir showing why the verse matched. Clicking any word shows its root, lemma and how many
verses use that root, with a link to all of them. Also: Arabic/English interface with the Sahih International
translation (`?lang=en`), voice search, "did you mean" for typos, a chart of where a root appears across surahs,
surah/juz filters, shareable links (`?q=...&m=word`) and Excel export. The developer benchmark opens with `?dev=1`.

**User ratings (optional):** 👍/👎 under each result are saved to a GitHub Gist when the app has two Streamlit
secrets, `GIST_TOKEN` (a token with the *gist* scope) and `GIST_ID`. Without them the buttons are hidden.
The summary appears in the `?dev=1` panel.

**Tested and not adopted:** adding a tafsir-match signal to the ranking (weight 0.1–0.5) raised R@20 slightly but
lowered P@10 and the known-topic MRR, so `TAFSIR_WEIGHT = 0`.

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
| v3: hybrid + Quranic Arabic Corpus | 0.953 | 0.788 | 0.519 |
| **v3 + topical index (final, deployed)** | **1.000** | **0.812** | **0.524** |

Details, per-query results and the comparison with multilingual-e5 and BGE-M3 are in Chapter Five of the report.

## Files

| File | Purpose |
|---|---|
| `app.py` | The whole application (Streamlit). |
| `verses.csv` | 6,236 verses: `sura, aya, text`. |
| `segments.csv` | 10,675 Tajweed segments: `sura, aya, part_num, text`. |
| `segment_embeddings.npy` | Embeddings of the segments, shape (10675, 768), float16. Must match `segments.csv` row for row. |
| `translation_en.csv` | English translation, Sahih International: `sura, aya, english`. |
| `docs/` | Landing page for GitHub Pages (link preview, QR code, Google indexing). |
| `tafsir_muyassar.csv` | Tafsir al-Muyassar (King Fahd Complex), one row per verse: `sura, aya, tafsir`. |
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
