# ============================================================================
# دمج (Ensemble) النموذجين: mpnet العام + Arabic-mpnet المتخصص
# لكل آية: نأخذ أفضل درجة تشابه بين النموذجين (مو حفظ إجابات، قاعدة عامة
# تنطبق على أي استعلام جديد تلقائيًا)
# ============================================================================

import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

ARABIC_DIACRITICS = re.compile(r"[ً-ٰٟۖ-ۭ]")
def remove_diacritics(text):
    text = re.sub(r"ٰ", "ا", text)
    return ARABIC_DIACRITICS.sub("", text)
def normalize_arabic(text):
    text = remove_diacritics(text)
    text = re.sub(r"[ؐ-ؚۖ-ۜ۟-۪ۨ-ۭ]", "", text)
    text = re.sub(r"[إأآٱ]", "ا", text)
    text = re.sub(r"ـ", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

segments_df = pd.read_csv("segments.csv")
texts_normalized = segments_df["text"].apply(normalize_arabic).tolist()

print("جاري تحميل النموذجين (هذا بياخذ وقت ومساحة ذاكرة أكبر)...")
model_mpnet = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
model_arabic = SentenceTransformer("Omartificial-Intelligence-Space/Arabic-mpnet-base-all-nli-triplet")

print("جاري توليد embeddings بالنموذج الأول (mpnet)...")
emb_mpnet = model_mpnet.encode(texts_normalized, show_progress_bar=True, batch_size=32).astype(np.float32)
np.save("segment_embeddings_mpnet_v2.npy", emb_mpnet)

print("جاري توليد embeddings بالنموذج الثاني (Arabic)...")
emb_arabic = model_arabic.encode(texts_normalized, show_progress_bar=True, batch_size=32).astype(np.float32)
np.save("segment_embeddings_arabic_v2.npy", emb_arabic)


def check_ensemble(query, target_sura, target_aya, top_n_show=5):
    q_norm = normalize_arabic(query)
    q_mpnet = model_mpnet.encode(q_norm)
    q_arabic = model_arabic.encode(q_norm)

    s_mpnet = util.cos_sim(q_mpnet, emb_mpnet)[0].numpy()
    s_arabic = util.cos_sim(q_arabic, emb_arabic)[0].numpy()

    # الدمج: أفضل درجة من النموذجين لكل جزء (قاعدة عامة، بدون أي معرفة
    # مسبقة بالسؤال أو إجابته)
    combined = np.maximum(s_mpnet, s_arabic)
    which_model = np.where(s_mpnet >= s_arabic, "mpnet", "arabic")

    ranked = combined.argsort()[::-1]
    best_rank = None
    for rank, idx in enumerate(ranked, start=1):
        row = segments_df.iloc[idx]
        if row['sura'] == target_sura and row['aya'] == target_aya:
            best_rank = (rank, combined[idx])
            break

    print(f"\n{'='*70}\nالاستعلام: '{query}'   (الهدف: {target_sura}:{target_aya})")
    if best_rank:
        r, s = best_rank
        flag = "✅ ممتاز" if r <= 10 else ("⚠️ ضعيف" if r <= 200 else "❌ شبه مفقود")
        print(f"  رتبة الآية الصحيحة (مدمجة): {r}  (score={s:.3f})  {flag}")
    else:
        print("  ⚠️ الهدف مو موجود")

    print(f"  أفضل {top_n_show} نتائج فعليًا (مدمجة):")
    for rank, idx in enumerate(ranked[:top_n_show], start=1):
        row = segments_df.iloc[idx]
        print(f"    {rank}. ({row['sura']}:{row['aya']}) score={combined[idx]:.3f} [{which_model[idx]}]  {row['text']}")


print("\n\n" + "#"*70)
print("# نتائج الدمج (Ensemble)")
print("#"*70)

check_ensemble("النهي عن الغيبة والنميمة", 49, 12)
check_ensemble("حرية الاعتقاد وعدم الإكراه في الدين", 2, 256)
check_ensemble("الصبر على البلاء والمصائب", 2, 155)
check_ensemble("جزاء من يقتل نفسا بغير حق", 5, 32)
check_ensemble("التوكل على الله في طلب الرزق", 65, 3)
check_ensemble("التفاخر بالمال والأولاد بدون فائدة حقيقية", 18, 46)
check_ensemble("والنجم", 53, 1)
check_ensemble("والنجم إذا هوى", 53, 1)
