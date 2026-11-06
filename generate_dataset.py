"""Creates a realistic student-performance dataset (1000 students).

Use this ONLY if you don't have a Kaggle dataset. It deliberately adds a few
missing values and duplicate rows so the cleaning step in train.py has real work to do.
CT-1 and CT-2 are class-test marks out of 30.
Run:  python src/generate_dataset.py
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 1000
CT_MAX = 30

ability = rng.normal(0, 1, N)   # hidden "academic ability" that links all marks together

ct1_pct = np.clip(62 + 13 * ability + rng.normal(0, 7, N), 10, 100)
ct2_pct = np.clip(64 + 13 * ability + rng.normal(0, 7, N), 10, 100)
assignment_score = np.clip(68 + 10 * ability + rng.normal(0, 8, N), 10, 100)   # % avg
study_hours = np.clip(3 + 0.4 * ability + rng.normal(0, 1.4, N), 0, 10)        # hours/day
participation = np.clip(np.round(3 + 0.6 * ability + rng.normal(0, 1.0, N)), 1, 5).astype(int)
sleep_hours = np.clip(rng.normal(6.8, 1.2, N), 3, 10)                          # hours/night
distraction_hours = np.clip(rng.normal(3.5, 1.8, N), 0, 10)                    # phone/social media

final_marks = (
    0.25 * ct1_pct
    + 0.30 * ct2_pct
    + 0.15 * assignment_score
    + 2.2 * study_hours
    + 1.6 * participation
    - 0.9 * distraction_hours
    - 1.2 * np.abs(sleep_hours - 7.5)       # too little OR too much sleep hurts
    + rng.normal(0, 4, N)                   # real-world randomness
    + 5
)
final_marks = np.clip(final_marks, 0, 100)

df = pd.DataFrame({
    "CT1_Marks": (ct1_pct * CT_MAX / 100).round(1),
    "CT2_Marks": (ct2_pct * CT_MAX / 100).round(1),
    "Assignment_Score": assignment_score.round(1),
    "Study_Hours": study_hours.round(1),
    "Participation": participation,
    "Sleep_Hours": sleep_hours.round(1),
    "Distraction_Hours": distraction_hours.round(1),
    "Final_Marks": final_marks.round(1),
})

# Make it "messy" like real data: missing values + duplicates
for col in ["CT1_Marks", "Study_Hours", "Sleep_Hours"]:
    df.loc[rng.choice(N, 15, replace=False), col] = np.nan
df = pd.concat([df, df.sample(10, random_state=1)], ignore_index=True)

df.to_csv("data/student_performance.csv", index=False)
print("Saved data/student_performance.csv", df.shape)
