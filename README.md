# 🎓 Student Performance Prediction (PBL – Machine Learning)

A machine learning system that predicts a student's expected final marks from 7 academic and behavioural factors, shows which factors matter most, and provides a simple web UI.

## Features
- Data cleaning (duplicates, missing values) and exploratory data analysis
- 3 models compared: Linear Regression, Random Forest, Gradient Boosting
- Metrics: MAE, RMSE, R²
- Feature-importance analysis
- Streamlit web app: answer 7 questions (CT-1, CT-2, assignments, study hours, participation, sleep, phone use) and get predicted marks, a category and suggestions

## Project Structure
```
student-performance-predictor/
├── Student_Performance_Prediction.ipynb  # full notebook (VS Code / Colab)
├── app.py                      # Streamlit UI (run this)
├── requirements.txt
├── README.md
├── data/
│   └── student_performance.csv # dataset
├── src/
│   ├── generate_dataset.py     # creates the sample dataset
│   ├── eda.py                  # exploratory data analysis charts
│   ├── train.py                # clean, split, train, evaluate, save model
│   └── predict.py              # loads model, predicts for one student
├── models/
│   └── model.joblib            # saved trained model
├── outputs/                    # charts + metrics.json (used in the PPT)
└── docs/
    ├── PPT_outline.md          # slide-by-slide content
    ├── PROJECT_DOCUMENTATION.md
    └── VIVA_NOTES.md           # simple explanations for questions
```

## How to Run
**Option A: Notebook (VS Code or Google Colab):** open `Student_Performance_Prediction.ipynb` and Run All. On Colab, upload the notebook and, when asked, `data/student_performance.csv`.

**Option B: Scripts + web app:**
```bash
pip install -r requirements.txt
python src/generate_dataset.py   # skip if you use a Kaggle CSV
python src/eda.py                # EDA charts -> outputs/
python src/train.py              # trains + saves model -> models/
streamlit run app.py             # opens the UI in browser
```

## Using a real Kaggle dataset (optional)
Download a student-performance CSV (e.g. "Student Performance Factors" by lainguyn123 on Kaggle),
save it as `data/student_performance.csv`, and rename its columns in the CSV to:
`CT1_Marks, CT2_Marks, Assignment_Score, Study_Hours, Participation, Sleep_Hours, Distraction_Hours, Final_Marks`
(or edit the feature list in `train.py` and the form in `app.py` to match your columns).

## Tech Stack
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Streamlit, joblib

## Subject
Machine Learning – 5th Semester PBL
