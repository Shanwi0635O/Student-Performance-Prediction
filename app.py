"""Streamlit UI.  Run:  streamlit run app.py"""
import matplotlib.pyplot as plt
import streamlit as st

from src.predict import grade_category, load_feature_importance, predict_student

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="centered")

st.title("🎓 Student Performance Predictor")
st.caption("Answer 7 quick questions and the ML model will predict your expected final marks.")

with st.form("student_form"):
    col1, col2 = st.columns(2)
    with col1:
        ct1_marks = st.slider("1. CT-1 marks (out of 30)", 0.0, 30.0, 20.0, 0.5)
        ct2_marks = st.slider("2. CT-2 marks (out of 30)", 0.0, 30.0, 21.0, 0.5)
        assignment_score = st.slider("3. Average assignment score (%)", 0, 100, 70)
        study_hours = st.slider("4. Study hours per day", 0.0, 10.0, 3.0, 0.5)
    with col2:
        participation = st.select_slider(
            "5. Class participation (1 = low, 5 = high)", options=[1, 2, 3, 4, 5], value=3
        )
        sleep_hours = st.slider("6. Sleep hours per night", 3.0, 10.0, 7.0, 0.5)
        distraction_hours = st.slider("7. Phone / social media hours per day", 0.0, 10.0, 3.0, 0.5)
    submitted = st.form_submit_button("Predict my performance", use_container_width=True)

if submitted:
    answers = {
        "CT1_Marks": ct1_marks,
        "CT2_Marks": ct2_marks,
        "Assignment_Score": assignment_score,
        "Study_Hours": study_hours,
        "Participation": participation,
        "Sleep_Hours": sleep_hours,
        "Distraction_Hours": distraction_hours,
    }
    result = predict_student(answers)
    st.divider()
    c1, c2 = st.columns(2)
    c1.metric("Predicted Final Marks", f"{result['marks']} / 100")
    c2.metric("Performance Category", result["category"])
    st.progress(int(result["marks"]))

    tips = []
    if ct2_marks < ct1_marks:
        tips.append("Your CT-2 marks dropped compared to CT-1. Revise the weak topics.")
    if (ct1_marks + ct2_marks) / 2 < 15:
        tips.append("Your class-test average is low. Focus on test preparation.")
    if study_hours < 2.5:
        tips.append("Add more daily study time.")
    if distraction_hours > 4:
        tips.append("Reduce phone / social media time.")
    if sleep_hours < 6:
        tips.append("Aim for 7-8 hours of sleep.")
    if assignment_score < 60:
        tips.append("Improve assignment submissions and quality.")
    st.subheader("Suggestions")
    if tips:
        for t in tips:
            st.write("• " + t)
    else:
        st.success("Your habits look strong. Keep it up!")

st.divider()
st.subheader("What matters most? (Feature importance)")
imp = load_feature_importance().sort_values()          # smallest -> largest, so biggest is on top
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.barh(imp.index, imp.values, color="#2E7D6B")
ax.bar_label(bars, labels=[f"{v:.2f}" for v in imp.values], padding=3)
ax.set_xlim(0, imp.max() * 1.15)                       # axis always starts at 0
ax.set_xlabel("Importance")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
st.pyplot(fig)
