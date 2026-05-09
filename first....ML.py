import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="Health Risk Predictor", page_icon="🩺", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>🩺 Health Risk Prediction App</h1>
    <p style='text-align: center; font-size:18px;'>
    Age & BMI দিয়ে জানুন আপনি Healthy নাকি At Risk
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------- DATA ----------------
data = {
    'Age': [25, 30, 45, 50, 22, 35, 60, 55, 20, 40],
    'BMI': [22, 24, 30, 32, 21, 28, 35, 31, 19, 29],
    'Status': [0, 0, 1, 1, 0, 1, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df[['Age', 'BMI']]
y = df['Status']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_scaled, y)

# ---------------- INPUT UI ----------------
st.subheader("📥 আপনার তথ্য দিন")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 15, 70, 25)

with col2:
    bmi = st.slider("BMI", 15, 40, 22)

# ---------------- PREDICTION ----------------
new_person = np.array([[age, bmi]])
new_person_scaled = scaler.transform(new_person)
prediction = knn.predict(new_person_scaled)

st.markdown("---")

# ---------------- RESULT CARD ----------------
if prediction[0] == 0:
    st.success("🟢 Result: You are Healthy")
    st.balloons()
else:
    st.error("🔴 Result: You are At Risk")

# ---------------- METRICS ----------------
st.subheader("📊 Your Info Summary")

col1, col2 = st.columns(2)

col1.metric("Age", age)
col2.metric("BMI", bmi)

# ---------------- GRAPH ----------------
st.subheader("📈 Visualization")

fig, ax = plt.subplots(figsize=(6,4))

ax.scatter(df[df['Status']==0]['Age'],
           df[df['Status']==0]['BMI'],
           color='green', label='Healthy', s=100)

ax.scatter(df[df['Status']==1]['Age'],
           df[df['Status']==1]['BMI'],
           color='red', label='At Risk', s=100)

ax.scatter(age, bmi, color='blue', marker='*', s=200, label='You')

ax.set_xlabel("Age")
ax.set_ylabel("BMI")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;'>Made with ❤️ using Streamlit</p>",
    unsafe_allow_html=True
)