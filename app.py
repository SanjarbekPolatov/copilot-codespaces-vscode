import streamlit as st
import pickle
import numpy as np
import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
import pandas as pd
from PIL import Image


# Modelni yuklash
with open('saraton.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit interfeysi
st.title("Saraton Kasalligi Bashorat Qiluvchi Model")
st.write("Yuqorida yuklangan modeldan foydalanib, saraton kasalligi ehtimolini bashorat qilishingiz mumkin.")

# Foydalanuvchi kiritishi uchun maydonlar
st.sidebar.header("Ma'lumotlarni kiriting:")
age = st.sidebar.number_input("Yosh:", min_value=0, max_value=120, value=30, step=1)
smoking = st.sidebar.selectbox("Chekish odati:", options=["Yo'q", "Ha"])
alcohol = st.sidebar.selectbox("Spirtli ichimlik iste'moli:", options=["Yo'q", "Ha"])
diet = st.sidebar.selectbox("Parhez turi:", options=["Sog'lom", "Nosog'lom"])
exercise = st.sidebar.selectbox("Jismoniy faollik:", options=["Kam", "O'rtacha", "Ko'p"])

# Ma'lumotlarni qayta ishlash
smoking = 1 if smoking == "Ha" else 0
alcohol = 1 if alcohol == "Ha" else 0
diet = 1 if diet == "Sog'lom" else 0
exercise_map = {"Kam": 0, "O'rtacha": 1, "Ko'p": 2}
exercise = exercise_map[exercise]

# Kirish ma'lumotlari
features = np.array([[age, smoking, alcohol, diet, exercise]])

# Bashorat qilish
if st.button("Bashorat qilish"):
    prediction = model.predict(features)
    probability = model.predict_proba(features)
    if prediction[0] == 1:
        st.error(f"Bashorat: Saraton kasalligi aniqlangan! Ehtimollik: {probability[0][1]:.2f}")
    else:
        st.success(f"Bashorat: Saraton kasalligi aniqlanmadi. Ehtimollik: {probability[0][0]:.2f}")

# Qo'shimcha ma'lumot
st.sidebar.info("Dastur faqat ma'lumot berish maqsadida ishlatiladi.")
