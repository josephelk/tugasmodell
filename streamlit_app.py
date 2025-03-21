import streamlit as st
import pandas as pd
import numpy as np

st.title("Machine Learning App")
st.info("This app will predict your obesity level!")

dff = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv") 
df = pd.DataFrame(dff)

# Expander to show raw data
with st.expander("Data"):
    st.write("This is a raw data")
    st.dataframe(df)

def predict_with_model(model, scaler, label_encoders, user_input):
    # Konversi input kategori ke numerik
    user_input[0] = label_encoders["Gender"].transform([user_input[0]])[0]  
    user_input[4] = label_encoders["family_history_with_overweight"].transform([user_input[4]])[0]
    user_input[5] = label_encoders["FAVC"].transform([user_input[5]])[0]
    user_input[8] = label_encoders["CAEC"].transform([user_input[8]])[0]
    user_input[9] = label_encoders["SMOKE"].transform([user_input[9]])[0]
    user_input[11] = label_encoders["SCC"].transform([user_input[11]])[0]
    user_input[14] = label_encoders["CALC"].transform([user_input[14]])[0]
    user_input[15] = label_encoders["MTRANS"].transform([user_input[15]])[0]

    # Normalisasi input numerik
    numerical_indices = [1, 2, 3, 6, 7, 10, 12, 13]
    user_input[numerical_indices] = scaler.transform([user_input[numerical_indices]])[0]

    # Konversi ke array
    input_array = np.array(user_input).reshape(1, -1)

    # Prediksi probabilitas
    probabilities = model.predict_proba(input_array)
    prediction = model.predict(input_array)

    predicted_class = label_encoders["NObeyesdad"].inverse_transform(prediction)[0]

    return predicted_class, probabilities

# **User Input**
    st.subheader("🔹 User Input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 0, 100, 25)
        height = st.number_input("Height (m)", 1.0, 2.5, 1.7)
        weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0)
        family_history = st.selectbox("Family History of Overweight", ["yes", "no"])
        favc = st.selectbox("Frequent Consumption of High-Calorie Food", ["yes", "no"])
        fcvc = st.slider("Vegetable Consumption Frequency", 1.0, 3.0, 2.0)
        ncp = st.slider("Number of Meals per Day", 1.0, 4.0, 3.0)

    with col2:
        caec = st.selectbox("Consumption of Food Between Meals", ["no", "Sometimes", "Frequently", "Always"])
        smoke = st.selectbox("Do you smoke?", ["yes", "no"])
        ch2o = st.slider("Water Consumption (liters per day)", 1.0, 3.0, 2.0)
        scc = st.selectbox("Calories Monitoring?", ["yes", "no"])
        faf = st.slider("Physical Activity Frequency", 0.0, 3.0, 1.0)
        tue = st.slider("Time Using Technology (hours)", 0.0, 2.0, 1.0)
        calc = st.selectbox("Alcohol Consumption", ["no", "Sometimes", "Frequently", "Always"])
        mtrans = st.selectbox("Mode of Transportation", ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"])

    # **Tampilkan input pengguna dalam tabel**
    user_data = pd.DataFrame([[gender, age, height, weight, family_history, favc, fcvc, ncp, caec, smoke, ch2o, scc, faf, tue, calc, mtrans]],
                             columns=["Gender", "Age", "Height", "Weight", "family_history_with_overweight", "FAVC", "FCVC", "NCP", "CAEC", "SMOKE", "CH2O", "SCC", "FAF", "TUE", "CALC", "MTRANS"])
    
    st.subheader("📝 Data input by user")
    st.dataframe(user_data)
