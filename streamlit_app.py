import streamlit as st
import pandas as pd
import joblib

# Load Model, Encoder, dan Scaler
model = joblib.load('trained_modelll.pkl')
encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.pkl')

def convert_input_to_dataframe(user_data):
    """Mengonversi input user menjadi DataFrame."""
    columns = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 
               'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 
               'FAF', 'TUE', 'CALC', 'MTRANS']
    return pd.DataFrame([user_data], columns=columns)

def preprocess_data(df):
    """Melakukan encoding dan normalisasi pada data input."""
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = encoder.fit_transform(df[col])
    return scaler.transform(df)

def get_prediction(model, processed_data):
    """Memprediksi hasil berdasarkan input yang telah diproses."""
    return model.predict(processed_data)[0], model.predict_proba(processed_data)

def main():
    st.title('Machine Learning App')
    st.info('This app will predict your obesity level!')

    # Menampilkan Data
    with st.expander('**Dataset**'):
        st.write('Raw Data:')
        dataset = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
        st.write(dataset)

        st.write('**X (Features)**')
        X = dataset.drop('NObeyesdad', axis=1)
        st.write(X)

        st.write('**y (Target)**')
        y = dataset['NObeyesdad']
        st.write(y)

    # Visualisasi Data
    with st.expander('**Data Visualization**'):
        st.scatter_chart(data=dataset, x='Height', y='Weight', color='NObeyesdad')

    # Input User
    Age = st.slider('Age', 14, 61, 24)
    Height = st.slider('Height', 1.45, 1.98, 1.7)
    Weight = st.slider('Weight', 39, 173, 86)
    FCVC = st.slider('FCVC', 1, 3, 2)
    NCP = st.slider('NCP', 1, 4, 3)
    CH2O = st.slider('CH2O', 1, 3, 2)
    FAF = st.slider('FAF', 0, 3, 1)
    TUE = st.slider('TUE', 0, 2, 1)

    Gender = st.selectbox('Gender', ['Male', 'Female'])
    family_history = st.selectbox('Family history with overweight', ['yes', 'no'])
    FAVC = st.selectbox('Frequent consumption of high-caloric food (FAVC)', ['yes', 'no'])
    CAEC = st.selectbox('Consumption of food between meals (CAEC)', ['Sometimes', 'Frequently', 'Always', 'no'])
    SMOKE = st.selectbox('SMOKE', ['yes', 'no'])
    SCC = st.selectbox('SCC', ['yes', 'no'])
    CALC = st.selectbox('CALC', ['Sometimes', 'no', 'Frequently', 'Always'])
    MTRANS = st.selectbox('MTRANS', ['Public_Transportation', 'Automobile', 'Walking', 'Motorbike', 'Bike'])

    # Memproses Input User
    user_input = [Gender, Age, Height, Weight, family_history, FAVC, FCVC, NCP, 
                  CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS]
    
    df_input = convert_input_to_dataframe(user_input)
    st.write('User Input:')
    st.write(df_input)

    df_encoded = preprocess_data(df_input)
    prediction, prediction_proba = get_prediction(model, df_encoded)

    # Menampilkan Hasil Prediksi
    st.write('**Obesity Prediction**')
    df_proba = pd.DataFrame(prediction_proba, columns=['Insufficient Weight', 'Normal Weight', 
                                                        'Overweight Level I', 'Overweight Level II', 
                                                        'Obesity Type I', 'Obesity Type II', 'Obesity Type III'])
    st.write(df_proba)
    
    st.write('Predicted Obesity Level:', prediction)

if __name__ == '__main__':
    main()
