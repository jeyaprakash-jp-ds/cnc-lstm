import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

# Function to load pickle files
def load_pickle_model(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)

# Load encoder, scaler, and deep learning model
encoder = load_pickle_model("Encoder_MP.pkl")
scaler = load_pickle_model("scale.pkl")
model = tf.keras.models.load_model("final_lstm.h5")

# Streamlit UI
st.set_page_config(page_title="CNC Prediction", layout="wide")

# Sidebar
st.sidebar.image("logo.png", width=200)  # Add a logo (if available)
st.sidebar.title("Navigation")
st.sidebar.write("Upload data and get predictions")

st.markdown("## **FINAL CNC PREDICTION PROJECT**")
st.write("Upload a CSV file to predict machining conditions.")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    with st.spinner("Processing file..."):
        try:
            df = pd.read_csv(uploaded_file, dtype=str)
            
            # Convert numeric columns
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    pass

            # Encode categorical column
            if 'Machining_Process' in df.columns:
                df['Machining_Process'] = encoder.transform(df[["Machining_Process"]])
            else:
                st.error("Column 'Machining_Process' not found in the uploaded file.")

        except Exception as e:
            st.error(f"Error: {e}")

    st.write("### Uploaded Data Preview:")
    st.dataframe(df)

    if st.button('Get Predictions'):
        with st.spinner("Generating predictions..."):
            df_scaled = scaler.transform(df)
            df_scaled = df_scaled.reshape(df_scaled.shape[0], 1, df_scaled.shape[1])  # Reshape for LSTM
            predictions = model.predict(df_scaled)

            # Assign prediction results
            df["Tool Wear"] = ["Worn" if p[0] > 0.5 else "Unworn" for p in predictions]
            df["Visual Inspection"] = ["Properly Clamped" if p[1] > 0.5 else "Not Properly Clamped" for p in predictions]
            df["Machining Completion"] = ["Completed" if p[2] > 0.5 else "Not Completed" for p in predictions]

        st.success("Predictions Generated Successfully!")
        st.write("### **Predicted Results:**")
        st.dataframe(df)

        # Download button
        csv = df.to_csv(index=False)
        st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
