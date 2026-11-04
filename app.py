import streamlit as st
import pandas as pd
import pickle

# Set page configuration
st.set_page_config(page_title="Student Prediction App", layout="centered", page_icon="🎓")

# --- Custom UI Styling (Background & Forms) ---
def add_bg_from_css():
    st.markdown(
        """
        <style>
        /* Main background gradient */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            color: white;
        }
        
        /* Make header transparent */
        [data-testid="stHeader"] {
            background-color: rgba(0,0,0,0);
        }

        /* Glassmorphism effect for the form */
        [data-testid="stForm"] {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.18);
            padding: 20px;
        }

        /* Style tweaks for text to ensure readability */
        h1, h2, h3, p {
            color: #f1f2f6 !important;
        }
        
        /* Style the submit button */
        button[kind="primaryFormSubmit"] {
            background-color: #00b894 !important;
            color: white !important;
            border-radius: 8px !important;
            border: none !important;
            font-weight: bold !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_css()

# --- Load Models and Artifacts ---
@st.cache_resource
def load_artifacts():
    """Loads the model, scaler, and column names."""
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('columns.pkl', 'rb') as f:
            columns = pickle.load(f)
        return model, scaler, columns
    except FileNotFoundError as e:
        st.error(f"Error loading files: {e}. Please ensure model.pkl, scaler.pkl, and columns.pkl exist.")
        st.stop()

model, scaler, columns = load_artifacts()

# --- App Interface ---
st.title("🎓 Student Success Predictor")
st.markdown("Enter the student's academic and demographic details below to generate a prediction.")

# --- Dynamic Input Form ---
input_data = {}

with st.form("prediction_form"):
    st.subheader("Student Features")
    
    col1, col2 = st.columns(2)
    
    for i, col in enumerate(columns):
        if i % 2 == 0:
            with col1:
                input_data[col] = st.number_input(label=col, value=0.0, step=1.0)
        else:
            with col2:
                input_data[col] = st.number_input(label=col, value=0.0, step=1.0)
    
    st.markdown("<br>", unsafe_allow_html=True)
    submit_button = st.form_submit_button(label="Generate Prediction ✨")

# --- Prediction Logic ---
if submit_button:
    input_df = pd.DataFrame([input_data])
    
    try:
        scaled_features = scaler.transform(input_df)
        raw_prediction = model.predict(scaled_features)[0]
        
        st.divider()
        
        # Map True/False to Pass/Fail
        prediction_str = str(raw_prediction).strip().lower()
        if prediction_str in ['true', '1', '1.0']:
            st.success("### Predicted Outcome: **Pass ✅**")
        elif prediction_str in ['false', '0', '0.0']:
            st.error("### Predicted Outcome: **Fail ❌**")
        else:
            # Fallback just in case the classes are named differently
            st.info(f"### Predicted Outcome: **{raw_prediction}**")
        
        # Display probabilities if available
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(scaled_features)[0]
            st.markdown("#### Confidence Breakdown:")
            
            for class_label, prob in zip(model.classes_, probabilities):
                # Map the class labels for the progress bars as well
                label_str = str(class_label).strip().lower()
                if label_str in ['true', '1', '1.0']:
                    display_label = "Pass"
                elif label_str in ['false', '0', '0.0']:
                    display_label = "Fail"
                else:
                    display_label = class_label
                    
                st.write(f"**{display_label}**")
                st.progress(float(prob), text=f"{prob:.1%}")
                
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")