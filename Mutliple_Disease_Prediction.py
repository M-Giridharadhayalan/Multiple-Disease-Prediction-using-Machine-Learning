import pickle
import streamlit as st
import streamlit_options_menu as option_menu 
import numpy as np

#loading the models
with open('.venv\\Include\\kidney.pkl','rb') as n:
    kidney_model = pickle.load(n)

with open('.venv\liver_model_new.pkl', 'rb') as f:
    liver_model= pickle.load(f)

with open('.venv\parkinsons_knn_model.pkl', 'rb') as f:
    parkinsons_model= pickle.load(f)

#side bars 
with st.sidebar:
    selected = st.sidebar.selectbox('🧠🩺 Choose Prediction Type',
                           ['🩻 Kidney Disease Prediction', '🫁 Liver Disease Prediction', '🧠 Parkinsons Prediction'])
    
#Prediction Page 
if selected.startswith ('🩻 Kidney Disease Prediction'):
    #Page Title 
    st.title('Kidney Disease Prediction using ML')

# Split into 5 columns for neat layout
col1, col2, col3, col4, col5 = st.columns(5)

# === Numerical Inputs ===
with col1:
    Age = st.number_input('Enter Age', min_value=0, max_value=120, step=1)
with col2:
    bloodpressure = st.number_input('Blood Pressure (mmHg)', min_value=0.0)
with col3:
    specificgravity = st.number_input('Specific Gravity', min_value=1.0, step=0.01)
with col4:
    albumin = st.number_input('Albumin Level', min_value=0.0, step=0.1)
with col5:
    sugar = st.number_input('Sugar Level', min_value=0.0, step=0.1)

with col1:
    bloodglucoserandom = st.number_input('Blood Glucose (Random)', min_value=0.0)
with col2:
    bloodurea = st.number_input('Blood Urea', min_value=0.0)
with col3:
    serumcreatinine = st.number_input('Serum Creatinine', min_value=0.0)
with col4:
    sodium = st.number_input('Sodium', min_value=0.0)
with col5:
    pottasium = st.number_input('Potassium', min_value=0.0)

with col1:
    haemoglobin = st.number_input('Haemoglobin', min_value=0.0)
with col2:
    packedcellvolume = st.number_input('Packed Cell Volume', min_value=0.0)
with col3:
    whitebloodcellcount = st.number_input('White Blood Cell Count', min_value=0.0)
with col4:
    redbloodcellcount = st.number_input('Red Blood Cell Count', min_value=0.0)

# === Categorical Inputs ===
with col5:
    redbloodcells = st.selectbox('Red Blood Cells', ['normal', 'abnormal'])
with col1:
    puscell = st.selectbox('Pus Cell', ['normal', 'abnormal'])
with col2:
    puscellclumps = st.selectbox('Pus Cell Clumps', ['present', 'notpresent'])
with col3:
    bacteria = st.selectbox('Bacteria', ['present', 'notpresent'])
with col4:
    hpertension = st.selectbox('Hypertension', ['yes', 'no'])
with col5:
    diabetesmelitus = st.selectbox('Diabetes Mellitus', ['yes', 'no'])

with col1:
    coronaryarterydisease = st.selectbox('Coronary Artery Disease', ['yes', 'no'])
with col2:
    apetite = st.selectbox('Appetite', ['good', 'poor'])
with col3:
    pedaedima = st.selectbox('Pedal Edema', ['yes', 'no'])
with col4:
    anemia = st.selectbox('Anemia', ['yes', 'no'])

# === Encode categorical inputs ===
redbloodcells_val = 1 if redbloodcells == 'normal' else 0
puscell_val = 1 if puscell == 'normal' else 0
puscellclumps_val = 1 if puscellclumps == 'present' else 0
bacteria_val = 1 if bacteria == 'present' else 0
hpertension_val = 1 if hpertension == 'yes' else 0
diabetesmelitus_val = 1 if diabetesmelitus == 'yes' else 0
coronaryarterydisease_val = 1 if coronaryarterydisease == 'yes' else 0
apetite_val = 1 if apetite == 'good' else 0
pedaedima_val = 1 if pedaedima == 'yes' else 0
anemia_val = 1 if anemia == 'yes' else 0

# === Prediction logic ===
kidney_disease = ''

if st.button('Kidney Disease_Results'):
    input_data = [[Age, bloodpressure, specificgravity, albumin, sugar,
                   redbloodcells_val, puscell_val, puscellclumps_val, bacteria_val,
                   bloodglucoserandom, bloodurea, serumcreatinine, sodium, pottasium,
                   haemoglobin, packedcellvolume, whitebloodcellcount, redbloodcellcount,
                   hpertension_val, diabetesmelitus_val, coronaryarterydisease_val,
                   apetite_val, pedaedima_val, anemia_val]]

    kidney_disease_prediction = kidney_model.predict(input_data)

    if kidney_disease_prediction[0] == 1:
        kidney_disease = "🛑 The person is **likely** to have kidney disease."
    else:
        kidney_disease = "✅ The person is **not likely** to have kidney disease."

st.success(kidney_disease)


# --- Liver Prediction Page ---
if selected.startswith("🫁 Liver Disease Prediction"):
    st.title("Liver Disease Prediction using ML")

    # --- Input layout ---
    with st.container():
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            Age = st.number_input("Age", min_value=0, max_value=100, key="age")
        with col2:
            Gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
        with col3:
            Total_Bilirubin = st.number_input("Total Bilirubin", key="tb")
        with col4:
            Direct_Bilirubin = st.number_input("Direct Bilirubin", key="db")
        with col5:
            Alkaline_Phosphotase = st.number_input("Alkaline Phosphotase", key="ap")

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            Alamine_Aminotransferase = st.number_input("Alamine Aminotransferase", key="alt")
        with col2:
            Aspartate_Aminotransferase = st.number_input("Aspartate Aminotransferase", key="ast")
        with col3:
            Total_Protiens = st.number_input("Total Proteins", key="tp")
        with col4:
            Albumin = st.number_input("Albumin", key="alb")
        with col5:
            Albumin_and_Globulin_Ratio = st.number_input("Albumin/Globulin Ratio", key="agr")

    # --- Encode Gender ---
    gender_val = 1 if Gender == "Male" else 0

    # --- Prediction logic with safety check ---
    if st.button("🧪 Predict Liver Health"):
        liver_input_data = [[
            Age, gender_val, Total_Bilirubin, Direct_Bilirubin,
            Alkaline_Phosphotase, Alamine_Aminotransferase,
            Aspartate_Aminotransferase, Total_Protiens, Albumin,
            Albumin_and_Globulin_Ratio
        ]]

        if liver_model is None or not hasattr(liver_model, "predict"):
            st.error("🧠 Model isn't loaded correctly or was overwritten. Please recheck your logic or the model file path.")
        else:
            prediction_result = liver_model.predict(liver_input_data)

            # --- Expressive result feedback ---
            if prediction_result[0] == 1:
                st.markdown("🛑 **Alert:** This person is likely to have liver disease. Please consult a healthcare provider.")
            else:
                st.markdown("✅ **Good news:** This person is unlikely to have liver disease. Keep up the healthy lifestyle!")


if selected.startswith('🧠 Parkinsons Prediction'):
    st.title('Parkinsons Prediction using ML')

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1: MDVPFo = st.text_input('Enter the MDVP:Fo(Hz)')
    with col2: MDVPFhi = st.text_input('Enter the MDVP:Fhi(Hz)')
    with col3: MDVPFlo = st.text_input('Enter the MDVP:Flo(Hz)')
    with col4: MDVPJitter = st.text_input('Enter the MDVP Jitter(%)')
    with col5: MDVPJitterAbs = st.text_input('Enter the MDVP Jitter Abs')

    with col1: MDVPRAP = st.text_input('Enter the MDVP RAP')
    with col2: MDVPPPQ = st.text_input('Enter the MDVP PPQ')
    with col3: JitterDDP = st.text_input('Enter the Jitter DDP')
    with col4: MDVPShimmer = st.text_input('Enter the MDVP Shimmer')
    with col5: ShimmerAPQ3 = st.text_input('Enter the Shimmer APQ3')

    with col1: ShimmerAPQ5 = st.text_input('Enter the Shimmer APQ5')
    with col2: MDVPAPQ = st.text_input('Enter the MDVP APQ')
    with col3: ShimmerDDA = st.text_input('Enter the Shimmer DDA')
    with col4: NHR = st.text_input('Enter the NHR')
    with col5: HNR = st.text_input('Enter the HNR')

    with col1: RPDE = st.text_input('Enter the RPDE')
    with col2: DFA = st.text_input('Enter the DFA')
    with col3: spread1 = st.text_input('Enter the Spread1')
    with col4: spread = st.text_input('Enter the Spread2')
    with col5: D2 = st.text_input('Enter the D2')

    with col1: PPE = st.text_input('Enter the PPE')

    Parkinsons_disease = ""

    if st.button('Parkinsons Disease_Results'):
        try:
            # 🎯 Convert inputs to float
            input_data = [
                float(MDVPFo), float(MDVPFhi), float(MDVPFlo), float(MDVPJitter),
                float(MDVPJitterAbs), float(MDVPRAP), float(MDVPPPQ), float(JitterDDP),
                float(MDVPShimmer), float(ShimmerAPQ3), float(ShimmerAPQ5), float(MDVPAPQ),
                float(ShimmerDDA), float(NHR), float(HNR), float(RPDE),
                float(DFA), float(spread1), float(spread), float(D2), float(PPE)
            ]

            # 🧠 Load saved model and scaler
            with open("parkinsons_knn_model.pkl", "rb") as f:
                parkinsons_model = pickle.load(f)
            with open("parkinsons_scaler.pkl", "rb") as f:
                scaler = pickle.load(f)

            # 📐 Scale input
            input_scaled = scaler.transform([input_data])

            # 🔍 Prediction
            prediction = parkinsons_model.predict(input_scaled)

            if prediction[0] == 1:
                Parkinsons_disease = '🧠 The person **has** Parkinson’s disease'
            else:
                Parkinsons_disease = '💪 The person **does not** have Parkinson’s disease'

        except ValueError:
            Parkinsons_disease = '⚠️ Please enter valid numbers for all fields'

        with st.expander("🧠 Parkinsons Disease Prediction Result"):
            st.success(Parkinsons_disease)
