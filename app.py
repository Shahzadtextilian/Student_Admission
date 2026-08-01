import streamlit as st
import pickle
import numpy as np

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Student Admission Predictor",
    page_icon="🎓",
    layout="wide"
)

# -------------------------
# Custom CSS
# -------------------------
st.markdown("""
<style>

.stApp{
    background: linear-gradient(135deg,#0F172A,#1E3A8A);
}

.main-title{
    font-size:48px;
    font-weight:700;
    color:white;
    text-align:center;
}

.sub-title{
    text-align:center;
    color:#dbeafe;
    font-size:20px;
}

.prediction-box{
    padding:25px;
    border-radius:20px;
    background-color:#ffffff15;
    backdrop-filter: blur(8px);
}

.metric-card{
    background:white;
    padding:15px;
    border-radius:15px;
    box-shadow:0px 8px 20px rgba(0,0,0,.15);
}

div.stButton > button{
    width:100%;
    height:60px;
    font-size:22px;
    border-radius:12px;
    background:#2563EB;
    color:white;
    font-weight:bold;
}

div.stButton > button:hover{
    background:#1D4ED8;
    color:white;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# Load Model
# -------------------------
model = pickle.load(open("model.pkl", "rb"))

# -------------------------
# Header
# -------------------------
st.markdown("<h1 class='main-title'>🎓 Student Admission Chance Predictor</h1>", unsafe_allow_html=True)

st.markdown("<p class='sub-title'>Predict the probability of admission using Machine Learning</p>", unsafe_allow_html=True)

st.write("")

# -------------------------
# Layout
# -------------------------
col1, col2 = st.columns([1,1])

with col1:

    st.markdown("### 📊 Student Academic Profile")

    gre = st.slider("GRE Score",260,340,310)

    toefl = st.slider("TOEFL Score",80,120,105)

    university = st.slider("University Rating",1,5,3)

    sop = st.slider("SOP Strength",1.0,5.0,3.0)

    lor = st.slider("LOR Strength",1.0,5.0,3.0)

    cgpa = st.slider("CGPA",6.0,10.0,8.5)

    research = st.selectbox("Research Experience",[0,1])

with col2:

    st.markdown("### 📈 Prediction")

    st.markdown('<div class="prediction-box">',unsafe_allow_html=True)

    st.write("Enter the student details and click the button below.")

    if st.button("Predict Admission Chance"):

        data = np.array([[gre,
                          toefl,
                          university,
                          sop,
                          lor,
                          cgpa,
                          research]])

        prediction = model.predict(data)[0]

        prediction = max(0, min(prediction,1))

        st.success(f"Admission Chance: **{prediction*100:.2f}%**")

        if prediction >= 0.80:
            st.balloons()
            st.success("Excellent admission chances! 🎉")

        elif prediction >= 0.60:
            st.info("Good chances of admission.")

        elif prediction >= 0.40:
            st.warning("Moderate chances. Consider improving your profile.")

        else:
            st.error("Low admission probability.")

    st.markdown("</div>",unsafe_allow_html=True)

st.write("")

st.markdown("---")

st.caption("Built with ❤️ using Streamlit & Scikit-Learn")