import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Graduate Admission Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# Load Model
# -------------------------------------------------
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# -------------------------------------------------
# Custom CSS
# -------------------------------------------------
st.markdown("""
<style>

.stApp{
background: linear-gradient(135deg,#0f172a,#1e3a8a,#2563eb);
background-size:400% 400%;
animation: gradient 15s ease infinite;
}

@keyframes gradient{
0%{background-position:0% 50%;}
50%{background-position:100% 50%;}
100%{background-position:0% 50%;}
}

.title{
text-align:center;
font-size:50px;
font-weight:bold;
color:white;
}

.subtitle{
text-align:center;
font-size:20px;
color:#dbeafe;
margin-bottom:20px;
}

.glass{
background: rgba(255,255,255,0.10);
padding:25px;
border-radius:20px;
backdrop-filter: blur(14px);
border:1px solid rgba(255,255,255,.25);
box-shadow:0 10px 30px rgba(0,0,0,.25);
}

.metric{
background:white;
padding:18px;
border-radius:15px;
text-align:center;
box-shadow:0 10px 20px rgba(0,0,0,.20);
}

.big{
font-size:42px;
font-weight:bold;
color:#2563EB;
}

.small{
font-size:18px;
color:gray;
}

div.stButton > button{
background:#2563EB;
color:white;
font-size:22px;
font-weight:bold;
height:60px;
width:100%;
border-radius:12px;
border:none;
transition:0.3s;
}

div.stButton > button:hover{
background:#1D4ED8;
transform:scale(1.02);
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:

    st.title("🎓 Graduate Admission")

    st.markdown("---")

    st.write("### About")

    st.info("""
This application predicts the probability of admission
using a Linear Regression model trained on the
Graduate Admissions dataset.

### Features Used

- GRE Score
- TOEFL Score
- University Rating
- SOP
- LOR
- CGPA
- Research
""")

# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown("<div class='title'>🎓 Graduate Admission Predictor</div>", unsafe_allow_html=True)

st.markdown(
"<div class='subtitle'>Predict your probability of admission using Machine Learning</div>",
unsafe_allow_html=True
)

st.write("")

# -------------------------------------------------
# Layout
# -------------------------------------------------
left, right = st.columns([1.2,1])

# -------------------------------------------------
# INPUTS
# -------------------------------------------------
with left:

    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    st.subheader("📚 Student Academic Profile")

    gre = st.slider("GRE Score",260,340,320)

    toefl = st.slider("TOEFL Score",80,120,110)

    university = st.select_slider(
        "University Rating",
        options=[1,2,3,4,5],
        value=3
    )

    sop = st.slider("Statement of Purpose (SOP)",1.0,5.0,4.0,0.5)

    lor = st.slider("Letter of Recommendation (LOR)",1.0,5.0,4.0,0.5)

    cgpa = st.slider("CGPA",6.0,10.0,8.8,0.01)

    research = st.radio(
        "Research Experience",
        ["No","Yes"],
        horizontal=True
    )

    research_value = 1 if research=="Yes" else 0

    predict = st.button("🚀 Predict Admission Chance")

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
with right:

    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    st.subheader("📈 Prediction")

    if predict:

        X = np.array([[
            gre,
            toefl,
            university,
            sop,
            lor,
            cgpa,
            research_value
        ]])

        prediction = model.predict(X)[0]

        prediction = max(0, min(prediction,1))

        percent = prediction*100

        if percent >= 90:
            label="🏆 Excellent Chance"
            color="green"
            st.balloons()

        elif percent >=80:
            label="🎉 Very High Chance"
            color="green"

        elif percent >=70:
            label="✅ Good Chance"
            color="orange"

        elif percent >=60:
            label="👍 Moderate Chance"
            color="orange"

        else:
            label="📚 Needs Improvement"
            color="red"

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=percent,
            number={'suffix':"%"},
            title={'text':"Admission Chance"},
            gauge={
                'axis':{'range':[0,100]},
                'bar':{'color':'royalblue'},
                'steps':[
                    {'range':[0,40],'color':'#ff4d4d'},
                    {'range':[40,70],'color':'#ffd54f'},
                    {'range':[70,100],'color':'#66bb6a'}
                ]
            }
        ))

        fig.update_layout(
            height=350,
            margin=dict(l=20,r=20,t=40,b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )

        st.plotly_chart(fig, use_container_width=True)

        st.progress(percent/100)

        st.markdown(
            f"""
            <div class='metric'>
                <div class='big'>{percent:.2f}%</div>
                <div class='small'>{label}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.info("Fill in the student's academic profile and click **Predict Admission Chance**.")

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# Profile Summary
# -------------------------------------------------
if predict:

    st.write("")

    st.markdown("## 📋 Student Profile Summary")

    c1,c2,c3,c4 = st.columns(4)

    c1.metric("GRE",gre)

    c2.metric("TOEFL",toefl)

    c3.metric("CGPA",cgpa)

    c4.metric("Research",research)

    c5,c6,c7 = st.columns(3)

    c5.metric("University Rating",university)

    c6.metric("SOP",sop)

    c7.metric("LOR",lor)

st.write("")
st.markdown("---")
st.caption("Made with ❤️ using Streamlit • Scikit-Learn • Plotly")
