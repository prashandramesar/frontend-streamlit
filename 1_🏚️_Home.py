# Source: https://github.com/Sven-Bo

# Imports
from pathlib import Path
import streamlit as st
from PIL import Image


# --- PATH SETTINGS ---
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
css_file = current_dir / "styles" / "main.css"
resume_file = current_dir / "assets" / "CV.pdf"
profile_pic = current_dir / "assets" / "profile-pic.jpg"

# --- GENERAL SETTINGS ---
PAGE_TITLE = "Digital CV | Prashand Ramesar"
PAGE_ICON = ":wave:"
NAME = "Prashand Ramesar"
DESCRIPTION = """
Data Scientist, unlocking value for business using Machine Learning
"""

SOCIAL_MEDIA = {
    "LinkedIn": "https://www.linkedin.com/in/prashand-ramesar/",
    "GitHub": "https://github.com/prashandramesar"
}

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)

# --- LOAD CSS, PDF & PROFILE PIC ---
with open(css_file) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
with open(resume_file, "rb") as pdf_file:
    PDFbyte = pdf_file.read()
profile_pic = Image.open(profile_pic)

# --- HERO SECTION ---
col1, col2 = st.columns(2, gap="small")
with col1:
    st.image(profile_pic, width=230)

with col2:
    st.title(NAME)
    st.write(DESCRIPTION)
    st.download_button(
        label=" 📄 Download Resume",
        data=PDFbyte,
        file_name=resume_file.name,
        mime="application/octet-stream",
    )

# --- SOCIAL LINKS ---
st.write('\n')
cols = st.columns(len(SOCIAL_MEDIA))
for index, (platform, link) in enumerate(SOCIAL_MEDIA.items()):
    cols[index].write(f"[{platform}]({link})")


# --- EXPERIENCE & QUALIFICATIONS ---
st.write('\n')
st.header("Experience & Qualifications")
st.write("---")
st.write(
    """
- ✔️ Close on 4 years industry experience building ML models
- ✔️ Strong hands on experience and knowledge in Python and related packages
- ✔️ Good understanding of bringing models into production using AWS SageMaker 
- ✔️ MSc Computer Science, BSc Computer Science & Economics
- ✔️ Self-starter, but being on the same page is key
"""
)

# --- SKILLS ---
st.write('\n')
st.header("Hard Skills")
st.write("---")
st.write(
    """
- 👩‍💻 Coding: Python (Scikit-learn, Pandas, SciPy, Numpy, Scikit-learn, PyTorch), SQL, R
- 📊 Data Visulization: Plotly, MatplotLib, Folium, GeoPandas
- 📚 Modeling: Regression, Classification, Forecasting, Clustering, NLP
- ☁️ Cloud stack: AWS
"""
)

# --- WORK HISTORY ---
st.write('\n')
st.header("Relevant Work History")
st.write("---")

# --- JOB 1
st.subheader("🚧 **Data Scientist | ANWB**")
st.write("11/2019 - Present")
st.write(
    """
- ► Building ML models (regression, classification, forecasting) from scratch to production
through AWS SageMaker
- ► Identifying (under)performing marketing channels through Marketing Mix Modeling
- ► Categorizing Dutch reviews through text mining efforts
- ► Initiative: authoring and publishing data blogs: https://medium.com/anwb-data-driven
- ► Mentoring and helping Jr. Data Scientists & Trainees
- ► Webinar speaker at Big Data Expo 2020 about ANWB’s First-Time-Right model
"""
)

# --- JOB 2
st.write('\n')
st.subheader("🚧 **Graduate Cyber Security intern | TNO**")
st.write("09/2018 - 04/2019")
st.write(
    """
- ► MSc thesis project: designing computational algorithms that estimate the risk of visiting
domains on the Internet, graded with an 8. I mostly used Python, SQL, Pandas,
NumPy, SciPy, Linux. Retrieve from: https://theses.liacs.nl/1611.
"""
)

# --- JOB 3
st.write('\n')
st.subheader("🚧 **Machine Learning Intern | Merkle Nederland**")
st.write("04/2018 - 07/2018")
st.write(
    """
- ► Analyzing customer data and building ML models that predict which product each
customer is most likely to purchase next. Graded with an 8.5. I used Python, R, Pandas,
NumPy, SciPy, Scikit-learn, Tensorflow and Keras during this internship.
"""
)