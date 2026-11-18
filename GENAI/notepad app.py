import streamlit as st

st.title("GENAI PROJECT")
st.write("Welcome to my Generative AI application!")

question = st.text_input("Enter your question")

if st.button("Submit"):
    st.write("Your question:", question)