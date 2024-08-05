import streamlit as st
import requests

# Base URL for the FastAPI app
FASTAPI_URL = "http://localhost:8000"

st.title("Intelligent Bus Inquiry Assistance Chat Bot")

# AI Query Section
st.header("AI Query")
ai_query = st.text_input("Enter your query:")
if st.button("Submit AI Query"):
    response = requests.post(f"{FASTAPI_URL}/ai/", json={"query": ai_query})
    if response.status_code == 200:
        st.write(response.json().get("answer"))
    else:
        st.error("Failed to get a response from the AI endpoint.")

# PDF Upload Section
st.header("Upload PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
if uploaded_file and st.button("Upload PDF"):
    files = {'file': uploaded_file.getvalue()}
    response = requests.post(f"{FASTAPI_URL}/pdf/", files=files)
    if response.status_code == 200:
        st.success("PDF uploaded successfully.")
    else:
        st.error("Failed to upload PDF.")

# PDF Query Section
st.header("Query PDF")
pdf_query = st.text_input("Enter your query for the PDF:")
if st.button("Submit PDF Query"):
    response = requests.post(f"{FASTAPI_URL}/ask_pdf/", json={"query": pdf_query})
    if response.status_code == 200:
        response_data = response.json()
        st.write(response_data.get("answer"))
        st.write("Sources:")
        for source in response_data.get("sources", []):
            st.write(f"- {source['source']}: {source['page_content']}")
    else:
        st.error("Failed to get a response from the PDF query endpoint.")

# Agent Query Section
st.header("Agent Query")
agent_query = st.text_input("Enter your agent query:")
if st.button("Submit Agent Query"):
    response = requests.post(f"{FASTAPI_URL}/agent_query/", json={"query": agent_query})
    if response.status_code == 200:
        st.write(response.json().get("answer"))
    else:
        st.error("Failed to get a response from the agent query endpoint.")
