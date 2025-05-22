import os
import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import google.generativeai as genai
import faiss
import numpy as np
import tiktoken

# Get API key securely from Streamlit secrets
api_key = st.secrets["GOOGLE_GEMINI_API_KEY"]
genai.configure(api_key=api_key)

# Extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    try:
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

# Extract data from Excel
def extract_data_from_excel(file):
    try:
        sheets = pd.read_excel(file, sheet_name=None)
        formatted_data = {sheet: df.to_json() for sheet, df in sheets.items()}
        return formatted_data
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return {}

# Chunk large text
def chunk_text(text, max_tokens=500):
    encoder = tiktoken.get_encoding("cl100k_base")
    tokens = encoder.encode(text)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return ["".join(encoder.decode(chunk)) for chunk in chunks]

# FAISS vector DB
def create_vector_db(chunks):
    embeddings = np.random.rand(len(chunks), 384).astype("float32")  # Dummy embeddings
    index = faiss.IndexFlatL2(384)
    index.add(embeddings)
    return index, chunks

# Retrieve top relevant chunks
def retrieve_chunks(query, index, chunks, top_k=3):
    query_embedding = np.random.rand(1, 384).astype("float32")  # Dummy
    _, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

# Get AI response from Gemini
def get_ai_response(context, question):
    model = genai.GenerativeModel("gemini-1.5-pro-latest")

    chat_history = "\n".join(
        [f"User: {msg['message']}" if msg['role'] == 'user' else f"AI: {msg['message']}" 
         for msg in st.session_state.get("chat_history", [])]
    )

    prompt = f"Chat History:\n{chat_history}\n\nContext: {context}\nUser Question: {question}\nAnswer:"
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip() if response else "No response received."
    except Exception as e:
        return f"Error generating AI response: {e}"

# UI Starts Here
st.title("📄 Document Assistant Chatbot")
st.markdown("**Upload your document and ask questions! 🤖**")

uploaded_file = st.file_uploader("**📂 Upload a document (PDF or Excel)**", type=["pdf", "xlsx"])

extracted_text = ""
excel_data = None  

if uploaded_file:
    file_type = uploaded_file.type

    with st.spinner("📖 Extracting content..."):
        if "pdf" in file_type:
            extracted_text = extract_text_from_pdf(uploaded_file)
            preview_text = extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text
        elif "excel" in file_type or "spreadsheet" in file_type:
            excel_data = extract_data_from_excel(uploaded_file)
            preview_text = str(list(excel_data.keys()))

    st.subheader("🔍 Document Preview")
    with st.expander("**📜 Click to Preview Extracted Text**"):
        if extracted_text:
            st.markdown(
                f'<div style="max-height: 150px; overflow-y: auto; border: 2px solid #ccc; padding: 10px; border-radius: 5px;">{preview_text}</div>',
                unsafe_allow_html=True
            )
        elif excel_data:
            for sheet, data in excel_data.items():
                st.write(f"**📄 Sheet: {sheet}**")
                df = pd.read_json(data)
                st.dataframe(df.head())

# Chunk & create index
if extracted_text:
    text_chunks = chunk_text(extracted_text)
    index, stored_chunks = create_vector_db(text_chunks)
elif excel_data:
    text_chunks = [f"{sheet}: {data}" for sheet, data in excel_data.items()]
    index, stored_chunks = create_vector_db(text_chunks)
else:
    text_chunks, index, stored_chunks = [], None, []

# Chat History
st.subheader("📜 Chat History")
for msg in st.session_state.get("chat_history", []):
    if msg["role"] == "user":
        st.markdown(f"""
            <div style="display: flex; justify-content: flex-end; margin: 5px 0;">
                <div style="background-color: #0052cc; color: white; padding: 12px; border-radius: 10px; max-width: 60%;">
                    <b>👤 You:</b> {msg["message"]}
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style="display: flex; justify-content: flex-start; margin: 5px 0;">
                <div style="background-color: #44475a; color: white; padding: 12px; border-radius: 10px; max-width: 60%;">
                    <b>🤖 AI:</b> {msg["message"]}
                </div>
            </div>
        """, unsafe_allow_html=True)

# Input
st.subheader("💬 Ask a Question")
user_input = st.text_input("Type your question here:")
col1, col2 = st.columns([0.8, 0.2])

with col1:
    if st.button("🚀 Send") and user_input:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        st.session_state.chat_history.append({"role": "user", "message": user_input})

        with st.spinner("🤖 Generating AI response..."):
            context = "\n".join(retrieve_chunks(user_input, index, stored_chunks)) if stored_chunks else "No document content available."
            answer = get_ai_response(context, user_input)
            st.session_state.chat_history.append({"role": "ai", "message": answer})
        st.rerun()

with col2:
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()
