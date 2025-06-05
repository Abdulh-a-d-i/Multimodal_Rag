import streamlit as st
import requests
import os
import time
from typing import List, Dict
import tempfile

# Configuration
BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="Multimodal RAG Application", layout="wide")

def display_source(source: Dict):
    if source['type'] == "pdf":
        st.markdown(f"""
        **PDF Source**: {source['source']}  
        **Page**: {source['page']}  
        **Preview**: {source['content']}
        """)
    else:
        st.markdown(f"""
        **Video Source**: {source['source']}  
        **Timestamp**: {source['timestamp']}  
        **Preview**: {source['content']}
        """)

def upload_pdf(file):
    with st.spinner("Processing PDF..."):
        files = {"file": file}
        response = requests.post(f"{BACKEND_URL}/upload/pdf/", files=files)
        return response.json()

def upload_video(file):
    with st.spinner("Processing Video..."):
        files = {"file": file}
        response = requests.post(f"{BACKEND_URL}/upload/video/", files=files)
        return response.json()

def query_rag(question: str):
    with st.spinner("Searching for answers..."):
        response = requests.post(
            f"{BACKEND_URL}/query/",
            json={"question": question}
        )
        return response.json()

def main():
    st.title("Multimodal RAG Application")
    st.markdown("Upload PDFs and videos, then ask questions about their content.")
    
    # File upload section
    with st.expander("Upload Files", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Upload PDF")
            pdf_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_uploader")
            if pdf_file:
                result = upload_pdf(pdf_file)
                st.success(f"PDF processed successfully! File ID: {result['file_id']}")
        
        with col2:
            st.subheader("Upload Video")
            video_file = st.file_uploader("Choose an MP4 file", type="mp4", key="video_uploader")
            if video_file:
                result = upload_video(video_file)
                st.success(f"Video processed successfully! File ID: {result['file_id']}")
    
    # Query section
    st.divider()
    st.subheader("Ask a Question")
    
    question = st.text_input("Enter your question about the uploaded content:")
    
    if question:
        response = query_rag(question)
        
        st.subheader("Answer")
        st.write(response['answer'])
        
        st.subheader("Sources")
        for source in response['sources']:
            display_source(source)
            st.divider()

if __name__ == "__main__":
    main()