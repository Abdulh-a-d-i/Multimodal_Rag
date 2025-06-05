# Multimodal RAG Application

![image](https://github.com/user-attachments/assets/230264b1-ffa2-4762-84e3-b46987f56975)


A powerful Retrieval-Augmented Generation (RAG) application that processes both PDF documents and video files, then answers questions using the OpenRouter API with various AI models.

## Features

- **Multimodal Processing**:
  - PDF text extraction with page level metadata
  - Video transcription with timestamp segmentation
- **AI-Powered Q&A**:
  - Query documents using state-of-the-art LLMs via OpenRouter
  - Supports multiple models (Claude, GPT, deepseek.)
- **Interactive Interface**:
  - Streamlit-based web UI
  - Document preview with text selection
  - Video player with transcript navigation

## Technologies Used

- **Frontend**: Streamlit
- **Backend**: FastAPI
- **Vector Database**: ChromaDB
- **LLM Gateway**: OpenRouter API
- **File Processing**:
  - PyPDF2 (PDFs)
  - MoviePy (video)
  - SpeechRecognition (audio)

## Getting Started

### Prerequisites

- Python 3.9+
- Git
- OpenRouter API key (get from [OpenRouter](https://openrouter.ai/))

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Abdulh-a-d-i/Multimodal_Rag.git
   cd Multimodal_Rag
