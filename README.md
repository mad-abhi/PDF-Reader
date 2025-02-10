# PDF Reader and Question Answering with Ollama

This project is a Streamlit-based application that allows users to upload a PDF file and ask questions about its content. The system processes the PDF, extracts text, generates embeddings, and performs similarity-based retrieval to answer user queries using Ollama.

## Features
- Upload a PDF file and extract text from it.
- Chunk and store text embeddings using FAISS.
- Perform semantic search on the extracted text.
- Answer questions based on the PDF content using an LLM (Ollama).

## Usage
1. Upload a PDF file.
2. Enter a question related to the document.
3. Get an AI-generated response based on the document's content.

## Requirements
Install dependencies using:
```bash
pip install -r requirements.txt
```

## How to Run
1. start ollama
   ```bash
   ollama run llama2
   ```
2. Run the stramlit app
   ```bash
   streamlit run app.py
   ```


![alt text](https://github.com/mad-abhi/PDF-Reader/blob/main/working.png)
