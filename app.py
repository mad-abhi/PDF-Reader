import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama

load_dotenv()

llm = Ollama(base_url="http://localhost:11434")

embeddings_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")


def process_pdf(uploaded_file):
    """Processes the uploaded PDF and returns a FAISS vector store."""
    reader = PdfReader(uploaded_file)
    raw_text = ""
    for page in reader.pages:
        raw_text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    texts = text_splitter.split_text(raw_text)

    db = FAISS.from_texts(texts, embeddings_model)
    return db


# Streamlit app
st.title("PDF Reader and Question Answering with Ollama")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    db = process_pdf(uploaded_file)

    query = st.text_input("Enter your question about the PDF:")

    if query:
        prompt_template = """Use the following context to answer the question at the end. If the context doesn't contain the answer, say "I don't have enough information to answer."

        {context}

        Question: {query}
        """
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "query"]
        )

        docs = db.similarity_search(query)
        context = "\n".join([doc.page_content for doc in docs])

        final_prompt = PROMPT.format(context=context, query=query)
        response = llm.invoke(final_prompt)

        st.write("Answer:")
        st.write(response)
