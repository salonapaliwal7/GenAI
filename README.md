# GenAI
# 📄 Chat with Your PDF - Streamlit App

This is a simple Streamlit-based chatbot that allows you to upload a PDF file and ask questions about its content. It uses **LangChain**, **OpenAI embeddings**, and **FAISS** for document retrieval and Q&A.

---

## 🔍 Features

- Upload a PDF file from the sidebar
- Extract text from the PDF using `PyPDF2`
- Split the text into meaningful chunks using LangChain's `RecursiveCharacterTextSplitter`
- Generate embeddings using OpenAI's `text-embedding-ada-002`
- Store and search embeddings using **FAISS**
- Ask natural language questions about your PDF
- Get responses using OpenAI's `gpt-3.5-turbo` model

---

## 📦 Dependencies

Make sure you install the following packages (preferably in a virtual environment):

```bash
pip install streamlit PyPDF2 langchain openai faiss-cpu

