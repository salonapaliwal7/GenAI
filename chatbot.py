import sys
print(sys.executable)
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community .chat_models import ChatOpenAI

OPENAI_API_KEY = "<your_openai_api_key>"
# Upload pdf files
st.header("My first chatbot")

with st.sidebar:
    st.title("Your documents")
    file = st.file_uploader("Upload a file and start asking questions", type="pdf")

# Extract text
if file is not None:
    pdfReader = PdfReader(file)
    text=""
    for page in pdfReader.pages:
        text +=page.extract_text()
        # st.write(text)

# break into chunks
    textSplitter = RecursiveCharacterTextSplitter(
        separators = "\n",
        chunk_size = 1000,
        chunk_overlap = 150, # this is to retain the meaning of previous chunk so noen of the chunks start in abrupt manner
        length_function = len
    )
    chunks = textSplitter.split_text(text)
    # st.write(chunks)  
       
    # Generating embedddings
    embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)

    # Creating vector store - FAISS
    vector_store = FAISS.from_texts(chunks, embeddings)

    # get user question 
    user_question = st.text_input("Type your question here")


    # do similarity search
    if user_question:
        match = vector_store.similarity_search(user_question)
        # st.write(match)


    # define llm
    llm = ChatOpenAI(
        openai_api_key = OPENAI_API_KEY,
        temperature = 0, # meaning we are asking the llm to not generate random values, meaning the answer should not be too lengthy
        max_tokens = 1000, # defines the limit of response
        model_name = "gpt-3.5-turbo"
    )

    # output results
    # chain -> take the question, get relevant documents, pass it to llm, genrate output
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain.run(input_documents = match, question = user_question)
    str.write(response)
