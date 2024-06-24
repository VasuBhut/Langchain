import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
# from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
# from keys import openai_api_key
from keys import google_api_key

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

# os.environ["OPENAI_API_KEY"]=google_api_key
os.environ["GOOGLE_API_KEY"]=google_api_key


st.title("News Research Tool")
st.sidebar.title("News Artical URLs")

urls=[]

for i in range(3):
    url=st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_urls=st.sidebar.button("Process URL")
main_placeholder=st.empty()

# llm = OpenAI(temperature=0.9, max_tokens=500) 
llm=genai.configure(api_key=google_api_key)


if process_urls:
    loader=UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started✅✅✅")
    data=loader.load()
    
    recursive_splitter=RecursiveCharacterTextSplitter(
        separators=["\n\n","\n",".",","],
        chunk_size=1000,
    )

    main_placeholder.text("Text Splitter...Started✅✅✅")

    chunks=recursive_splitter.split_documents(data)

    # embeddings=OpenAIEmbeddings()
    embeddings=HuggingFaceBgeEmbeddings()
    vectorstore_openai=FAISS.from_documents(chunks,embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index and metadata
    faiss_dir_path = "vector_index.faiss"
    vectorstore_openai.save_local(faiss_dir_path)

query=main_placeholder.text_input("Question: ")

if query:
        faiss_dir_path = "vector_index.faiss"
        embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore_openai = FAISS.load_local(faiss_dir_path, embeddings=embeddings,allow_dangerous_deserialization=True)
        chain = RetrievalQAWithSourcesChain(retriever=vectorstore_openai.as_retriever())
        result=chain({"question":query})
        st.header("Answer")
        st.write(result["answer"])

        sources=result.get("Sources:",result["sources"])
        
        if sources:
            st.subheader("Sources: ")
            sources_list=sources.split("\n")
            for source in sources_list:
                 st.write(source)


# https://www.moneycontrol.com/news/business/tata-motors-mahindra-gain-certificates-for-production-linked-payouts-11281691.html
# https://www.moneycontrol.com/news/business/tata-motors-launches-punch-icng-price-starts-at-rs-7-1-lakh-11098751.html
# https://www.moneycontrol.com/news/business/stocks/buy-tata-motors-target-of-rs-743-kr-choksey-11080811.html