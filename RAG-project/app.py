import os
import streamlit as st
from langchain_groq import ChatGroq 
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
import time
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
if not groq_api_key:
    st.error("GROQ_API_KEY não está definido. Verifique suas variáveis de ambiente.")
    st.stop()

st.title("RAG com páginas web")

document_url = st.text_input("Forneça a URL da página aqui")

if document_url:
    if "vector" not in st.session_state:
        try:
            st.session_state.embeddings = OllamaEmbeddings()
            st.session_state.loader = WebBaseLoader(document_url)
            st.session_state.docs = st.session_state.loader.load()
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
            st.session_state.vector = FAISS.from_documents(st.session_state.documents, st.session_state.embeddings)
        except Exception as e:
            st.error(f"Erro ao inicializar embeddings ou carregar documentos: {e}")
            st.stop()

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name='mixtral-8x7b-32768'
    )

    prompt = ChatPromptTemplate.from_template(
        """
        Escreva somente em português a resposta da questão a seguir com base apenas no contexto fornecido. Pense passo a passo antes de fornecer uma resposta detalhada.
        <context>
        {context}
        <context>

        Question: {input}
        """
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    prompt = st.text_input("Digite a sua pergunta aqui")

    if prompt:
        try:
            start = time.process_time()
            response = retrieval_chain.invoke({"input": prompt})
            st.write(f"Response time: {time.process_time() - start}")
            st.write(response["answer"])

            with st.expander("Busca documentos similares"):
                for i, doc in enumerate(response["context"]):
                    st.write(doc.page_content)
                    st.write("------------------------------------------")
        except Exception as e:
            st.error(f"Erro ao obter resposta: {e}")
