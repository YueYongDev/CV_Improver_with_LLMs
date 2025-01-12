# Streamlit
# Other libraries
import glob
import os
import warnings

import streamlit as st
import tiktoken
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
# document loader
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.embeddings import ZhipuAIEmbeddings
# FAISS vector database
from langchain_community.vectorstores import FAISS
# Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

# Data Directories: where temp files and vectorstores will be saved
from app_constants import TMP_DIR

warnings.filterwarnings("ignore", category=FutureWarning)


def langchain_document_loader(file_path):
    """Load and split a PDF file in Langchain.
    Parameters:
        - file_path (str): path of the file.
    Output:
        - documents: list of Langchain Documents."""

    if file_path.endswith(".pdf"):
        loader = PDFMinerLoader(file_path=file_path)
    else:
        st.error("You can only upload .pdf files!")

    # 1. Load and split documents
    documents = loader.load_and_split()

    # 2. Update the metadata: add document number to metadata
    for i in range(len(documents)):
        documents[i].metadata = {
            "source": documents[i].metadata["source"],
            "doc_number": i,
        }

    return documents


def delte_temp_files():
    """delete temp files from TMP_DIR"""
    files = glob.glob(TMP_DIR.as_posix() + "/*")
    for f in files:
        try:
            os.remove(f)
        except:
            pass


def save_uploaded_file(uploaded_file):
    """Save the uploaded file (output of the Streamlit File Uploader widget) to TMP_DIR."""

    temp_file_path = ""
    try:
        temp_file_path = os.path.join(TMP_DIR.as_posix(), uploaded_file.name)
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.read())
        return temp_file_path
    except Exception as error:
        st.error(f"An error occured: {error}")

    return temp_file_path


def tiktoken_tokens(documents, model="gpt-3.5-turbo-0125"):
    """Use tiktoken (tokeniser for OpenAI models) to return a list of token length per document."""

    # Get the encoding used by the model.
    encoding = tiktoken.encoding_for_model(model)

    # Calculate the token length of documents
    tokens_length = [len(encoding.encode(doc)) for doc in documents]

    return tokens_length


def select_embeddings_model(LLM_service="OpenAI"):
    """Select the Embeddings model: OpenAIEmbeddings or GoogleGenerativeAIEmbeddings."""

    if LLM_service == "OpenAI":
        embeddings = OpenAIEmbeddings(api_key=st.session_state.api_key)
    if LLM_service == "ZhiPu":
        embeddings = ZhipuAIEmbeddings(model="embedding-3", api_key=st.session_state.api_key)
    if LLM_service == "Qwen":
        embeddings = DashScopeEmbeddings(model="text-embedding-v2", dashscope_api_key=st.session_state.api_key)
    if LLM_service == "Ollama":
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
    # deepseek 暂不支持embedding
    # if LLM_service == "DeepSeek":
    #     embeddings = OpenAIEmbeddings(api_key=st.session_state.api_key,
    #                                   base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    #                                   model="text-embedding-v1")

    return embeddings


def create_vectorstore(embeddings, documents):
    """Create a Faiss vector database."""
    vector_store = FAISS.from_documents(documents=documents, embedding=embeddings)

    return vector_store


def Vectorstore_backed_retriever(
        vectorstore, search_type="similarity", k=4, score_threshold=None
):
    """create a vectorsore-backed retriever
    Parameters:
        search_type: Defines the type of search that the Retriever should perform.
            Can be "similarity" (default), "mmr", or "similarity_score_threshold"
        k: number of documents to return (Default: 4)
        score_threshold: Minimum relevance threshold for similarity_score_threshold (default=None)
    """
    search_kwargs = {}
    if k is not None:
        search_kwargs["k"] = k
    if score_threshold is not None:
        search_kwargs["score_threshold"] = score_threshold

    retriever = vectorstore.as_retriever(
        search_type=search_type, search_kwargs=search_kwargs
    )
    return retriever


def OpenAIEmbeddings_retriever(base_retriever, api_provider="OpenAI", api_key=None, top_n=4):
    # 根据选择的提供者初始化 OpenAIEmbeddings
    embeddings = select_embeddings_model(api_provider)
    # 使用 embeddings 创建压缩器
    compressor = EmbeddingsFilter(embeddings=embeddings, top_k=top_n)

    # 创建一个基于 ContextualCompression 的检索器
    retriever_openai = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )
    return retriever_openai


def retrieval_main():
    """Create a Langchain retrieval, which includes document loaders to upload the resume,
    embeddings to create a numerical representation of the text, FAISS vector database to store the embeddings,
    and CohereRerank retriever to find the most relevant documents.
    """

    # 1. Delete old temp files from TMP directory.
    delte_temp_files()

    if st.session_state.uploaded_file is not None:
        # 2. Save uploaded_file to TMP directory.
        saved_file_path = save_uploaded_file(st.session_state.uploaded_file)

        # 3. Load documents with Langchain loaders
        documents = langchain_document_loader(saved_file_path)
        st.session_state.documents = documents

        # 4. Embeddings
        embeddings = select_embeddings_model(st.session_state.LLM_provider)
        # 5. Create a Faiss vector database
        try:
            st.session_state.vector_store = create_vectorstore(
                embeddings=embeddings, documents=documents
            )

            # 6. Create CohereRerank retriever
            base_retriever = Vectorstore_backed_retriever(
                st.session_state.vector_store, "similarity", k=min(4, len(documents))
            )

            # 7. Use OpenAI Embeddings for reranking
            st.session_state.retriever = OpenAIEmbeddings_retriever(
                base_retriever=base_retriever,
                api_provider=st.session_state.LLM_provider,
                api_key=st.session_state.api_key,
                top_n=min(2, len(documents)),
            )

        except Exception as error:
            st.error(f"An error occured:\n {error}")

    else:
        st.error("Please upload a resume!")
        st.stop()
