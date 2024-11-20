from langchain_community.document_loaders import WebBaseLoader
import bs4
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_experimental.text_splitter import SemanticChunker

def download_and_split_contents(url: str, embedding_model: HuggingFaceEmbeddings):
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )),
    )
    docs = loader.load()
    text_splitter = SemanticChunker(embedding_model)
    splits = text_splitter.split_documents(docs)
    vector_store = Chroma.from_documents(documents=splits, embedding=embedding_model)
    return vector_store.as_retriever()