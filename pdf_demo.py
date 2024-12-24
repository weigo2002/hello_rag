import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import (
    SpacyTextSplitter
)
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

from embedding import get_embeddings
from llm import get_llm, get_prompt_template
from vector_store import get_vector_store

def create_collection(collection_name: str, client: QdrantClient):

    if not client.collection_exists(collection_name):
        vector_config = VectorParams(
            size=512,
            distance=Distance.COSINE
        )
        client.create_collection(collection_name, vector_config)

def process_pdf(file_path: str) -> list[str]:
    pdf_loader = PyPDFLoader(file_path, extract_images=False)
    text_splitter = SpacyTextSplitter(
        chunk_size=512,
        chunk_overlap=128,
        pipeline="zh_core_web_sm",
    )
    pdf_contents = pdf_loader.load()
    pdf_text = "\n".join([page.page_content for page in pdf_contents])
    print(f"PDF文档的总字符数: {len(pdf_text)}")

    chunks = text_splitter.split_text(pdf_text)
    print(f"分割的文本chunk数量: {len(chunks)}")

    return chunks

if __name__ == "__main__":
    os.environ["qdrant_api_key"] = "<API KEY>"
    os.environ["qdrant_host"] = "https://d6645da3-45f6-4bd1-a4cf-41114d64fb7e.us-east4-0.gcp.cloud.qdrant.io"

    model_name = "qwen2.5"
    embedding_model = "BAAI/bge-small-zh-v1.5"
    collection_name = "pdf_demo"

    client = QdrantClient(
        url=os.environ["qdrant_host"],
        api_key=os.environ["qdrant_api_key"],
    )

    create_collection(collection_name, client)
    print(f"Collection created: {collection_name}")

    embedding = get_embeddings(embedding_model)
    chunks = process_pdf("海尔智家股份有限公司2024年第三季度报告.pdf")
    vector_store = get_vector_store(client, collection_name, embedding)
    print("start adding vectors")
    vector_store.add_texts(chunks)

    llm = get_llm(model_name)

    prompt = get_prompt_template()

    rag_chain = ({
        "context": vector_store.as_retriever(),
        "question": RunnablePassthrough()
    } | prompt
      | llm
      | StrOutputParser())
    result = rag_chain.invoke("海尔本报告期内净利润，营业利润，经营活动现金流")
    print(result)