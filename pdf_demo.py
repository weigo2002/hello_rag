from click import prompt
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from torch.nn.functional import embedding

from embedding import get_embeddings
from llm import get_llm, get_prompt_template


def process_pdf(file_path: str, embedding: Embeddings):
    pdf_loader = PyPDFLoader(file_path, extract_images=False)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=128
    )
    pdf_contents = pdf_loader.load()
    pdf_text = "\n".join([page.page_content for page in pdf_contents])
    print(f"PDF文档的总字符数: {len(pdf_text)}")

    chunks = text_splitter.split_text(pdf_text)
    print(f"分割的文本chunk数量: {len(chunks)}")

    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embedding,
    )

    return vector_store.as_retriever()

if __name__ == "__main__":
    model_name = "qwen2.5"
    embedding_model = "BAAI/bge-small-zh-v1.5"

    embedding = get_embeddings(embedding_model)
    retriever = process_pdf("test.pdf", embedding)
    llm = get_llm(model_name)

    prompt = get_prompt_template()

    rag_chain = ({
        "context": retriever,
        "question": RunnablePassthrough()
    } | prompt
      | llm
      | StrOutputParser())
    result = rag_chain.invoke("请总结一下报告的中心内容")
    print(result)