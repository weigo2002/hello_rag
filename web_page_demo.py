from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from downloader import download_and_split_contents
from embedding import get_embeddings
from llm import get_llm, get_prompt_template


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == '__main__':
    content_url = "https://kbourne.github.io/chapter1.html"
    embedding_model = "sentence-transformers/all-mpnet-base-v2"
    llm_model = "vanilj/Phi-4"

    embedding = get_embeddings(embedding_model)
    retriever =  download_and_split_contents(content_url, embedding)
    llm = get_llm(llm_model)
    prompt = get_prompt_template()

    rag_chain = ({
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    } | prompt
      | llm
      | StrOutputParser())
    output = rag_chain.invoke("What are the advantages of using RAG?")
    print(output)