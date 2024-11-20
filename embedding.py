from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embeddings(model_name:str):
    model_kwargs = {"device":"cpu"}
    encode_kwargs = {"normalize_embeddings": False}

    hf_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    return hf_embeddings