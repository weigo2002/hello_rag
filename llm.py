from langchain_community.llms.llamacpp import LlamaCpp
from langchain import hub


def get_llm(model_path: str):
    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=-1,
        max_tokens=500,
        n_ctx=2048,
        seed=42,
        verbose=False
    )
    return llm

def get_prompt_template():
    return hub.pull("jclemens24/rag-prompt")