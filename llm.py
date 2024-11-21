from langchain import hub
from langchain_ollama import OllamaLLM


def get_llm(model_name: str):
    return OllamaLLM(model=model_name)

def get_prompt_template():
    return hub.pull("jclemens24/rag-prompt")