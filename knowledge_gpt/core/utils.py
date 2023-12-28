from typing import List
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.docstore.document import Document

#from langchain.chat_models import ChatOpenAI
from langchain.chat_models import AzureChatOpenAI
from knowledge_gpt.core.debug import FakeChatModel
from langchain.chat_models.base import BaseChatModel

import os

def pop_docs_upto_limit(
    query: str, chain: StuffDocumentsChain, docs: List[Document], max_len: int
) -> List[Document]:
    """Pops documents from a list until the final prompt length is less
    than the max length."""

    token_count: int = chain.prompt_length(docs, question=query)  # type: ignore

    while token_count > max_len and len(docs) > 0:
        docs.pop()
        token_count = chain.prompt_length(docs, question=query)  # type: ignore

    return docs

# def get_llm(model: str, **kwargs) -> BaseChatModel:
#     if model == "debug":
#         return FakeChatModel()
 
#     if "gpt" in model:
#         print(os.environ.get("AZUREAI_API_KEY"))
#         print(os.environ.get("AZUREAI_API_BASE"))
#         print(os.environ.get("AZUREAI_API_VERSION"))
#         llm = AzureOpenAI(
#             cache=None,
#             verbose=True,
#             model_name="text-davinci-003",
#             top_p=1,
#             openai_api_key=os.environ.get("AZUREAI_API_KEY"),
#             openai_api_base=os.environ.get("AZUREAI_API_BASE"),
#             openai_api_version=os.environ.get("AZUREAI_API_VERSION"),
#             deployment_name="text-davinci-003",
#         )
#         return llm
 
#     raise NotImplementedError(f"Model {model} not supported!")

def get_llm(model: str, **kwargs) -> BaseChatModel:
    if model == "debug":
        return FakeChatModel()

    if "gpt" in model:
        return AzureChatOpenAI(model_name=model, **kwargs)  # type: ignore

    raise NotImplementedError(f"Model {model} not supported!")
