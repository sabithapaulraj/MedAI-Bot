# # from langchain_community.embeddings.ollama import OllamaEmbeddings
# from langchain_community.embeddings import OllamaEmbeddings
# # from langchain_community.embeddings.bedrock import BedrockEmbeddings
# # from langchain_aws import BedrockEmbeddings

# def get_embedding_function():
#     embeddings = OllamaEmbeddings(
#         embeddings = OllamaEmbeddings(model="nomic-embed-text")
#     )
#     # credentials_profile_name="default", region_name="us-east-1"
#     return embeddings

from langchain_community.embeddings import OllamaEmbeddings

def get_embedding_function():
    # Initialize OllamaEmbeddings with the correct arguments
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings
