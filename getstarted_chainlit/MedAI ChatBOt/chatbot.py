# from langchain import RetrievalQA
from langchain.chains import VectorDBQA
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
# from langchain_community.chat_models import Ollama
from langchain_community.llms.ollama import Ollama
from langchain.schema import SystemMessage, HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from get_embedding_function import get_embedding_function
import os
import pdfplumber

# Step 1: Load and parse your PDF files
pdf_folder_path = "datasets"
pdf_files = [os.path.join(pdf_folder_path, file) for file in os.listdir(pdf_folder_path) if file.endswith(".pdf")]

# Use PyPDFLoader to load PDFs
documents = []
for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    documents.extend(loader.load())
    # text = ""
    # with pdfplumber.open(pdf_file) as pdf:
    #     for page in pdf.pages:
    #         text += page.extract_text()

# Step 2: Create embeddings for your documents using HuggingFace embeddings
embedding_model = get_embedding_function()
document_embeddings = embedding_model.embed_documents([doc.page_content for doc in documents])

# Step 3: Store embeddings in FAISS vector store
vectorstore = FAISS.from_documents(documents, embedding_model)

# Step 4: Setup the retriever from FAISS
retriever = vectorstore.as_retriever()

# Step 5: Define the Ollama LLM
llm = Ollama(base_url="http://test1.dgx.saveetha.in:8080/v1", model="llama3.1")

# Step 6: Create a RAG model with the Ollama LLM and FAISS retriever
rag_chain = VectorDBQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

# Step 7: Define the prompt template
prompt_template = PromptTemplate(
    input_variables=["query"],
    template="""
    You are a healthcare chatbot.
    Based on the context retrieved, answer the following question:
    {query}
    """
)

# Step 8: Input messages for the chatbot
messages = [
    SystemMessage(content="You are a healthcare chatbot"),
    HumanMessage(content="List the symptoms of Heart Valve Disease")
]

# Extract the query from HumanMessage
query = messages[-1].content

# Step 9: Get the response from the RAG model
response = rag_chain.run(prompt_template.format(query=query))

# Output the response
print(response)
