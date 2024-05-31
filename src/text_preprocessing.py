from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def read_documents(directory):
  """
  Loads PDF documents from the specified directory.
  """
  file_loader = PyPDFDirectoryLoader(directory)
  documents = file_loader.load()
  return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=150):
  """
  Splits documents into chunks using RecursiveCharacterTextSplitter.
  """
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  return text_splitter.split_documents(documents)

def create_embeddings(model_path="sentence-transformers/all-mpnet-base-v2", documents=None):
  """
  Creates document embeddings using HuggingFaceEmbeddings.
  """
  embeddings = HuggingFaceEmbeddings(
      model_name=model_path, encode_kwargs={'normalize_embeddings': False}  # Experiment with normalization
  )
  if documents:
      db = FAISS.from_documents(documents, embeddings)
      retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 3})
      return embeddings, retriever
  else:
      return embeddings
