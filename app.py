from fastapi import FastAPI
from models import load_mistral_model, create_text_generation_pipeline  # Import for completeness
from text_processing import read_documents, split_documents, create_embeddings
from rag_chain import create_prompt_template
from answering import create_rag_chain, answer_question

app = FastAPI()

# Load models (replace with your paths and configurations)
model, tokenizer = load_mistral_model()
# Optional text generation pipeline (not used in this case)
# mistral_llm = create_text_generation_pipeline(model, tokenizer)
documents_directory = "documents/"  # Replace with your PDF directory
rag_chain = create_rag_chain(model, tokenizer, documents_directory)

@app.post("/answer")
async def answer(question: str):
  """
  API endpoint to answer questions based on the documents in the specified directory.
  """
  answer = answer_question(rag_chain, question)
  return {"answer": answer}

if __name__ == "__main__":
  import uvicorn
  uvicorn.run("app:app", host="0.0.0.0", port=8000)  # Change host and port as needed
