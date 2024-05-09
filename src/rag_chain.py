from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain

def create_prompt_template():
  """
  Defines the prompt template for guiding the LLM.
  """
  prompt_template = """
  ### [INST] Instruction: Answer the question based on your fantasy football knowledge. Here is context to help:

  {context}

  ### QUESTION:
  {question} [/INST]
  """
  return PromptTemplate(input_variables=["context", "question"], template=
