from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain import hub
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from typing_extensions import TypedDict
from typing import List
from langchain.schema import Document
from langgraph.graph import END, StateGraph
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv

load_dotenv()

local_llm = "llama3"
tavily_api_key = os.getenv("TAVILY_API_KEY")

app = FastAPI()

# Add middleware to enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # Allow requests from any origin (you might want to restrict this in production)
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    # Raise an error if no files were uploaded
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    results = []
    temp_dir = os.path.join(os.getcwd(), "temp")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    for file in files:
        contents = await file.read()

        temp_file_path = os.path.join(temp_dir, file.filename)
        with open(f"{temp_file_path}", "wb") as f:
            f.write(contents)

        try:
            loader = PyPDFLoader(temp_file_path)
            data = loader.load()
            print(f"Data loaded for {file.filename}")
        except Exception as e:
            print(f"Failed to load {file.filename}: {str(e)}")
        finally:
            os.remove(temp_file_path)

        results.append({"filename": file.filename, "file_length": len(contents)})
    return results
