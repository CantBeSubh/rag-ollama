from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List

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
    results = []
    for file in files:
        contents = await file.read()  # Read the contents of the uploaded file
        # Process the PDF file here (you can use libraries like PyPDF2 or pdfplumber)
        # For demonstration purposes, let's just return the filename and length of each file
        results.append({"filename": file.filename, "file_length": len(contents)})
    return results
