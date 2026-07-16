import os
import io
import json
from typing import List, Optional
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pdf2image import convert_from_bytes
import easyocr
from google import genai
from google.genai import types

# Load environment variables from .env
load_dotenv()

if not os.environ.get("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY is not set in the environment or .env file.")

# Initialize FastAPI app
app = FastAPI(
    title="Syllabus Extraction API",
    description="Extracts structured curriculum information from syllabus PDFs using EasyOCR and Gemini."
)

# Initialize EasyOCR Reader globally once to cache model weights in memory/GPU
print("Initializing EasyOCR reader...")
reader = easyocr.Reader(['en'], gpu=True)

# Initialize the Google GenAI Client
client = genai.Client()

# ==========================================
# Pydantic Schemas for Structured JSON Output
# ==========================================

class UnitBlueprint(BaseModel):
    unitNumber: str = Field(description="Roman numeral or digit representing the unit section (e.g., I, II, 1, 2).")
    unitTitle: str = Field(description="The formal title of the unit.")
    suggestedHours: int = Field(description="The number of lecture or class hours allocated to this unit.")
    coreConcepts: List[str] = Field(description="List of specific, individual topics, principles, or subjects taught in this unit.")
    caseStudiesIncluded: bool = Field(description="True if the text explicitly states case studies are included for this unit, otherwise False.")

class PracticalExerciseBlueprint(BaseModel):
    exerciseNumber: int = Field(description="The chronological index/number of the experiment or exercise.")
    title: str = Field(description="The title or description of the practical task.")
    focusArea: str = Field(description="The overarching domain or conceptual framework this exercise belongs to.")

class LearningResourcesBlueprint(BaseModel):
    textbooks: List[str] = Field(default_factory=list, description="List of primary assigned textbooks.")
    references: List[str] = Field(default_factory=list, description="List of secondary reference books or manuals.")
    digitalResources: List[str] = Field(default_factory=list, description="List of relevant URLs, software, or digital platforms.")

class SyllabusObject(BaseModel):
    courseCode: str = Field(description="The unique alphanumeric identifier for the course (e.g., 24CST33).")
    courseName: str = Field(description="The complete name of the course.")
    category: str = Field(description="Course classification abbreviation (e.g., ES, MC, PC).")
    prerequisites: List[str] = Field(default_factory=list, description="List of prerequisite courses required, or ['Nil'] if none.")
    preamble: str = Field(description="The introductory rationale or overview text for the course.")
    units: List[UnitBlueprint] = Field(default_factory=list, description="List of academic breakdown units. Leave empty if it's a practical lab course.")
    practicalExercises: List[PracticalExerciseBlueprint] = Field(default_factory=list, description="List of exercises or lab experiments. Leave empty if it's a pure theory course.")
    learningResources: LearningResourcesBlueprint = Field(description="Associated course literature, books, and references.")

# The root response structure containing an array of syllabus objects
class SyllabusExtractionResponse(BaseModel):
    courses: List[SyllabusObject] = Field(description="A collection of structured syllabus data extracted from the document text.")

# ==========================================
# Core Processing Logic
# ==========================================

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Converts PDF bytes to images and performs OCR across all pages."""
    try:
        print("Converting incoming PDF bytes to images...")
        pages = convert_from_bytes(pdf_bytes, dpi=300)
        total_pages = len(pages)
        print(f"Successfully converted PDF. Total pages to process: {total_pages}")
        
        full_text_list = []
        for index, page in enumerate(pages):
            page_num = index + 1
            print(f"Processing page {page_num}/{total_pages} via EasyOCR...")
            
            # Convert PIL image to numpy array for EasyOCR processing
            page_np = np.array(page)
            results = reader.readtext(page_np, detail=0)
            
            page_text = "\n".join(results)
            full_text_list.append(f"--- PAGE {page_num} ---\n{page_text}\n\n")
            
        return "".join(full_text_list)
    except Exception as e:
        raise RuntimeError(f"OCR Pipeline failed: {str(e)}")

def map_text_to_syllabus_objects(raw_ocr_text: str) -> SyllabusExtractionResponse:
    """Uses the new google-genai SDK with response_schema to enforce structured JSON output."""
    prompt = (
        "You are an expert curriculum data parser. Take the following messy OCR output extracted "
        "from a university course syllabus document and organize it completely into a structured, "
        "clean array of syllabus objects following the provided schema. Match specific concepts, "
        "units, or laboratory assignments perfectly to the correct course."
    )
    
    try:
        # Request structure validation directly from the Gemini API
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, raw_ocr_text],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SyllabusExtractionResponse,
                temperature=0.1,  # Low temperature preserves high factual fidelity to source text
            ),
        )
        
        # The response text is structurally guaranteed to fit our Pydantic schema perfectly
        parsed_data = SyllabusExtractionResponse.model_validate_json(response.text)
        return parsed_data
        
    except Exception as e:
        raise RuntimeError(f"LLM Structure Mapping failed: {str(e)}")

# ==========================================
# API Route Endpoints
# ==========================================

@app.post("/extract-syllabus/", response_model=SyllabusExtractionResponse)
async def extract_syllabus(file: UploadFile = File(...)):
    """
    Upload a multi-page syllabus PDF document. The endpoint processes the file via OCR,
    structures the layout contents using Gemini, and returns an array of cleanly formatted 
    course objects.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF file.")
    
    try:
        pdf_bytes = await file.read()
        
        # Step 1: Execute OCR Text Extraction
        ocr_text = extract_text_from_pdf_bytes(pdf_bytes)
        
        # Step 2: Route structured mapping payload directly to Gemini
        structured_syllabus = map_text_to_syllabus_objects(ocr_text)
        
        return structured_syllabus
        
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))

if __name__ == "__main__":
    import uvicorn
    # Start app server locally on port 8000
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)