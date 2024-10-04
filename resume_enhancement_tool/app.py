import os
import gradio as gr
import tensorflow as tf
from huggingface_hub import InferenceClient
from tensorflow.keras.models import load_model
from transformers import pipeline
import fitz  # PyMuPDF for PDF handling
from docx import Document  # For handling MS Word documents
import io  # To handle in-memory binary streams

# Step 1: Initialize Hugging Face Inference Client for LLaMA
def initialize_inference_client():
    try:
        client = InferenceClient(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            token=os.getenv("HUGGING_FACE_KEY")  # Ensure API key is set properly
        )
        return client
    except Exception as e:
        print(f"Error initializing InferenceClient: {e}")
        return None

# Step 2: Initialize Hugging Face NER pipeline
def initialize_ner_pipeline():
    try:
        ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
        return ner_pipeline
    except Exception as e:
        print(f"Error loading Hugging Face NER model: {e}")
        return None

# Step 3: Load custom resume generation model (.h5 file)
def load_resume_model():
    try:
        return load_model("resume_enhanced_generator_model.keras")
    except Exception as e:
        print(f"Error loading resume generation model: {e}")
        return None

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_binary_content):
    text = ""
    try:
        # Open the PDF from binary content
        pdf_document = fitz.open(stream=pdf_binary_content, filetype="pdf")
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
    except Exception as e:
        return f"Error extracting text from PDF: {e}"
    return text

# Function to extract text from a Word document
def extract_text_from_word(docx_binary_content):
    text = ""
    try:
        # Use io.BytesIO to handle the binary content of the DOCX file
        docx_stream = io.BytesIO(docx_binary_content)
        doc = Document(docx_stream)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        return f"Error extracting text from Word document: {e}"
    return text

# Function to enhance resume using the custom .h5 model
def enhance_with_custom_model(resume_text, resume_model):
    try:
        # Assuming the .h5 model takes text input and generates enhancement suggestions
        predictions = resume_model.predict([resume_text])
        enhancements = " ".join(predictions)  # Convert predictions to text (adjust as needed)
        return enhancements
    except Exception as e:
        return f"Error using the custom .h5 model: {e}"

# Function to optimize resume based on job title
def optimize_resume(resume_text, job_title, client):
    prompt = f"Optimize the following resume for the job title '{job_title}':\n\n{resume_text}"
    responses = []
    try:
        for message in client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            stream=True,
        ):
            responses.append(message.choices[0].delta.content)
    except Exception as e:
        return f"Error during model inference: {e}"
    
    return ''.join(responses)

# Function to process the resume and job title inputs
def process_resume(file, job_title, client, resume_model):
    try:
        # Use the file object directly with file.name and open the file as needed
        file_name = file.name
        
        if file_name.endswith(".pdf"):
            # Open and read the binary PDF content
            with open(file_name, "rb") as f:
                resume_text = extract_text_from_pdf(f.read())
        elif file_name.endswith(".docx"):
            # Open and read the binary DOCX content
            with open(file_name, "rb") as f:
                resume_text = extract_text_from_word(f.read())
        else:
            # Assume it's a text file, open and read it as UTF-8 text
            with open(file_name, "r", encoding="utf-8") as f:
                resume_text = f.read()
        
        # Step 1: Use the custom .h5 model to enhance the resume
        enhanced_resume = enhance_with_custom_model(resume_text, resume_model)

        # Step 2: Optimize the enhanced resume using the LLaMA model
        optimized_resume = optimize_resume(enhanced_resume, job_title, client)
        
        return optimized_resume
    except Exception as e:
        return f"Error processing resume: {e}"

# Initialize external resources
client = initialize_inference_client()
ner_pipeline = initialize_ner_pipeline()
resume_model = load_resume_model()

# Gradio Interface
interface = gr.Interface(
    fn=lambda file, job_title: process_resume(file, job_title, client, resume_model),
    inputs=[
        gr.File(label="Upload your resume (Word or PDF)"),
        gr.Textbox(lines=1, placeholder="Enter the job title...", label="Job Title"),
    ],
    outputs=gr.Textbox(label="Optimized Resume", lines=20),
    title="Resume Enhancement Tool",
    description="Upload your resume and specify a job title to optimize your resume for that position."
)

# Launch the Gradio app
interface.launch(share=True)
