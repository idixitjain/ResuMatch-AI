# Import necessary libraries
import streamlit as st
import os
import time
import json
import logging
import random
import uuid
import sqlite3
import requests
import re
import PyPDF2
import docx2txt
import threading
from datetime import datetime
import io
import traceback
from PIL import Image
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Tuple, Optional

# Set up page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="ResuMatch",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Rest of imports and initial setup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL = "llama3-70b-8192"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Download necessary NLTK data
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')

    return True

# Initialize resources
nltk_loaded = download_nltk_resources()
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk

# Configure API keys
def get_api_key():
    """Get the Groq API key from various sources with better error handling."""
    # Try to get from secrets, then environment, then fallback
    api_key = None
    
    # Try getting from Streamlit secrets first
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
        if api_key:
            logger.info("Using Groq API key from Streamlit secrets")
            return api_key
    except Exception as e:
        logger.warning(f"Could not access Streamlit secrets: {e}")
    
    # Try environment variable next
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        logger.info("Using Groq API key from environment variable")
        return api_key
    
    # Use fallback key as last resort
    fallback_key = "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"
    logger.warning("Using fallback Groq API key")
    return fallback_key

GROQ_API_KEY = get_api_key()

# Update the color scheme for better dark mode compatibility
def load_css():
    """
    Load custom CSS for improved styling
    """
    return """
    <style>
        /* Design System Variables */
        :root {
            --primary: #6366F1;        /* Indigo for primary actions */
            --primary-light: #818CF8;  /* Light indigo for highlights */
            --primary-dark: #4F46E5;   /* Dark indigo for hover states */
            --secondary: #10B981;      /* Emerald green for success states */
            --accent: #F59E0B;         /* Amber for warnings/notifications */
            --dark-bg: #111827;        /* Dark slate for background */
            --card-bg: #1F2937;        /* Slightly lighter for cards */
            --hover-bg: #374151;       /* Hover state for cards/buttons */
            --text-primary: #F9FAFB;   /* Very light gray for main text */
            --text-secondary: #E5E7EB; /* Light gray for secondary text */
            --text-muted: #9CA3AF;     /* Muted gray for less important text */
            --error: #EF4444;          /* Red for errors */
            --success: #10B981;        /* Green for success */
            --warning: #F59E0B;        /* Amber for warnings */
            --border: rgba(255, 255, 255, 0.1);  /* Subtle borders */
            --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            --radius-sm: 0.375rem;
            --radius-md: 0.5rem;
            --radius-lg: 0.75rem;
            --radius-full: 9999px;
            --spacing-1: 0.25rem;
            --spacing-2: 0.5rem;
            --spacing-3: 0.75rem;
            --spacing-4: 1rem;
            --spacing-6: 1.5rem;
            --spacing-8: 2rem;
            --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            --font-display: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            --font-serif: 'Georgia', serif;
            --transition: all 0.2s ease-in-out;
        }
        
        /* Base Styles */
        .stApp {
            background-color: var(--dark-bg);
            color: var(--text-primary);
            font-family: var(--font-sans);
        }
        
        h1, h2, h3, h4, h5, h6 {
            font-family: var(--font-display);
            color: var(--text-primary);
            font-weight: 600;
            margin-bottom: var(--spacing-4);
            letter-spacing: -0.025em;
        }
        
        h1 {
            font-size: 2.25rem;
            line-height: 2.5rem;
            margin-bottom: var(--spacing-6);
        }
        
        h2, .stSubheader {
            font-size: 1.5rem !important;
            line-height: 2rem;
            margin-top: var(--spacing-8) !important;
            margin-bottom: var(--spacing-4) !important;
            border-bottom: 1px solid var(--border);
            padding-bottom: var(--spacing-2);
        }
        
        h3 {
            font-size: 1.25rem;
            line-height: 1.75rem;
            margin-top: var(--spacing-6);
            margin-bottom: var(--spacing-3);
            font-weight: 600;
        }
        
        p, span, div, li {
            color: var(--text-secondary);
        }
        
        a {
            color: var(--primary-light);
            text-decoration: none;
            transition: var(--transition);
        }
        
        a:hover {
            color: var(--primary);
            text-decoration: underline;
        }
        
        /* Card Component */
        .card {
            background-color: var(--card-bg);
            border-radius: var(--radius-md);
            padding: var(--spacing-6);
            margin-bottom: var(--spacing-6);
            border: 1px solid var(--border);
            box-shadow: var(--shadow-sm);
            transition: var(--transition);
        }
        
        .card:hover {
            box-shadow: var(--shadow-md);
            border-color: var(--primary-light);
        }
        
        /* Skill Tags */
        .skill-tag {
            display: inline-block;
            padding: var(--spacing-1) var(--spacing-3);
            margin: var(--spacing-1);
            border-radius: var(--radius-full);
            font-size: 0.875rem;
            font-weight: 500;
            transition: var(--transition);
        }
        
        .matching-skill {
            background-color: rgba(16, 185, 129, 0.15);
            color: #34D399;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        
        .matching-skill:hover {
            background-color: rgba(16, 185, 129, 0.25);
            transform: translateY(-1px);
        }
        
        .missing-skill {
            background-color: rgba(239, 68, 68, 0.15);
            color: #F87171;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }
        
        .missing-skill:hover {
            background-color: rgba(239, 68, 68, 0.25);
            transform: translateY(-1px);
        }
        
        /* Match Score */
        .match-score {
            display: inline-block;
            padding: var(--spacing-2) var(--spacing-6);
            border-radius: var(--radius-full);
            font-size: 1.5rem;
            font-weight: 700;
            text-align: center;
            transition: var(--transition);
            box-shadow: var(--shadow-sm);
        }
        
        .match-score:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
        }
        
        .high-match {
            background-color: rgba(16, 185, 129, 0.15);
            color: #34D399;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        
        .medium-match {
            background-color: rgba(245, 158, 11, 0.15);
            color: #FBBF24;
            border: 1px solid rgba(245, 158, 11, 0.3);
        }
        
        .low-match {
            background-color: rgba(239, 68, 68, 0.15);
            color: #F87171;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }
        
        /* Improved Animations */
        .animate-fade-in {
            animation: fadeIn 0.5s ease-in-out;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes pulse {
            0% { opacity: 0.6; transform: scale(0.98); }
            50% { opacity: 1; transform: scale(1); }
            100% { opacity: 0.6; transform: scale(0.98); }
        }
        
        .animate-pulse {
            animation: pulse 2s infinite ease-in-out;
        }
        
        /* Cover Letter Styling */
        .cover-letter {
            background-color: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: var(--spacing-6);
            white-space: pre-line;
            font-family: var(--font-serif);
            line-height: 1.6;
            color: var(--text-primary);
            max-height: 500px;
            overflow-y: auto;
        }
        
        /* File Uploader */
        .stFileUploader {
            padding: var(--spacing-4);
            border: 2px dashed var(--border);
            border-radius: var(--radius-md);
            transition: var(--transition);
        }
        
        .stFileUploader:hover {
            border-color: var(--primary-light);
        }
        
        /* Button Styling */
        .stButton button {
            background-color: var(--primary) !important;
            color: white !important;
            border: none !important;
            padding: var(--spacing-2) var(--spacing-4) !important;
            border-radius: var(--radius-sm) !important;
            font-weight: 500 !important;
            transition: var(--transition) !important;
            font-size: 1rem !important;
        }
        
        .stButton button:hover {
            background-color: var(--primary-dark) !important;
            box-shadow: var(--shadow-md) !important;
            transform: translateY(-1px) !important;
        }
        
        .stButton button:active {
            transform: translateY(1px) !important;
        }
        
        /* Primary button */
        button[kind="primary"] {
            background-color: var(--primary) !important;
        }
        
        /* Secondary button */
        button[kind="secondary"] {
            background-color: var(--card-bg) !important;
            border: 1px solid var(--border) !important;
        }
        
        /* Download Buttons */
        .download-buttons {
            display: flex;
            gap: var(--spacing-2);
            margin-top: var(--spacing-4);
        }
        
        .download-buttons button {
            flex: 1;
            background-color: var(--card-bg) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border) !important;
        }
        
        .download-buttons button:hover {
            background-color: var(--hover-bg) !important;
            border-color: var(--primary-light) !important;
        }
        
        /* Select Input */
        .stSelectbox {
            margin-bottom: var(--spacing-4);
        }
        
        .stSelectbox > div > div > div {
            background-color: var(--card-bg);
            border-color: var(--border);
            color: var(--text-primary);
        }
        
        /* Text Input */
        .stTextInput > div > div > input {
            background-color: var(--card-bg);
            border-color: var(--border);
            color: var(--text-primary);
            border-radius: var(--radius-sm);
            padding: var(--spacing-2) var(--spacing-3);
        }
        
        /* Progress Bar */
        .stProgress > div > div > div {
            background-color: var(--primary);
        }
        
        /* Metrics Styling */
        [data-testid="stMetricValue"] {
            font-size: 1.5rem !important;
            font-weight: 700 !important;
            color: var(--primary) !important;
        }
        
        [data-testid="stMetricLabel"] {
            color: var(--text-muted) !important;
        }
        
        /* Status Messages */
        .stAlert {
            padding: var(--spacing-4) !important;
            border-radius: var(--radius-md) !important;
            margin-bottom: var(--spacing-4) !important;
        }
        
        /* Success Message */
        [data-baseweb="notification"] {
            background-color: rgba(16, 185, 129, 0.1) !important;
            border-left-color: var(--success) !important;
        }
        
        /* Error Message */
        .stException {
            background-color: rgba(239, 68, 68, 0.1) !important;
            border-left-color: var(--error) !important;
            color: #F87171 !important;
        }
        
        /* Warning Message */
        .stWarning {
            background-color: rgba(245, 158, 11, 0.1) !important;
            border-left-color: var(--warning) !important;
        }
        
        /* Loading Animation */
        @keyframes dots {
            0%, 20% { color: rgba(0,0,0,0); text-shadow: 0.25em 0 0 rgba(0,0,0,0), 0.5em 0 0 rgba(0,0,0,0); }
            40% { color: var(--text-primary); text-shadow: 0.25em 0 0 rgba(0,0,0,0), 0.5em 0 0 rgba(0,0,0,0); }
            60% { text-shadow: 0.25em 0 0 var(--text-primary), 0.5em 0 0 rgba(0,0,0,0); }
            80%, 100% { text-shadow: 0.25em 0 0 var(--text-primary), 0.5em 0 0 var(--text-primary); }
        }
        
        .loading:after {
            content: ".";
            animation: dots 1.5s steps(5, end) infinite;
        }
    </style>
    """

# Improved resume text extraction with better error handling
@st.cache_data
def extract_text_from_pdf(file):
    try:
        # Get file bytes from UploadedFile object
        file_bytes = file.getvalue()
        
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = []
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            text.append(page_text.strip())
        
        # Only clean the text after extraction is complete
        full_text = '\n\n'.join(text)
        # Enhanced text cleaning while preserving important structures
        full_text = re.sub(r'\s+', ' ', full_text)  # Replace multiple spaces with single space
        full_text = re.sub(r'[^\x00-\x7F]+', '', full_text)  # Remove non-ASCII chars
        full_text = re.sub(r'[\r\n]+', '\n', full_text)  # Normalize line breaks
        
        logger.info(f"Successfully extracted {len(pdf_reader.pages)} pages from PDF")
        return full_text.strip()
    except Exception as e:
        logger.error(f"PDF extraction error: {str(e)}")
        logger.error(traceback.format_exc())
        return ""

# Enhanced DOCX text extraction with improved error handling
@st.cache_data
def extract_text_from_docx(file):
    try:
        # Get file bytes from UploadedFile object
        file_bytes = file.getvalue()
        
        text = docx2txt.process(io.BytesIO(file_bytes))
        if not text.strip():
            logger.warning("docx2txt returned empty text, trying alternative extraction")
            # Fall back to a more basic extraction if needed
            # This could be expanded with alternative DOCX parsers
        
        # Improved text cleaning while preserving structure
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
        text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII chars
        text = re.sub(r'[\r\n]{3,}', '\n\n', text)  # Normalize excessive line breaks
        
        logger.info("Successfully extracted text from DOCX")
        return text.strip()
    except Exception as e:
        logger.error(f"DOCX extraction error: {str(e)}")
        logger.error(traceback.format_exc())
        return ""

def get_file_extension(file):
    return os.path.splitext(file.name)[1].lower()

def read_resume(uploaded_file):
    if uploaded_file is None:
        return ""
    
    file_extension = get_file_extension(uploaded_file)
    
    try:
        # Process based on file type
        if file_extension == ".pdf":
            resume_text = extract_text_from_pdf(uploaded_file)
        elif file_extension in [".docx", ".doc"]:
            resume_text = extract_text_from_docx(uploaded_file)
        elif file_extension == ".txt":
            resume_text = str(uploaded_file.getvalue(), "utf-8")
        else:
            logger.error(f"Unsupported file format: {file_extension}")
            return ""
        
        # If we got text but it's too short, there might be an issue with extraction
        if len(resume_text.strip()) < 100:
            logger.warning(f"Extracted resume text is suspiciously short ({len(resume_text)} chars)")
            
        return resume_text
    except Exception as e:
        logger.error(f"Error reading file: {str(e)}")
        logger.error(traceback.format_exc())
        return ""

# Improved API error handling and retry mechanism with exponential backoff
def call_groq_api(messages, model=DEFAULT_MODEL, function_schema=None, function_call=None, max_retries=3, timeout=15):
    """
    Call the Groq API with improved error handling and optimized performance.
    
    Args:
        messages (list): List of message objects
        model (str, optional): The model to use. Defaults to DEFAULT_MODEL.
        function_schema (dict, optional): Function schema for function calling
        function_call (dict, optional): Function call object
        max_retries (int, optional): Maximum number of retries. Defaults to 3.
        timeout (int, optional): Request timeout in seconds. Defaults to 15.
        
    Returns:
        dict: The API response
    """
    api_key = get_api_key()
    if not api_key:
        logging.error("No API key found")
        return None
        
    # Log the API request for debugging
    logging.info(f"Calling Groq API with model: {model}")
    logging.info(f"Request to Groq API: {model}, message count: {len(messages)}")
    
    # Early validation to check if API key looks valid
    if not api_key.startswith("gsk_"):
        logging.error("Invalid API key format. Groq API keys should start with 'gsk_'")
        return None
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
    }
    
    # Add function calling if provided
    if function_schema and function_call:
        payload["functions"] = [function_schema]
        payload["function_call"] = function_call
    
    for attempt in range(max_retries):
        try:
            # Use a shorter timeout for faster failure detection
            response = requests.post(url, json=payload, headers=headers, timeout=timeout - (attempt * 2))
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                logging.error("Authentication error with Groq API. Check your API key.")
                # Authentication errors won't be resolved with retries
                return None
            elif response.status_code == 429:
                logging.warning("Rate limit hit. Implementing exponential backoff.")
                # For rate limits, use exponential backoff
                wait_time = min((2 ** attempt) * 0.5, 10)
                time.sleep(wait_time)
                continue
            else:
                logging.error(f"API error: {response.status_code} - {response.text}")
                # Other errors might resolve with retries, but use a shorter wait time
                if attempt < max_retries - 1:
                    time.sleep(min(1 + attempt, 3))
                continue
                
        except requests.exceptions.Timeout:
            logging.warning(f"Request timed out (attempt {attempt + 1}/{max_retries})")
            # Don't wait as long for timeouts
            if attempt < max_retries - 1:
                time.sleep(0.5)
                
        except requests.exceptions.RequestException as e:
            logging.error(f"Request error: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(min(1 + attempt, 3))
                
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(min(1 + attempt, 3))
    
    logging.warning("All API call attempts failed")
    return None

# Enhanced resume parsing with multiple techniques and better error handling
def parse_resume_with_llm(resume_text):
    # First try with LLM
    try:
        logger.info("Starting LLM-based resume parsing")
        if len(resume_text.strip()) < 50:
            logger.warning("Resume text is too short for LLM parsing")
            return enhanced_fallback_resume_parsing(resume_text)
            
        function_schema = {
            "name": "parse_resume",
            "description": "Parse a resume into structured data",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The full name of the candidate"},
                    "email": {"type": "string", "description": "The email address of the candidate"},
                    "phone": {"type": "string", "description": "The phone number of the candidate"},
                    "skills": {
                        "type": "array",
                        "description": "A comprehensive list of all technical and soft skills mentioned in the resume",
                        "items": {"type": "string"}
                    },
                    "education": {
                        "type": "array",
                        "description": "Details about educational background",
                        "items": {
                            "type": "object",
                            "properties": {
                                "degree": {"type": "string", "description": "The degree or certification obtained"},
                                "institution": {"type": "string", "description": "The school, college or university name"},
                                "year": {"type": "string", "description": "The year or years of study/graduation"}
                            }
                        }
                    },
                    "work_experience": {
                        "type": "array",
                        "description": "Details about work experience",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string", "description": "Job title or position"},
                                "company": {"type": "string", "description": "Company or organization name"},
                                "duration": {"type": "string", "description": "Time period worked at this position"},
                                "description": {"type": "string", "description": "Brief description of responsibilities and achievements"}
                            }
                        }
                    }
                },
                "required": ["name", "skills"]
            }
        }
        
        messages = [
            {"role": "system", "content": "You are a helpful resume parsing assistant specialized in extracting structured information from resumes. Extract as much detail as possible accurately, focusing especially on technical skills and experience."},
            {"role": "user", "content": f"Parse the following resume into structured data. Be thorough and extract all relevant information, especially all technical skills and technologies mentioned:\n\n{resume_text}"}
        ]
        
        logger.info("Calling LLM API for resume parsing")
        response_data = call_groq_api(
            messages=messages,
            function_schema=function_schema,
            function_call={"name": "parse_resume"}
        )
        
        if not response_data:
            logger.warning("No response from API, falling back to local processing")
            return enhanced_fallback_resume_parsing(resume_text)
            
        if "choices" not in response_data or len(response_data["choices"]) == 0:
            logger.warning("Invalid response from API, falling back to local processing")
            return enhanced_fallback_resume_parsing(resume_text)
        
        function_call = response_data["choices"][0]["message"].get("function_call", {})
        
        if not function_call or "arguments" not in function_call:
            logger.warning("No function call in response, falling back to local processing")
            return enhanced_fallback_resume_parsing(resume_text)
            
        try:
            parsed_data = json.loads(function_call["arguments"])
            logger.info("Successfully parsed resume with LLM")
            
            # Enhance with fallback methods for any missing fields
            fallback_data = enhanced_fallback_resume_parsing(resume_text)
            
            # Check and fill missing or empty fields
            if not parsed_data.get("email"):
                parsed_data["email"] = fallback_data.get("email", "")
                logger.info("Using fallback email data")
                
            if not parsed_data.get("phone"):
                parsed_data["phone"] = fallback_data.get("phone", "")
                logger.info("Using fallback phone data")
                
            # Combine skills from both sources for maximum coverage
            if not parsed_data.get("skills") or len(parsed_data.get("skills", [])) < 3:
                parsed_data["skills"] = fallback_data.get("skills", [])
                logger.info("Using fallback skills data")
            else:
                # Merge skills from both sources and remove duplicates
                combined_skills = parsed_data.get("skills", []) + fallback_data.get("skills", [])
                # Normalize skill names for deduplication
                normalized_skills = {}
                for skill in combined_skills:
                    key = skill.lower().strip()
                    if key and key not in normalized_skills:
                        normalized_skills[key] = skill  # Keep original capitalization
                
                parsed_data["skills"] = list(normalized_skills.values())
                logger.info(f"Combined skills list has {len(parsed_data['skills'])} items")
            
            return parsed_data
            
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON from LLM response")
            return enhanced_fallback_resume_parsing(resume_text)
    
    except Exception as e:
        logger.error(f"Error parsing resume with LLM: {str(e)}")
        logger.error(traceback.format_exc())
        return enhanced_fallback_resume_parsing(resume_text)

# Enhanced local resume parsing with improved skill detection
def enhanced_fallback_resume_parsing(resume_text):
    try:
        logger.info("Using enhanced fallback resume parsing")
        
        # Define comprehensive skills dictionary with categories
        common_skills = {
            "programming": ["python", "java", "javascript", "typescript", "c\\+\\+", "c#", "ruby", "php", "swift", "kotlin", "go", "rust", "scala", "perl", "r", "shell", "bash", "powershell"],
            "web": ["html", "css", "react", "angular", "vue", "node\\.js", "express", "django", "flask", "bootstrap", "tailwind", "jquery", "webpack", "sass", "less", "graphql", "rest", "soap", "json", "xml"],
            "database": ["sql", "nosql", "mongodb", "mysql", "postgresql", "sqlite", "oracle", "redis", "elasticsearch", "dynamodb", "cassandra", "mariadb", "neo4j", "firebase", "supabase"],
            "cloud": ["aws", "azure", "gcp", "docker", "kubernetes", "terraform", "cloudformation", "lambda", "s3", "ec2", "heroku", "netlify", "vercel", "digital ocean", "linode"],
            "tools": ["git", "github", "gitlab", "bitbucket", "jira", "confluence", "jenkins", "travis", "circleci", "maven", "gradle", "npm", "yarn", "pip", "junit", "selenium", "postman", "swagger"],
            "methodologies": ["agile", "scrum", "kanban", "waterfall", "tdd", "bdd", "devops", "ci/cd", "oop", "functional programming", "microservices", "soa", "rest", "solid principles"],
            "data": ["machine learning", "deep learning", "data analysis", "tensorflow", "pytorch", "pandas", "numpy", "scikit-learn", "tableau", "power bi", "hadoop", "spark", "etl", "statistics", "data mining", "neural networks", "nlp", "computer vision", "data science", "big data"],
            "design": ["photoshop", "illustrator", "indesign", "figma", "sketch", "xd", "ui/ux", "responsive design", "wireframing", "prototyping", "user research", "usability testing"],
            "soft_skills": ["leadership", "communication", "teamwork", "problem solving", "critical thinking", "time management", "project management", "adaptability", "creativity", "collaboration"],
            "languages": ["english", "spanish", "french", "german", "mandarin", "japanese", "russian", "arabic", "hindi", "portuguese", "italian", "chinese"],
            "frameworks": ["spring", "hibernate", ".net", "symfony", "laravel", "rails", "flutter", "xamarin", "next.js", "gatsby", "svelte", "jquery", "asp.net"],
            "testing": ["unit testing", "integration testing", "qa", "quality assurance", "jasmine", "mocha", "jest", "pytest", "testng", "selenium", "cypress", "playwright"],
            "office": ["excel", "powerpoint", "word", "outlook", "access", "sharepoint", "teams", "onedrive", "visio", "office 365", "google workspace"],
            "security": ["cybersecurity", "infosec", "penetration testing", "encryption", "authentication", "authorization", "oauth", "jwt", "ssl/tls", "firewall", "vpn"],
            "mobile": ["android", "ios", "react native", "swift", "kotlin", "objective-c", "mobile development", "pwa", "app development", "flutter", "xamarin"]
        }
        
        # Flatten skills list for searching while preserving categories
        all_skills = {}
        for category, skills in common_skills.items():
            for skill in skills:
                # Store skill with its category to improve output organization
                all_skills[skill] = category
        
        # Improved extraction with multiple pattern matching approaches
        found_skills = []
        
        # 1. Word boundary matching (most accurate)
        for skill, category in all_skills.items():
            if re.search(r'\b' + skill + r'\b', resume_text.lower()):
                # Capitalize skill name properly
                found_skill = ' '.join(word.capitalize() if word.lower() not in ['and', 'of', 'the', 'in', 'for', 'with', 'on', 'to'] else word for word in skill.split())
                if found_skill not in found_skills:  # Avoid duplicates
                    found_skills.append(found_skill)
                    
        # 2. Look for common technical abbreviations that might be missed
        tech_abbreviations = ["AI", "ML", "NLP", "API", "UI", "UX", "CI/CD", "OOP", "AWS", "GCP", "SQL", "NoSQL", "CSS", "HTML", "JS", "TS"]
        for abbr in tech_abbreviations:
            if re.search(r'\b' + abbr + r'\b', resume_text):
                if abbr not in found_skills:
                    found_skills.append(abbr)
        
        # 3. Look for skills with special characters or formats that might be missed
        special_format_skills = {
            "c\\+\\+": "C++",
            "c#": "C#",
            "\\.net": ".NET",
            "node\\.js": "Node.js",
            "react\\.js": "React.js",
            "vue\\.js": "Vue.js",
        }
        
        for pattern, skill_name in special_format_skills.items():
            if re.search(r'\b' + pattern + r'\b', resume_text.lower()):
                if skill_name not in found_skills:
                    found_skills.append(skill_name)
        
        # Enhanced name extraction
        name = extract_name(resume_text)
        
        # Better email extraction pattern
        email = extract_email(resume_text)
        
        # Better phone extraction pattern
        phone = extract_phone(resume_text)
        
        # Enhanced education extraction
        education = extract_education(resume_text)
        
        # Extract work experience
        work_experience = extract_work_experience(resume_text)
        
        return {
            "name": name,
            "email": email,
            "phone": phone,
            "skills": found_skills,
            "education": education,
            "work_experience": work_experience
        }
    except Exception as e:
        logger.error(f"Error in enhanced fallback parsing: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Return minimal data to avoid complete failure
        return {
            "name": "",
            "email": extract_email(resume_text),
            "phone": "",
            "skills": [],
            "education": [],
            "work_experience": []
        }

# Helper function to extract name using NER and patterns
def extract_name(text):
    # First try with common patterns
    name_patterns = [
        r'(?:name:?\s*)([\w\s]+)',
        r'^([\w\s]+)$',  # First line if it's just a name
        r'^([A-Z][a-z]+ [A-Z][a-z]+)'  # Capitalized first and last name at start
    ]
    
    for pattern in name_patterns:
        name_match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if name_match:
            potential_name = name_match.group(1).strip()
            if len(potential_name.split()) >= 2 and len(potential_name.split()) <= 4:  # First and last name (with potential middle)
                return potential_name
    
    # Try NLTK's named entity recognition
    try:
        tokens = word_tokenize(text.split('\n')[0])  # Only look at first few lines
        tagged = pos_tag(tokens)
        named_entities = ne_chunk(tagged)
        
        person_names = []
        for chunk in named_entities:
            if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                person_name = ' '.join(c[0] for c in chunk)
                if len(person_name.split()) >= 2:  # At least first and last name
                    person_names.append(person_name)
        
        if person_names:
            return person_names[0]
    except Exception as e:
        logger.error(f"Error extracting name with NER: {str(e)}")
    
    # Fallback: look for capitalized words at the beginning
    lines = text.strip().split('\n')
    for line in lines[:5]:  # Check first 5 lines
        line = line.strip()
        if line and 2 <= len(line.split()) <= 4:  # Likely a name if not too long or short
            # Check if words are capitalized (typical for names)
            words = line.split()
            if all(word[0].isupper() for word in words if word):
                return line
    
    return ""

# Improved helper functions for fallback parsing
def extract_email(text):
    # More robust email pattern
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    matches = re.findall(email_pattern, text)
    
    if matches:
        # Return the first email that doesn't look like a sample (doesn't contain example.com)
        for email in matches:
            if 'example' not in email.lower() and 'sample' not in email.lower():
                return email
        # If all look like samples, return the first one anyway
        return matches[0]
    
    return ""

def extract_phone(text):
    phone_patterns = [
        r'(?:\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}',  # US/Canada format
        r'\d{10}',  # Simple 10 digits
        r'\d{3}[\s.-]\d{3}[\s.-]?\d{4}',  # Separated by spaces, dots or hyphens
        r'\+\d{1,2}\s\d{10}',  # International format
        r'\+\d{1,2}\s\d{3}\s\d{3}\s\d{4}'  # International with spaces
    ]
    
    for pattern in phone_patterns:
        matches = re.findall(pattern, text)
        if matches:
            # Filter out potential false positives (dates, etc.)
            for match in matches:
                # Check if it's a phone number (exclude dates like 2021-03-01)
                if not re.match(r'^\d{4}[\s.-]\d{2}[\s.-]\d{2}$', match):
                    # Format the phone number consistently
                    digits_only = re.sub(r'\D', '', match)
                    if len(digits_only) >= 10:  # Valid phone numbers should have at least 10 digits
                        # Format as (XXX) XXX-XXXX for US numbers or keep original format
                        if len(digits_only) == 10:
                            return f"({digits_only[0:3]}) {digits_only[3:6]}-{digits_only[6:10]}"
                        else:
                            return match
    
    return ""

# Education extraction with improved patterns
def extract_education(text):
    education = []
    
    # Find education section
    edu_section = None
    education_headers = ["education", "academic background", "academic qualification", "qualification", "academic history"]
    
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if any(header in line.lower() for header in education_headers):
            start_idx = i
            # Find the end of the education section (next section or end of text)
            end_idx = len(lines)
            for j in range(i+1, len(lines)):
                if any(keyword in lines[j].lower() for keyword in ["experience", "skills", "projects", "employment", "certification"]):
                    end_idx = j
                    break
            
            edu_section = '\n'.join(lines[start_idx:end_idx])
            break
    
    if not edu_section:
        edu_section = text  # If we can't find a specific section, search the whole resume
    
    # Degree patterns
    degree_patterns = [
        r"(?:Bachelor|Master|Ph\.?D\.?|B\.?S\.?|M\.?S\.?|M\.?B\.?A\.?|B\.?A\.?|B\.?E\.?|B\.?Tech\.?|M\.?Tech\.?)[\s.](?:of|in)?[\s.]([A-Za-z\s,]+)",
        r"([A-Za-z\s]+(?:Degree|Certificate|Diploma))",
        r"(?:Bachelor|Master|Ph\.?D\.?|Doctorate|Associate)(?:'s|s)?(?:\sDegree)?(?:\sin|\sof)?\s([A-Za-z\s,]+)"
    ]
    
    # Institution patterns
    institution_patterns = [
        r"(?:University|College|Institute|School)\s(?:of|at)?\s([A-Za-z\s,]+)",
        r"([A-Za-z\s]+(?:University|College|Institute|School))"
    ]
    
    # Year pattern
    year_pattern = r"(?:19|20)\d{2}(?:\s*[-–]\s*(?:19|20)\d{2}|present|current|ongoing)?"
    
    # Extract degrees and institutions
    degrees = []
    for pattern in degree_patterns:
        matches = re.findall(pattern, edu_section, re.IGNORECASE)
        degrees.extend([match.strip() for match in matches if match.strip()])
    
    institutions = []
    for pattern in institution_patterns:
        matches = re.findall(pattern, edu_section, re.IGNORECASE)
        institutions.extend([match.strip() for match in matches if match.strip()])
    
    years = re.findall(year_pattern, edu_section)
    
    # Try to match degrees with institutions and years
    if degrees and institutions:
        # If we have equal numbers, assume they correspond
        if len(degrees) == len(institutions):
            for i in range(len(degrees)):
                year = years[i] if i < len(years) else ""
                education.append({
                    "degree": degrees[i],
                    "institution": institutions[i],
                    "year": year
                })
        else:
            # Otherwise, use proximity to match
            for degree in degrees:
                degree_pos = edu_section.lower().find(degree.lower())
                closest_inst = None
                min_dist = float('inf')
                
                for inst in institutions:
                    inst_pos = edu_section.lower().find(inst.lower())
                    dist = abs(degree_pos - inst_pos)
                    if dist < min_dist:
                        min_dist = dist
                        closest_inst = inst
                
                # Find closest year
                closest_year = ""
                min_year_dist = float('inf')
                for year in years:
                    year_pos = edu_section.lower().find(year.lower())
                    dist = abs(degree_pos - year_pos)
                    if dist < min_year_dist:
                        min_year_dist = dist
                        closest_year = year
                
                education.append({
                    "degree": degree,
                    "institution": closest_inst or "",
                    "year": closest_year
                })
    elif degrees:
        # If we only have degrees
        for i, degree in enumerate(degrees):
            year = years[i] if i < len(years) else ""
            education.append({
                "degree": degree,
                "institution": "",
                "year": year
            })
    elif institutions:
        # If we only have institutions
        for i, institution in enumerate(institutions):
            year = years[i] if i < len(years) else ""
            education.append({
                "degree": "",
                "institution": institution,
                "year": year
            })
    
    return education

# Work experience extraction
def extract_work_experience(text):
    work_experience = []
    
    # Find work experience section
    exp_section = None
    experience_headers = ["experience", "employment", "work history", "professional experience", "work experience"]
    
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if any(header in line.lower() for header in experience_headers):
            start_idx = i
            # Find the end of the experience section (next section or end of text)
            end_idx = len(lines)
            for j in range(i+1, len(lines)):
                if any(keyword in lines[j].lower() for keyword in ["education", "skills", "projects", "certification", "reference"]):
                    end_idx = j
                    break
            
            exp_section = '\n'.join(lines[start_idx:end_idx])
            break
    
    if not exp_section:
        return work_experience
    
    # Split into individual experiences (look for date ranges as separators)
    date_pattern = r"(?:19|20)\d{2}\s*[-–]\s*(?:(?:19|20)\d{2}|present|current|ongoing)"
    experience_blocks = re.split(date_pattern, exp_section)
    
    if len(experience_blocks) <= 1:
        # If we couldn't split by dates, try to split by company/title patterns
        company_patterns = [
            r"\n([A-Z][A-Za-z\s,]+(?:Inc\.?|LLC|Ltd\.?|Corporation|Corp\.?|Company|Co\.?))",
            r"\n([A-Z][A-Za-z\s,]+)\n"
        ]
        
        for pattern in company_patterns:
            experience_blocks = re.split(pattern, exp_section)
            if len(experience_blocks) > 1:
                break
    
    # Extract details from each block
    for i, block in enumerate(experience_blocks[1:] if len(experience_blocks) > 1 else experience_blocks):
        title = ""
        company = ""
        duration = ""
        description = block.strip()
        
        # Extract job title (usually capitalized at beginning of line)
        title_match = re.search(r"(?:^|\n)([A-Z][A-Za-z\s]+(?:Developer|Engineer|Manager|Analyst|Designer|Consultant|Director|Specialist|Lead))", block)
        if title_match:
            title = title_match.group(1).strip()
        
        # Extract company name
        company_match = re.search(r"(?:at|for|with)\s+([A-Z][A-Za-z\s,]+(?:Inc\.?|LLC|Ltd\.?|Corporation|Corp\.?|Company|Co\.?)?)", block)
        if company_match:
            company = company_match.group(1).strip()
        else:
            # Try alternative pattern
            company_match = re.search(r"([A-Z][A-Za-z\s,]+(?:Inc\.?|LLC|Ltd\.?|Corporation|Corp\.?|Company|Co\.?))", block)
            if company_match:
                company = company_match.group(1).strip()
        
        # Extract duration
        duration_match = re.search(date_pattern, block)
        if duration_match:
            duration = duration_match.group(0).strip()
        
        # Only add if we have at least a title or company
        if title or company:
            work_experience.append({
                "title": title,
                "company": company,
                "duration": duration,
                "description": description
            })
    
    return work_experience

# Improved matching score calculation with fuzzy matching
def calculate_match_score(resume_skills, required_skills, preferred_skills):
    try:
        # Convert all skills to lowercase for case-insensitive comparison
        resume_skills_lower = [skill.lower().strip() for skill in resume_skills if skill.strip()]
        required_skills_lower = [skill.lower().strip() for skill in required_skills if skill.strip()]
        preferred_skills_lower = [skill.lower().strip() for skill in preferred_skills if skill.strip()]
        
        if not resume_skills_lower or not required_skills_lower:
            logger.warning("Empty skills list detected in matching algorithm")
            if not resume_skills_lower:
                logger.warning("Resume skills list is empty")
            if not required_skills_lower:
                logger.warning("Required skills list is empty")
            return 0
        
        # Calculate required skills match with improved matching logic
        required_matches = 0
        for req_skill in required_skills_lower:
            # Check for exact matches first
            if req_skill in resume_skills_lower:
                required_matches += 1
                continue
                
            # Check for partial matches (for composite skills like "machine learning")
            for resume_skill in resume_skills_lower:
                # Check if the skill is a part of another skill or vice versa
                if (req_skill in resume_skill) or (resume_skill in req_skill):
                    required_matches += 0.8  # Partial match gets 80% weight
                    break
                    
                # Check for skills that are typically related or equivalent
                # Example: "react" and "react.js" or "javascript" and "js"
                if is_skill_equivalent(req_skill, resume_skill):
                    required_matches += 0.9  # Equivalent match gets 90% weight
                    break
        
        # Calculate weighted score for required skills (70% of total)
        if required_skills_lower:
            required_score = (required_matches / len(required_skills_lower)) * 0.7
        else:
            required_score = 0
        
        # Calculate preferred skills match using the same approach
        preferred_matches = 0
        if preferred_skills_lower:
            for pref_skill in preferred_skills_lower:
                # Exact match
                if pref_skill in resume_skills_lower:
                    preferred_matches += 1
                    continue
                    
                # Partial match
                for resume_skill in resume_skills_lower:
                    if (pref_skill in resume_skill) or (resume_skill in pref_skill):
                        preferred_matches += 0.8
                        break
                        
                    if is_skill_equivalent(pref_skill, resume_skill):
                        preferred_matches += 0.9
                        break
            
            preferred_score = (preferred_matches / len(preferred_skills_lower)) * 0.3
        else:
            preferred_score = 0
        
        # Calculate final score
        total_score = (required_score + preferred_score) * 100
        
        # Ensure the score is between 0 and 100
        final_score = max(0, min(100, total_score))
        return int(round(final_score))
    
    except Exception as e:
        logger.error(f"Error calculating match score: {str(e)}")
        logger.error(traceback.format_exc())
        return 30  # Default to a reasonable fallback value instead of 0

# Helper function to check if two skills are equivalent
def is_skill_equivalent(skill1, skill2):
    # Common equivalences in tech skills
    equivalences = {
        "javascript": ["js", "ecmascript"],
        "typescript": ["ts"],
        "python": ["py"],
        "java": ["jvm"],
        "react": ["reactjs", "react.js"],
        "node": ["nodejs", "node.js"],
        "angular": ["angularjs", "angular.js"],
        "vue": ["vuejs", "vue.js"],
        "machine learning": ["ml", "machine-learning"],
        "artificial intelligence": ["ai"],
        "amazon web services": ["aws"],
        "microsoft azure": ["azure"],
        "google cloud platform": ["gcp", "google cloud"],
        "postgresql": ["postgres"],
        "mongodb": ["mongo"],
        "c sharp": ["c#", "csharp"],
        "c plus plus": ["c++", "cplusplus"],
        "objective c": ["objective-c"],
        "ruby on rails": ["rails"],
        "version control": ["git", "svn"],
        "continuous integration": ["ci/cd", "ci", "cd"],
    }
    
    # Check direct equivalence
    for main_skill, alternatives in equivalences.items():
        if (skill1 == main_skill and skill2 in alternatives) or (skill2 == main_skill and skill1 in alternatives):
            return True
            
    # Check word similarity for longer phrases
    if len(skill1.split()) > 1 and len(skill2.split()) > 1:
        common_words = set(skill1.split()) & set(skill2.split())
        if len(common_words) >= 2:  # If they share at least 2 words
            return True
    
    return False

# Updated job descriptions with simplified formats
@st.cache_data
def get_job_descriptions():
    """
    Get job descriptions for different roles.
    
    Returns:
        dict: Dictionary of job descriptions for different roles
    """
    return {
        "Python Developer": """
Python Developer at Amazon
Responsibilities:
• Design, develop, and maintain scalable Python applications
• Write clean, maintainable, and efficient code
• Implement and optimize algorithms for large-scale data processing
• Collaborate with cross-functional teams to define features
• Perform code reviews and mentor junior developers
• Debug and fix production issues

Requirements:
• 3+ years of experience in Python development
• Proficiency in web frameworks such as Django or Flask
• Strong understanding of RESTful APIs and microservices architecture
• Experience with SQL and NoSQL databases
• Knowledge of AWS or other cloud services
• Familiarity with Git and CI/CD pipelines
• Excellent problem-solving and analytical skills
• Experience with data structures and algorithms
        """,
        
        "Data Scientist": """
Data Scientist at Microsoft
Responsibilities:
• Analyze and interpret complex data sets to identify patterns and trends
• Build predictive models using machine learning algorithms
• Develop and implement data collection and analysis methodologies
• Collaborate with product teams to translate business requirements into analytical solutions
• Communicate findings and insights to stakeholders
• Monitor model performance and improve models over time

Requirements:
• Advanced degree in Computer Science, Statistics, or related field
• 4+ years of experience in data science or related field
• Proficiency in Python, R, or similar language for data analysis
• Experience with machine learning libraries (scikit-learn, TensorFlow, PyTorch)
• Strong understanding of statistical analysis and experimental design
• Expertise in SQL and data visualization tools
• Experience with big data technologies (Spark, Hadoop)
• Excellent communication and presentation skills
        """,
        
        "Frontend Developer": """
Frontend Developer at Google
Responsibilities:
• Develop responsive and user-friendly web interfaces
• Implement UI/UX designs using HTML, CSS, and JavaScript
• Build and maintain front-end web applications using React
• Optimize applications for maximum speed and scalability
• Collaborate with designers, back-end developers, and product managers
• Ensure cross-browser compatibility and responsive design

Requirements:
• 3+ years of experience in frontend development
• Strong proficiency in JavaScript, HTML, CSS
• Experience with React, TypeScript, and modern JS libraries
• Knowledge of responsive design principles
• Understanding of CSS preprocessors (SASS, LESS)
• Familiarity with RESTful APIs and HTTP protocols
• Experience with version control systems like Git
• Strong problem-solving skills and attention to detail
        """,
        
        "Software Development Engineer": """
Software Development Engineer at Netflix
Responsibilities:
• Design, develop, and maintain scalable distributed systems
• Write high-quality, maintainable, and efficient code
• Build resilient cloud-based services and APIs
• Collaborate with cross-functional teams to deliver end-to-end solutions
• Participate in architectural discussions and code reviews
• Troubleshoot and resolve complex technical issues

Requirements:
• Bachelor's degree in Computer Science or equivalent experience
• 5+ years of experience in software development
• Strong proficiency in at least one programming language (Java, C++, Python, Go)
• Experience with distributed systems and microservices architecture
• Knowledge of data structures, algorithms, and system design
• Familiarity with cloud platforms (AWS, GCP, Azure)
• Understanding of containerization and orchestration (Docker, Kubernetes)
• Experience with continuous integration and deployment pipelines
        """
    }

# Improved match_resume_to_job function
def match_resume_to_job(resume_text, job_description, job_title):
    """
    Match resume text to job description and calculate match percentage.
    
    Args:
        resume_text (str): The extracted text from the resume
        job_description (str): The job description text
        job_title (str): The title of the job position
        
    Returns:
        dict: Match results including score, matching skills, missing skills, etc.
    """
    try:
        logging.info(f"Matching resume to job: {job_title}")
        
        # Parse the resume using LLM
        parsed_resume = parse_resume_with_llm(resume_text)
        
        if not parsed_resume:
            logging.warning("Failed to parse resume with LLM, using fallback method")
            parsed_resume = enhanced_fallback_resume_parsing(resume_text)
        
        # Extract skills from the job description
        job_skills = extract_skills_from_text(job_description)
        
        # Extract skills from the resume
        candidate_skills = set(parsed_resume.get("skills", []))
        
        # If LLM didn't extract skills, try extracting from text directly
        if not candidate_skills:
            extracted_skills = extract_skills_from_text(resume_text)
            candidate_skills = set(extracted_skills)
        
        # Calculate matching and missing skills
        matching_skills = []
        for job_skill in job_skills:
            # Check for exact match
            if job_skill.lower() in [s.lower() for s in candidate_skills]:
                matching_skills.append(job_skill)
                continue
                
            # Check for partial matches or synonyms
            for candidate_skill in candidate_skills:
                if is_skill_match(job_skill, candidate_skill):
                    matching_skills.append(job_skill)
                    break
        
        # List of skills in job description that are not in resume
        missing_skills = [skill for skill in job_skills if skill not in matching_skills]
        
        # Calculate match percentage
        if job_skills:
            match_percentage = int((len(matching_skills) / len(job_skills)) * 100)
        else:
            match_percentage = 0
        
        # Generate summary
        summary = generate_match_summary(match_percentage, matching_skills, missing_skills, job_title)
        
        return {
            "match_percentage": match_percentage,
            "matching_skills": matching_skills,
            "missing_skills": missing_skills,
            "parsed_resume": parsed_resume,
            "job_skills": job_skills,
            "summary": summary
        }
    except Exception as e:
        logging.error(f"Error matching resume to job: {str(e)}", exc_info=True)
        return None

def is_skill_match(job_skill, candidate_skill):
    """
    Check if a job skill matches a candidate skill, considering partial matches and synonyms.
    
    Args:
        job_skill (str): Skill from job description
        candidate_skill (str): Skill from candidate resume
        
    Returns:
        bool: True if skills match, False otherwise
    """
    job_skill_lower = job_skill.lower()
    candidate_skill_lower = candidate_skill.lower()
    
    # Check if one contains the other
    if job_skill_lower in candidate_skill_lower or candidate_skill_lower in job_skill_lower:
        # Avoid partial word matches (e.g., "C" shouldn't match "C++")
        if len(job_skill_lower) <= 2 or len(candidate_skill_lower) <= 2:
            return job_skill_lower == candidate_skill_lower
        return True
    
    # Check for common abbreviations
    abbreviations = {
        "javascript": "js",
        "typescript": "ts",
        "python": "py",
        "java": "j2ee",
        "react": "reactjs",
        "node": "nodejs",
        "aws": "amazon web services",
        "azure": "microsoft azure",
        "ui": "user interface",
        "ux": "user experience",
        "ml": "machine learning",
        "ai": "artificial intelligence",
        "ci": "continuous integration",
        "cd": "continuous deployment"
    }
    
    # Check if one is an abbreviation of the other
    if job_skill_lower in abbreviations and abbreviations[job_skill_lower] == candidate_skill_lower:
        return True
    if candidate_skill_lower in abbreviations and abbreviations[candidate_skill_lower] == job_skill_lower:
        return True
    
    # Calculate similarity score (simple approach)
    similarity = calculate_string_similarity(job_skill_lower, candidate_skill_lower)
    
    # Consider a match if similarity exceeds threshold
    return similarity > 0.8

def calculate_string_similarity(str1, str2):
    """
    Calculate similarity between two strings (0 to 1).
    
    Args:
        str1 (str): First string
        str2 (str): Second string
        
    Returns:
        float: Similarity score between 0 and 1
    """
    # Simple Jaccard similarity for quick comparison
    set1 = set(str1.lower())
    set2 = set(str2.lower())
    
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0

def generate_match_summary(match_percentage, matching_skills, missing_skills, job_title):
    """
    Generate a summary of the match results.
    
    Args:
        match_percentage (int): The calculated match percentage
        matching_skills (list): List of matching skills
        missing_skills (list): List of missing skills
        job_title (str): The job title
        
    Returns:
        str: A summary of the match results
    """
    if match_percentage >= 80:
        strength = "strong"
        action = "You are an excellent match for this position and should highlight your expertise in " + ", ".join(matching_skills[:3]) + " in your application."
    elif match_percentage >= 60:
        strength = "good"
        action = "Consider emphasizing your experience with " + ", ".join(matching_skills[:3]) + " while addressing how you can develop in areas like " + (", ".join(missing_skills[:2]) if missing_skills else "other relevant skills") + "."
    elif match_percentage >= 40:
        strength = "moderate"
        action = "You have some relevant skills, but may want to acquire experience in " + (", ".join(missing_skills[:3]) if missing_skills else "other relevant skills") + " to improve your candidacy."
    else:
        strength = "limited"
        action = "This position may not be an ideal match for your current skill set. Consider developing expertise in " + (", ".join(missing_skills[:3]) if missing_skills else "the required skills") + " or exploring positions that better match your experience."
    
    summary = f"You have a {strength} match ({match_percentage}%) with the {job_title} position. "
    
    if matching_skills:
        summary += f"Your strongest matching skills include {', '.join(matching_skills[:5])}. "
    
    if missing_skills:
        summary += f"You could improve your match by developing skills in {', '.join(missing_skills[:5])}. "
    
    summary += action
    
    return summary

# Generate a cover letter with Groq with better error handling and fallback
def generate_cover_letter(resume_text, job_title, job_description, company_name, candidate_name="", matching_skills=None, missing_skills=None):
    """
    Generate a cover letter based on the resume, job description, and company.
    
    Args:
        resume_text (str): The resume text
        job_title (str): The job title
        job_description (str): The job description
        company_name (str): The company name
        candidate_name (str): The candidate's name
        matching_skills (list): Skills that match the job requirements
        missing_skills (list): Skills that are missing from the candidate's profile
        
    Returns:
        str: The generated cover letter
    """
    if not company_name:
        company_name = "the company"
        
    if not matching_skills:
        matching_skills = []
        
    if not missing_skills:
        missing_skills = []
    
    logging.info(f"Generating cover letter for {job_title} at {company_name}")
    
    # Immediately check if API key is valid to avoid unnecessary retries
    api_key = get_api_key()
    if not api_key or api_key == "gsk_your_key_here":
        logging.warning("No valid API key found, using fallback cover letter generation immediately")
        return generate_fallback_cover_letter(candidate_name, company_name, job_title, matching_skills)
    
    # For faster performance, create a shortened resume text and job description
    shortened_resume = resume_text[:400] if len(resume_text) > 400 else resume_text
    shortened_job_desc = job_description[:250] if len(job_description) > 250 else job_description
    
    # Optimize prompt for shorter token count
    advanced_prompt = f"""
Generate a professional cover letter for {candidate_name or 'the candidate'}, applying for {job_title} at {company_name}.

Key information:
- Candidate skills: {', '.join(matching_skills[:6])}
- Job requirements: {', '.join(matching_skills[:4] + missing_skills[:2])}
- Job description summary: {shortened_job_desc}

Create a concise, personalized cover letter (300-350 words) with:
1. Engaging opening showing interest in the position
2. Paragraph highlighting how the candidate's skills match job requirements
3. Brief mention of company interest
4. Strong closing paragraph with call to action

Use a warm, professional tone and natural language.
"""
    
    # Try function calling first with more efficient retry logic
    max_retries = 2  # Reduce number of retries to reduce waiting time
    retry_count = 0
    
    # Define the function schema
    function_schema = {
        "name": "generate_cover_letter",
        "description": "Generate a professional cover letter",
        "parameters": {
            "type": "object",
            "properties": {
                "cover_letter": {
                    "type": "string",
                    "description": "The complete cover letter text"
                }
            },
            "required": ["cover_letter"]
        }
    }
    
    while retry_count < max_retries:
        try:
            logging.info(f"Calling Groq API with model: {DEFAULT_MODEL}")
            response = call_groq_api(
                messages=[
                    {"role": "system", "content": "You are an expert career coach who creates tailored cover letters."},
                    {"role": "user", "content": advanced_prompt}
                ],
                function_schema=function_schema,
                function_call={"name": "generate_cover_letter"},
                max_retries=1  # Reduce internal retries
            )
            
            if response and isinstance(response, dict) and "choices" in response:
                message = response["choices"][0]["message"]
                
                if "function_call" in message:
                    function_args = json.loads(message["function_call"]["arguments"])
                    cover_letter = function_args.get("cover_letter", "")
                    if cover_letter:
                        return cover_letter
            
            # If function calling failed, try standard completion with a shorter timeout
            logging.info("Function calling failed, attempting standard completion")
            response = call_groq_api(
                messages=[
                    {"role": "system", "content": "You are an expert career coach who creates tailored cover letters."},
                    {"role": "user", "content": advanced_prompt}
                ],
                max_retries=1  # Reduce internal retries
            )
            
            if response and isinstance(response, dict) and "choices" in response:
                content = response["choices"][0]["message"]["content"]
                if content:
                    return content
            
            retry_count += 1
            wait_time = 1.0 * (1.5 ** retry_count) * (0.8 + random.random() * 0.4)  # Faster backoff
            logging.info(f"Attempt {retry_count} failed, waiting {wait_time:.2f} seconds before retry")
            time.sleep(wait_time)
            
        except Exception as e:
            logging.error(f"Error in cover letter generation: {str(e)}")
            retry_count += 1
            if retry_count >= max_retries:
                break
                
            wait_time = 1.0 * (1.5 ** retry_count) * (0.8 + random.random() * 0.4)
            logging.info(f"Attempt {retry_count} failed, waiting {wait_time:.2f} seconds before retry")
            time.sleep(wait_time)
    
    # If all API attempts failed, use fallback cover letter generation
    logging.warning("All API attempts failed, using fallback cover letter generation")
    return generate_fallback_cover_letter(candidate_name, company_name, job_title, matching_skills)

def generate_fallback_cover_letter(candidate_name, company_name, job_title, matching_skills):
    """
    Generate a fallback cover letter when API calls fail. This is an optimized version
    that produces a good quality letter without API calls.
    
    Args:
        candidate_name (str): The candidate's name
        company_name (str): The company name
        job_title (str): The job title
        matching_skills (list): Skills that match the job requirements
        
    Returns:
        str: A basic cover letter
    """
    try:
        # Format current date
        current_date = datetime.now().strftime("%B %d, %Y")
        
        # Prepare clean inputs with defaults for empty values
        clean_name = candidate_name.strip() if candidate_name and candidate_name.strip() else "Applicant"
        clean_company = company_name.strip() if company_name and company_name.strip() else "the company"
        clean_job = job_title.strip() if job_title and job_title.strip() else "the position"
        
        # Get top skills (limited to 4 for conciseness)
        key_skills = matching_skills[:4] if matching_skills else ["relevant skills"]
        skills_text = ", ".join(key_skills)
        
        # Create template sections
        header = f"{current_date}\n\nDear Hiring Manager at {clean_company},"
        
        opening = f"I am writing to express my enthusiasm for the {clean_job} position at {clean_company}. With my background and expertise in {skills_text}, I am confident in my ability to make valuable contributions to your team."
        
        skills_section = f"Throughout my career, I have developed strong skills in {skills_text}, which directly align with the requirements outlined in your job description. These technical capabilities, combined with my problem-solving mindset, enable me to deliver high-quality solutions efficiently."
        
        company_interest = f"I am particularly drawn to {clean_company} because of its reputation for innovation and excellence in the industry. The opportunity to contribute to your team's success and grow professionally in such an environment is very exciting."
        
        closing = f"I would welcome the opportunity to discuss how my qualifications match your needs for the {clean_job} position. Thank you for considering my application."
        
        signature = f"Sincerely,\n{clean_name}"
        
        # Combine all sections
        cover_letter = f"{header}\n\n{opening}\n\n{skills_section}\n\n{company_interest}\n\n{closing}\n\n{signature}"
        
        return cover_letter
    except Exception as e:
        logging.error(f"Error generating fallback cover letter: {str(e)}")
        # Ultra minimal fallback in case of any error
        return f"Dear Hiring Manager at {company_name},\n\nI am writing to express my interest in the {job_title} position.\n\nSincerely,\n{candidate_name or 'Applicant'}"

def get_resume_text(file):
    """
    Extract text from resume file (PDF or DOCX).
    
    Args:
        file: The uploaded file object
        
    Returns:
        str: Extracted text from the resume
    """
    filename = file.name.lower()
    
    try:
        if filename.endswith('.pdf'):
            return extract_text_from_pdf(file)
        elif filename.endswith('.docx'):
            return extract_text_from_docx(file)
        else:
            logging.warning(f"Unsupported file format: {filename}")
            return ""
    except Exception as e:
        logging.error(f"Error extracting text from {filename}: {str(e)}", exc_info=True)
        return ""

def extract_skills_from_text(text):
    """
    Extract skills from text using a comprehensive skills dictionary.
    
    Args:
        text (str): The text to extract skills from
        
    Returns:
        list: List of extracted skills
    """
    # Comprehensive skills dictionary
    tech_skills = {
        # Programming Languages
        "python", "java", "javascript", "typescript", "c", "c++", "c#", "go", "golang", "ruby", 
        "php", "swift", "kotlin", "rust", "scala", "perl", "r", "dart", "bash", "powershell",
        "objective-c", "vba", "matlab", "groovy", "lua", "julia", "haskell", "erlang", "clojure",
        
        # Web Development
        "html", "css", "html5", "css3", "sass", "less", "bootstrap", "tailwind", "jquery",
        "react", "reactjs", "redux", "angular", "vue", "vuejs", "ember", "svelte", "nextjs", 
        "gatsby", "nuxtjs", "ajax", "xml", "json", "restful api", "graphql", "apollo", "symfony",
        "django", "flask", "laravel", "express", "spring", "spring boot", "jsp", "asp.net", 
        "node", "nodejs", "deno", "webpack", "babel", "responsive design", "pwa", "spa", "jamstack",
        
        # Mobile Development
        "android", "ios", "react native", "flutter", "xamarin", "ionic", "cordova", "swift ui",
        "jetpack compose", "kotlin multiplatform", "capacitor", "objective-c", "mobile development",
        
        # Data Science & Machine Learning
        "machine learning", "deep learning", "ai", "artificial intelligence", "data science",
        "data analysis", "data visualization", "pandas", "numpy", "scipy", "scikit-learn", 
        "tensorflow", "keras", "pytorch", "mxnet", "theano", "onnx", "nltk", "spacy", "gensim",
        "hugging face", "transformers", "computer vision", "nlp", "natural language processing",
        "recommender systems", "statistical analysis", "ab testing", "hypothesis testing",
        
        # Big Data
        "hadoop", "spark", "kafka", "airflow", "luigi", "flink", "storm", "hive", "pig", "hbase",
        "cassandra", "mongodb", "couchdb", "elasticsearch", "big data", "etl", "data pipeline",
        "data engineering", "data warehouse", "data lake", "data modeling",
        
        # Cloud & DevOps
        "aws", "azure", "gcp", "google cloud", "cloud computing", "devops", "ci/cd", "docker",
        "kubernetes", "k8s", "jenkins", "gitlab ci", "github actions", "travis ci", "terraform",
        "ansible", "puppet", "chef", "infrastructure as code", "serverless", "lambda", "s3",
        "ec2", "rds", "dynamo db", "cloudformation", "azure functions", "monitoring", "logging",
        "prometheus", "grafana", "elk", "microservices", "service mesh", "istio", "helm", 
        "openshift", "argo", "cloud native", "containers", "virtualization", "vmware", "vms",
        
        # Databases
        "sql", "mysql", "postgresql", "sqlite", "oracle", "sql server", "mariadb", "mongodb",
        "dynamodb", "cassandra", "redis", "couchbase", "neo4j", "firebase", "dax", "cosmosdb",
        "database design", "database modeling", "database administration", "database migration",
        "database performance", "database optimization", "caching", "indexing", "acid", 
        "oltp", "olap", "nosql", "rdbms",
        
        # Testing & QA
        "testing", "qa", "quality assurance", "unit testing", "integration testing", "e2e testing",
        "selenium", "cypress", "jest", "mocha", "chai", "sinon", "jasmine", "karma", "junit",
        "testng", "pytest", "rspec", "cucumber", "bdd", "tdd", "test automation", "performance testing",
        "load testing", "stress testing", "jmeter", "gatling", "postman", "swagger", "openapi",
        
        # Security
        "cybersecurity", "security", "encryption", "cryptography", "penetration testing", "pen testing",
        "ethical hacking", "vulnerability assessment", "security audit", "ssrf", "xss", "csrf",
        "sql injection", "ddos", "authentication", "authorization", "oauth", "oidc", "jwt",
        "kerberos", "ldap", "active directory", "ssl/tls", "https", "pki", "firewall", "vpn",
        "network security", "devsecops", "sast", "dast", "iast", "siem", "dlp", "iam",
        
        # Project Management & Methodologies
        "agile", "scrum", "kanban", "waterfall", "prince2", "itil", "pmp", "lean", "six sigma",
        "sdlc", "software development lifecycle", "jira", "confluence", "trello", "asana", 
        "project management", "requirements gathering", "user stories", "sprint planning",
        "retrospectives", "product owner", "scrum master", "stakeholder management",
        
        # Software Architecture
        "oop", "object oriented programming", "functional programming", "design patterns",
        "solid principles", "mvc", "mvvm", "clean architecture", "domain driven design",
        "microservices architecture", "soa", "event driven architecture", "cqrs", "event sourcing",
        "api design", "restful apis", "graphql apis", "grpc", "soap", "uml", "system design",
        "high availability", "fault tolerance", "load balancing", "scalability", "distributed systems",
        
        # Other Technical Skills
        "git", "github", "gitlab", "bitbucket", "svn", "mercurial", "regular expressions", "regex",
        "linux", "unix", "bash scripting", "shell scripting", "networking", "tcp/ip", "http",
        "websockets", "sockets", "algorithms", "data structures", "problem solving", "debugging",
        "performance optimization", "code review", "code quality", "static analysis", "ci",
        "continuous integration", "cd", "continuous deployment", "continuous delivery",
        
        # Soft Skills (often listed in technical resumes)
        "communication", "teamwork", "leadership", "problem solving", "critical thinking",
        "time management", "analytical skills", "attention to detail", "creativity", "mentoring",
        "collaboration", "adaptability", "customer service", "presentation skills"
    }
    
    # Normalize text for better matching
    text_lower = text.lower()
    
    # Remove punctuation and normalize whitespace
    for char in ".,;:!?()[]{}<>/\\\"'":
        text_lower = text_lower.replace(char, " ")
    text_lower = " ".join(text_lower.split())
    
    # Find all skills in the text
    found_skills = set()
    
    # First, look for exact matches with word boundaries
    for skill in tech_skills:
        # Add word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.add(skill)
    
    # Convert set to sorted list for consistent results
    return sorted(list(found_skills))

# Database setup
def init_db():
    """Initialize SQLite database with required tables"""
    conn = sqlite3.connect('resumatch.db')
    c = conn.cursor()
    
    # Create tables if they don't exist
    c.execute('''
    CREATE TABLE IF NOT EXISTS candidates (
        id TEXT PRIMARY KEY,
        name TEXT,
        email TEXT,
        phone TEXT,
        skills TEXT,
        education TEXT,
        work_experience TEXT,
        resume_text TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    c.execute('''
    CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY,
        title TEXT,
        company TEXT,
        description TEXT,
        skills TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    c.execute('''
    CREATE TABLE IF NOT EXISTS matches (
        id TEXT PRIMARY KEY,
        candidate_id TEXT,
        job_id TEXT,
        match_percentage INTEGER,
        matching_skills TEXT,
        missing_skills TEXT,
        summary TEXT,
        cover_letter TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (candidate_id) REFERENCES candidates (id),
        FOREIGN KEY (job_id) REFERENCES jobs (id)
    )
    ''')
    
    # Insert default jobs if none exist
    c.execute("SELECT COUNT(*) FROM jobs")
    count = c.fetchone()[0]
    
    if count == 0:
        # Add default jobs
        job_descriptions = get_job_descriptions()
        for title, description in job_descriptions.items():
            job_id = str(uuid.uuid4())
            company = "Sample Company"
            if "Amazon" in description:
                company = "Amazon"
            elif "Microsoft" in description:
                company = "Microsoft"
            elif "Google" in description:
                company = "Google"
            elif "Netflix" in description:
                company = "Netflix"
                
            # Extract skills from job description
            skills = json.dumps(extract_skills_from_text(description))
            
            c.execute(
                "INSERT INTO jobs (id, title, company, description, skills) VALUES (?, ?, ?, ?, ?)",
                (job_id, title, company, description, skills)
            )
    
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")

# Save candidate data to database
def save_candidate(parsed_resume, resume_text):
    conn = sqlite3.connect('resumatch.db')
    c = conn.cursor()
    
    candidate_id = str(uuid.uuid4())
    name = parsed_resume.get('name', '')
    email = parsed_resume.get('email', '')
    phone = parsed_resume.get('phone', '')
    skills = json.dumps(parsed_resume.get('skills', []))
    education = json.dumps(parsed_resume.get('education', []))
    work_experience = json.dumps(parsed_resume.get('work_experience', []))
    
    c.execute(
        "INSERT INTO candidates (id, name, email, phone, skills, education, work_experience, resume_text) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (candidate_id, name, email, phone, skills, education, work_experience, resume_text)
    )
    
    conn.commit()
    c.execute("SELECT id FROM candidates WHERE id = ?", (candidate_id,))
    result = c.fetchone()
    conn.close()
    
    return candidate_id if result else None

# Save match result to database
def save_match_result(candidate_id, job_title, match_result, cover_letter=None):
    conn = sqlite3.connect('resumatch.db')
    c = conn.cursor()
    
    # Get job_id based on title
    c.execute("SELECT id FROM jobs WHERE title = ?", (job_title,))
    job_result = c.fetchone()
    
    if not job_result:
        conn.close()
        return None
    
    job_id = job_result[0]
    match_id = str(uuid.uuid4())
    match_percentage = match_result.get('match_percentage', 0)
    matching_skills = json.dumps(match_result.get('matching_skills', []))
    missing_skills = json.dumps(match_result.get('missing_skills', []))
    summary = match_result.get('summary', '')
    
    c.execute(
        "INSERT INTO matches (id, candidate_id, job_id, match_percentage, matching_skills, missing_skills, summary, cover_letter) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (match_id, candidate_id, job_id, match_percentage, matching_skills, missing_skills, summary, cover_letter)
    )
    
    conn.commit()
    conn.close()
    
    return match_id

# Initialize the database when the app starts
init_db()

def main():
    # Load custom CSS
    st.markdown(load_css(), unsafe_allow_html=True)
    
    # Application header
    st.markdown(
        """
        <div style="text-align: center; padding-bottom: 1.5rem;">
            <h1>ResuMatch<span style="font-weight: 400; color: var(--primary-light);">AI</span></h1>
            <p style="font-size: 1.2rem; margin-top: -0.5rem; color: var(--text-secondary);">
                Upload your resume, select a job, and get instant feedback on your match!
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Initialize session state for database IDs
    if 'candidate_id' not in st.session_state:
        st.session_state.candidate_id = None
    if 'match_id' not in st.session_state:
        st.session_state.match_id = None
    
    # Create columns for layout without any empty containers
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Upload section with card styling
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Upload Your Resume")
        
        resume_file = st.file_uploader(
            "Choose a PDF or DOCX file", 
            type=["pdf", "docx"],
            help="Upload your resume in PDF or DOCX format"
        )
        
        job_options = ["Python Developer", "Data Scientist", "Frontend Developer", "Software Development Engineer"]
        selected_job = st.selectbox(
            "Select a Job Position", 
            job_options,
            help="Choose the job position you want to match your resume against"
        )
        
        analyze_button = st.button(
            "Analyze Resume", 
            type="primary",
            help="Click to analyze your resume and match it against the selected job"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if analyze_button:
            if resume_file is not None:
                try:
                    with st.spinner("Analyzing your resume..."):
                        # Get resume text
                        resume_text = get_resume_text(resume_file)
                        if not resume_text or len(resume_text.strip()) < 50:
                            st.error("Could not extract sufficient text from your resume. Please try a different file.")
                            st.stop()
                        
                        # Get job description
                        job_descriptions = get_job_descriptions()
                        job_description = job_descriptions.get(selected_job, "")
                        
                        if not job_description:
                            st.error(f"No description found for {selected_job}")
                            st.stop()
                        
                        # Match resume to job with progress indicator
                        progress_placeholder = st.empty()
                        progress_bar = progress_placeholder.progress(0)
                        
                        # Update progress
                        progress_bar.progress(25)
                        time.sleep(0.3)
                        
                        # Match resume to job
                        match_result = match_resume_to_job(resume_text, job_description, selected_job)
                        
                        progress_bar.progress(75)
                        time.sleep(0.3)
                        
                        if not match_result:
                            progress_placeholder.empty()
                            st.error("Failed to analyze resume. Please try again.")
                            st.stop()
                        
                        # Store parsed resume in database
                        parsed_resume = match_result.get("parsed_resume", {})
                        if parsed_resume:
                            candidate_id = save_candidate(parsed_resume, resume_text)
                            if candidate_id:
                                st.session_state.candidate_id = candidate_id
                                logger.info(f"Saved candidate with ID: {candidate_id}")
                        
                        # Store the match result in database
                        if st.session_state.candidate_id:
                            match_id = save_match_result(
                                st.session_state.candidate_id,
                                selected_job,
                                match_result
                            )
                            if match_id:
                                st.session_state.match_id = match_id
                                logger.info(f"Saved match result with ID: {match_id}")
                        
                        # Store results in session state
                        st.session_state.match_result = match_result
                        st.session_state.resume_text = resume_text
                        st.session_state.selected_job = selected_job
                        st.session_state.job_description = job_description
                        
                        # Complete progress and show success message
                        progress_bar.progress(100)
                        time.sleep(0.3)
                        progress_placeholder.empty()
                        
                        # Success message with database info
                        st.success("Resume analyzed and saved to database!")
                        
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    logging.error(f"Error analyzing resume: {str(e)}", exc_info=True)
            else:
                st.warning("Please upload a resume first.")
        
        # Generate cover letter section
        if 'match_result' in st.session_state and st.session_state.match_result:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Generate Cover Letter")
            
            col_company, col_button = st.columns([3, 1])
            with col_company:
                company_name = st.text_input(
                    "Company Name",
                    value="Google",
                    help="Enter the company name for your cover letter"
                )
            
            with col_button:
                generate_button = st.button(
                    "Generate",
                    type="primary",
                    help="Generate a tailored cover letter based on your resume and the selected job"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            if generate_button:
                try:
                    with st.spinner("Generating your cover letter..."):
                        # Start timer for generation metrics
                        start_time = time.time()
                        
                        # Show a loading message with custom styling and progress
                        cover_letter_placeholder = st.empty()
                        progress_placeholder = st.empty()
                        
                        # Initial progress message
                        cover_letter_placeholder.markdown(
                            """
                            <div style="text-align: center; padding: 20px; border-radius: 8px; background-color: var(--card-bg); border: 1px solid var(--border);">
                                <div class="animate-pulse">
                                    <p style="margin-bottom: 10px; font-weight: 500; color: var(--text-primary);">Creating your personalized cover letter...</p>
                                    <p style="font-size: 0.9rem; color: var(--text-muted);">This typically takes 15-30 seconds</p>
                                </div>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                        
                        # Create a progress bar
                        progress_bar = progress_placeholder.progress(0)
                        
                        # Background task to update progress while cover letter is being generated
                        def update_progress():
                            for percent in range(0, 95, 5):
                                progress_bar.progress(percent)
                                time.sleep(0.5)  # Adjust timing for smoother progress
                        
                        # Start progress bar update in a separate thread
                        progress_thread = threading.Thread(target=update_progress)
                        progress_thread.start()
                        
                        # Get necessary information from session state
                        resume_text = st.session_state.resume_text
                        job_title = st.session_state.selected_job
                        job_description = st.session_state.job_description
                        match_result = st.session_state.match_result
                        
                        # Extract candidate name from parsed resume if available
                        candidate_name = ""
                        if "parsed_resume" in match_result and match_result["parsed_resume"]:
                            parsed_resume = match_result["parsed_resume"]
                            candidate_name = parsed_resume.get("name", "")
                            candidate_email = parsed_resume.get("email", "")
                        
                        # Generate cover letter
                        cover_letter = generate_cover_letter(
                            resume_text=resume_text,
                            job_title=job_title,
                            job_description=job_description,
                            company_name=company_name,
                            candidate_name=candidate_name,
                            matching_skills=match_result.get("matching_skills", []),
                            missing_skills=match_result.get("missing_skills", [])
                        )
                        
                        # Store cover letter in session state
                        st.session_state.cover_letter = cover_letter
                        
                        # Update the match result in the database with the cover letter
                        if st.session_state.match_id:
                            conn = sqlite3.connect('resumatch.db')
                            c = conn.cursor()
                            c.execute(
                                "UPDATE matches SET cover_letter = ? WHERE id = ?",
                                (cover_letter, st.session_state.match_id)
                            )
                            conn.commit()
                            conn.close()
                            logger.info(f"Updated match {st.session_state.match_id} with cover letter")
                        
                        # Complete the progress bar
                        progress_bar.progress(100)
                        time.sleep(0.5)
                        progress_placeholder.empty()
                        
                        # Calculate generation time
                        generation_time = time.time() - start_time
                        
                        # Replace loading message with success message
                        cover_letter_placeholder.success(f"✓ Cover letter generated in {generation_time:.1f} seconds!")
                        
                except Exception as e:
                    st.error(f"Failed to generate cover letter: {str(e)}")
                    logging.error(f"Cover letter generation error: {str(e)}", exc_info=True)
                    # Set a fallback cover letter message
                    st.session_state.cover_letter = "We encountered an issue generating your cover letter. Please try again later."
    
    with col2:
        # Display match results if available
        if 'match_result' in st.session_state and st.session_state.match_result:
            match_result = st.session_state.match_result
            
            st.markdown('<div class="card animate-fade-in">', unsafe_allow_html=True)
            st.subheader(f"Match Results for {st.session_state.selected_job}")
            
            # Display match score with appropriate color class
            match_score = match_result.get("match_percentage", 0)
            match_class = "low-match" if match_score < 50 else ("medium-match" if match_score < 75 else "high-match")
            
            st.markdown(
                f"""
                <div style="text-align: center; margin-bottom: 1.5rem;">
                    <span class="match-score {match_class}">{match_score}% Match</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Display matching skills
            st.markdown("<h3>Matching Skills</h3>", unsafe_allow_html=True)
            matching_skills = match_result.get("matching_skills", [])
            if matching_skills:
                skills_html = ""
                for skill in matching_skills:
                    skills_html += f"<span class='skill-tag matching-skill'>{skill}</span>"
                st.markdown(f"<div style='margin-bottom: 1.5rem;'>{skills_html}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<p>No matching skills found.</p>", unsafe_allow_html=True)
            
            # Display missing skills
            st.markdown("<h3>Missing Skills</h3>", unsafe_allow_html=True)
            missing_skills = match_result.get("missing_skills", [])
            if missing_skills:
                skills_html = ""
                for skill in missing_skills:
                    skills_html += f"<span class='skill-tag missing-skill'>{skill}</span>"
                st.markdown(f"<div style='margin-bottom: 1.5rem;'>{skills_html}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<p>No missing skills - Great job!</p>", unsafe_allow_html=True)
            
            # Display match summary
            st.markdown("<h3>Summary</h3>", unsafe_allow_html=True)
            summary = match_result.get('summary', 'No summary available.')
            st.markdown(f"<p style='margin-bottom: 1.5rem;'>{summary}</p>", unsafe_allow_html=True)
            
            # Display database IDs (for demonstration)
            if st.session_state.candidate_id and st.session_state.match_id:
                st.markdown(
                    f"""
                    <div style='font-size: 0.8rem; color: var(--text-muted); margin-top: 1rem; border-top: 1px solid var(--border); padding-top: 0.5rem;'>
                        <p>Candidate ID: {st.session_state.candidate_id}<br>Match ID: {st.session_state.match_id}</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Display cover letter if available
        if 'cover_letter' in st.session_state and st.session_state.cover_letter:
            st.markdown('<div class="card animate-fade-in">', unsafe_allow_html=True)
            st.subheader("Your Personalized Cover Letter")
            
            # Format the cover letter with proper line breaks for display
            formatted_cover_letter = st.session_state.cover_letter.replace("\n", "<br>")
            
            # Display the formatted cover letter
            st.markdown(
                f"<div class='cover-letter'>{formatted_cover_letter}</div>", 
                unsafe_allow_html=True
            )
            
            # Add options to download the cover letter
            cover_letter_text = st.session_state.cover_letter
            
            st.markdown("<div class='download-buttons'>", unsafe_allow_html=True)
            
            # Download as TXT option
            download_txt = st.download_button(
                label="Download as TXT",
                data=cover_letter_text,
                file_name=f"Cover_Letter_{st.session_state.selected_job.replace(' ', '_')}.txt",
                mime="text/plain",
                key="txt_download"
            )
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Add guidance for using the cover letter
            st.markdown(
                """
                <div style="margin-top: 1rem; font-size: 0.9rem; color: var(--text-muted); border-top: 1px solid var(--border); padding-top: 0.75rem;">
                    <p><strong>Tips for using your cover letter:</strong></p>
                    <ul style="margin-top: 0.5rem;">
                        <li>Review and personalize further if needed</li>
                        <li>Proofread before sending</li>
                        <li>Add specific achievements relevant to the position</li>
                        <li>Format according to your preferred style</li>
                    </ul>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Add a section to show database statistics
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Database Statistics")
        
        conn = sqlite3.connect('resumatch.db')
        c = conn.cursor()
        
        # Count candidates
        c.execute("SELECT COUNT(*) FROM candidates")
        candidate_count = c.fetchone()[0]
        
        # Count jobs
        c.execute("SELECT COUNT(*) FROM jobs")
        job_count = c.fetchone()[0]
        
        # Count matches
        c.execute("SELECT COUNT(*) FROM matches")
        match_count = c.fetchone()[0]
        
        conn.close()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Candidates", candidate_count)
        col2.metric("Jobs", job_count)
        col3.metric("Matches", match_count)
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()