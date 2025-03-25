# ResuMatch: AI Resume & Job Matcher

ResuMatch is an AI-powered application that matches resumes to job descriptions and generates tailored cover letters. The application uses large language models (LLMs) to parse resumes, match them against job descriptions, and generate personalized cover letters.

## Features

- **Resume Parsing**: Upload PDF or DOCX resumes and extract structured data
- **Job Matching**: Match candidate skills against job requirements
- **Skills Analysis**: View matching and missing skills with visual indicators
- **Advanced Cover Letter Generation**: Create professional, tailored cover letters using sophisticated NLP techniques
- **Data Persistence**: Store resume data, job information, and matches in a database

## Requirements

- Python 3.8 or higher
- Streamlit
- PyPDF2
- docx2txt
- NLTK
- scikit-learn
- Groq API key (or other LLM API)

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/resumatch.git
cd resumatch
```

2. Create a virtual environment and activate it:
```
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the required packages:
```
pip install -r requirements.txt
```

4. Set up your Groq API key:
   - Option 1: Create a `.streamlit/secrets.toml` file with:
     ```
     GROQ_API_KEY = "your-key-here"
     ```
   - Option 2: Set an environment variable:
     ```
     export GROQ_API_KEY="your-key-here"  # On Windows: set GROQ_API_KEY=your-key-here
     ```

## Running the Application

To run the application:

```
streamlit run app.py
```

The application will be available at `http://localhost:8501`.

## How It Works

1. **Upload Resume**: Users upload their resume in PDF or DOCX format.
2. **Job Selection**: Users select a job position they want to apply for.
3. **Resume Analysis**: The system extracts text from the resume and uses an LLM to parse it into structured data.
4. **Job Matching**: The system matches the resume against the job description to calculate a match score.
5. **Cover Letter Generation**: Users can generate a tailored cover letter based on their resume and the job.

## Advanced Cover Letter Generation

ResuMatch uses a sophisticated approach to creating personalized cover letters:

- **Professional Tone**: Warm, professional language that avoids generic phrasing
- **Personalized Content**: Tailored to the specific role, company, and candidate skills
- **Structured Format**: Engaging opening, skills alignment, company connection, and strong closing
- **Robust Fallbacks**: Multiple fallback mechanisms ensure cover letter generation even without API access
- **Downloadable Output**: Easy download options for further customization

The cover letter generation process:
1. Analyzes the candidate's resume and extracted skills
2. Identifies matching skills and areas for improvement 
3. Incorporates job requirements and company information
4. Generates a professionally formatted letter with proper structure
5. Provides tips for further customization

## Multi-Step Function Calling

The application uses several LLM function calls:

1. `parse_resume_with_llm`: Parses the resume into structured data (name, skills, education, experience)
2. `extract_skills_from_text`: Extracts skills from text (used for both resumes and job descriptions)
3. `match_resume_to_job`: Matches resumes to jobs and calculates a match percentage
4. `generate_cover_letter`: Generates a tailored cover letter based on the match results

## Data Persistence

All data is stored in a SQLite database (`resumatch.db`) with the following tables:

- **candidates**: Stores parsed resume data
- **jobs**: Stores job information
- **matches**: Stores match results and cover letters

## Project Structure

- `app.py`: Main application file with Streamlit UI and logic
- `requirements.txt`: List of required packages
- `.streamlit/`: Configuration and secrets
- `resumatch.db`: SQLite database (created on first run)

## Future Improvements

- User authentication
- More detailed resume parsing
- Multiple resume uploads
- Job search functionality
- Saving and comparing multiple match results
- Email and sharing functionality
- Dockerization for easier deployment 