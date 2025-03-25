# Contributing to ResuMatch AI

Thank you for your interest in contributing to ResuMatch AI! This document provides guidelines for contributions to improve this project.

## Setting Up Development Environment

1. **Clone the Repository**
   ```bash
   git clone https://github.com/idixitjain/ResuMatch-AI.git
   cd ResuMatch-AI
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up API Keys**
   - Create a `.streamlit/secrets.toml` file
   - Add your Groq API key:
     ```toml
     GROQ_API_KEY = "your-api-key-here"
     ```

## Running the Application Locally

```bash
streamlit run app.py
```

## Project Structure

- **app.py**: Main application file with the Streamlit interface
- **requirements.txt**: List of Python dependencies
- **README.md**: Project documentation
- **.streamlit/secrets.toml**: Configuration for API keys (not tracked in git)

## Development Guidelines

When contributing to this project, please follow these guidelines:

### Code Style

- Follow PEP 8 style guidelines for Python code
- Use descriptive variable names
- Add comments for complex operations

### Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Feature Suggestions

If you have ideas for new features, consider:

- Resume parsing improvements
- Enhanced matching algorithms
- Additional export options for results
- User authentication
- Job search integration
- Advanced analytics

## Testing

Before submitting changes, please test thoroughly:

- Upload different resume formats (PDF, DOCX)
- Test with various job descriptions
- Verify cover letter generation
- Check database operations

## Deployment Options

### Streamlit Cloud

1. Visit [Streamlit Cloud](https://streamlit.io/cloud)
2. Connect your GitHub account
3. Select your forked repository
4. Set up your secret values (Groq API key)
5. Deploy the application

### Local Deployment

For production deployments, consider:
- Docker containerization
- Proper environment variable management
- Database backup strategies

## Questions?

Feel free to open an issue if you have questions or suggestions about contributing.

Thank you for helping improve ResuMatch AI! 