# Resume Enhancement and Job Market Analysis Project

## Table of Contents
1. [Overview](#overview)
2. [Web Scraping from Simply Hired](#1-web-scraping-from-simply-hired)
3. [Data Normalization and Preprocessing](#2-data-normalization-and-preprocessing)
4. [Model Training](#3-model-training)
5. [Resume Enhancement](#4-resume-enhancement)
6. [Data Analysis and Visualization](#5-data-analysis-and-visualization)
7. [Integrating Hugging Face Models](#6-integrating-hugging-face-models)
8. [Results and Insights](#7-results-and-insights)
9. [How to Run the Project](#8-how-to-run-the-project)
10. [Future Work](#9-future-work)
11. [Conclusion](#10-conclusion)

## Overview
This project aims to develop a machine learning model and a comprehensive analysis system for enhancing resumes based on job titles and job market trends. The project workflow includes web scraping job data, data normalization, clustering analysis, resume enhancement using advanced machine learning models, and visualization of results. The ultimate goal is to assist candidates by tailoring their resumes to align with market demands.

### Key Components:
- **Web Scraping and Data Collection**
- **Data Normalization and Preprocessing**
- **Model Training and Resume Enhancement**
- **Data Analysis and Visualization**
- **Integration with Hugging Face Models**
- **Cluster Analysis and Insights**

## 1. Web Scraping from Simply Hired
The first step of the project involved scraping job postings from [Simply Hired](https://www.simplyhired.com/) using a custom Python script. The job postings were categorized by job titles, job descriptions, and specific skill requirements. 

### Key Libraries:
- **BeautifulSoup** and **requests** for web scraping.
- **pandas** for organizing the data into a structured format.

### Challenges:
- Implementing compliance with the site’s terms of service and ensuring ethical data collection.
- Parsing complex HTML structures to extract relevant information like job descriptions, requirements, and company-specific details.

### Output:
- A CSV file containing the scraped job data, with the following columns: `Job_Title`, `Job_Details`, `Skills`, `Company`, and `Estimated_Salary`.

## 2. Data Normalization and Preprocessing
After gathering the raw job data, the next step was to clean and normalize it for further analysis. This included removing HTML tags, special characters, and redundant information.

### Key Steps:
1. **Tokenization and Text Cleaning**: 
   - Used `nltk` and `spaCy` for initial text tokenization and entity extraction.
   - Removed stop words, punctuation, and common phrases.
   
2. **Skill Extraction**:
   - Extracted skill keywords from the job descriptions.
   - Mapped each job title to a set of core and secondary skills.

3. **Text Embedding and Vectorization**:
   - Implemented BERT-based embeddings using **Hugging Face's Transformers** library.
   - Preprocessed text using the `meta-llama/Meta-Llama-3.1-8B-Instruct` model for further NLP tasks.

### Challenges:
- **NLTK and spaCy Integration Issues**: 
   - Initial attempts to combine `nltk` and `spaCy` with Hugging Face models faced incompatibility issues due to different tokenization schemes.
   - Resolved by switching entirely to Hugging Face's tokenizers for consistent embeddings.

## 3. Model Training
### Machine Learning Models Used:
1. **Custom TensorFlow Model**:
   - We created a custom deep learning model using **Keras** and trained it on the cleaned job descriptions and skills.
   - The model (`resume_generator_model.h5`) was trained over 100 epochs and achieved an accuracy of 72.85%.
   
2. **Hugging Face’s `Meta-Llama-3.1-8B-Instruct` Model**:
   - Fine-tuned this transformer model to handle contextual queries for enhancing resumes.
   - The model was accessed using a Pro API key and integrated into our pipeline using the **InferenceClient** from the `transformers` library.

### Approach:
- **Data Input**: Combined data from `Job_Title` and `Job_Details`.
- **Model Architecture**: A multi-layer perceptron (MLP) model with three dense layers.
- **Loss Function**: Categorical Cross-Entropy.
- **Optimizer**: Adam.

### Challenges:
- Integration of the `h5` Keras model with Llama required custom preprocessing layers.
- Handling large datasets efficiently within a limited-memory environment.

## 4. Resume Enhancement
The core feature of this project was to create a system that takes an existing resume and optimizes it based on a specified job title. The enhancement process is as follows:

### Step-by-Step Workflow:
1. **Input**: Upload a resume (PDF or text) and specify the desired job title.
2. **Text Extraction**:
   - Used **PyMuPDF (fitz)** to extract text from PDF files.
   - Applied BERT-based embeddings to parse the text.
   
3. **Keyword Mapping**:
   - Used the `resume_generator_model.h5` to identify gaps in the resume based on the job title’s requirements.
   - Applied additional enhancements using `Meta-Llama-3.1-8B-Instruct` to generate text aligned with the target job title.

4. **Output**: 
   - A dynamically enhanced resume, formatted using a pre-defined HTML template located in the `template` folder.

### Challenges:
- Ensuring the two models (Keras and Hugging Face) work together in sequence.
- Avoiding generation of filler content and maintaining relevancy to the job title.

## 5. Data Analysis and Visualization
After enhancing resumes, we performed an in-depth analysis of job postings using clustering and visualization techniques. This process helps identify trends and commonalities across different job markets.

### Techniques Used:
- **K-Means Clustering**: Grouped job postings into 5 clusters based on skills and job descriptions.
- **PCA (Principal Component Analysis)**: Reduced dimensionality for visualization.
- **Elbow Curve Analysis**: Determined the optimal number of clusters.
- **Silhouette Scores**: Evaluated cluster compactness and separation.

### Visualizations:
1. **Bar Charts**: Showed skill distributions for each cluster.
2. **Word Clouds**: Highlighted the most frequent keywords in the job descriptions.
3. **Cluster Plot**: Displayed job postings in a 2D space based on cluster membership.

### Output Analysis:
- Certain skills (e.g., software development, project management) were universally desired.
- Cluster analysis revealed distinct job segments (e.g., technical roles vs. managerial roles).

## 6. Integrating Hugging Face Models
The project heavily relies on Hugging Face’s transformer models for various NLP tasks. Key integration points include:
- **Text Embeddings**: Used BERT embeddings to create feature vectors.
- **Contextual Queries**: Employed `Meta-Llama-3.1-8B-Instruct` to generate and enhance content dynamically.
- **Inference Optimization**: Integrated using the `InferenceClient` for efficient processing of enhancement queries.

### Challenges with NLP Libraries:
- Initial attempts with **NLTK** and **spaCy** were abandoned due to incompatibility with Hugging Face’s transformers.
- Switching to Hugging Face’s native tokenizers provided a consistent pipeline.

## 7. Results and Insights
### Key Findings:
1. **Enhanced Resumes**: Resumes were successfully enhanced with keywords and skills that matched the target job titles.
2. **Cluster Analysis**: Revealed distinct job market segments.
3. **Optimal Models**: Combining the `resume_generator_model.h5` and `Meta-Llama-3.1-8B-Instruct` allowed for accurate and context-aware resume generation.

### Final Output:
- A comprehensive resume enhancement system that dynamically generates resumes tailored to the user’s specified job title.

## 8. How to Run the Project
### Prerequisites:
- Python 3.8+
- Hugging Face API Key (Pro access)
- TensorFlow, Keras, pandas, and sklearn libraries

### Steps to Run:
1. Clone the repository.
2. Set up a virtual environment and install the required libraries.
3. Download the pre-trained `resume_generator_model.h5` and place it in the project root.
4. Set up the Hugging Face API key in a `.env` file.
5. Run `app.py` to start the application.

### Usage:
- Upload a resume and specify a job title.
- View the dynamically enhanced resume in the output.

## 9. Future Work
- Expanding the dataset to include more job titles and industries.
- Enhancing the resume with company-specific content using LLMs.
- Implementing a feedback loop to refine resume generation based on user input.

## 10. Conclusion
This project successfully integrates traditional machine learning with state-of-the-art NLP models to build a robust system for resume enhancement and job market analysis. By leveraging clustering techniques and transformer models, we have created a tool that not only tailors resumes but also provides valuable insights into the job market landscape.
