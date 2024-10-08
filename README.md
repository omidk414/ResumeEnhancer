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
12. [Collaborators](#11-collaborators)
13. [Acknowledgements](#12-acknowledgements)

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
The first step of the project involved scraping job postings from [Simply Hired](https://www.simplyhired.com/) using **Selenium** for dynamic page interactions. Selenium was chosen due to its ability to handle JavaScript-rendered content and interact with page elements. We targeted job postings specifically filtered for positions located in California and included six job roles:

- **Data Engineer** jobs in California.
- **Software Engineer** jobs in California.
- **Data Scientist** jobs in California.
- **Data Analyst** jobs in California.
- **Business Systems Analyst** jobs in California.
- **Software Developer** jobs in California.

### Key Libraries Used:
- **Selenium** for HTML parsing.
- **pandas** for organizing the data into a structured format.

### CSV File Headers:
The scraped data was structured into a CSV file with the following headers:

- **Job Title**: The title of the job position (e.g., Data Scientist, Software Engineer).
- **Company Name**: The name of the company offering the job.
- **Job Location**: The location of the job.
- **Estimated Salary**: The estimated salary range for the position.
- **Job Details**: The detailed job description, including responsibilities and required qualifications.
- **Job Href**: The hyperlink to the job posting on Simply Hired.

### Focused Columns for Model Training:
For our machine learning model, we decided to use only the `Job Title` and `Job Details` columns. The other fields, such as `Company Name`, `Job Location`, `Estimated Salary`, and `Job Href`, were excluded because they were not essential for the purpose of resume enhancement. The goal was to use the job title and detailed description to enhance resumes to meet current job needs, making them more relevant for specific roles in the job market.

### Challenges:
- Implementing compliance with the site’s terms of service and ensuring ethical data collection.
- Handling dynamic content loading with Selenium and avoiding anti-bot detection mechanisms.

### Output:
- A CSV file containing the focused job data, with the following columns: `Job_Title` and `Job_Details`.

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

### Example Code:
~~~python
import gradio as gr

# Load the trained model
model = tf.keras.models.load_model('resume_generator_model.h5')

def enhance_resume(resume_text, job_title):
    # Process the resume text (e.g., using the same TF-IDF vectorization)
    resume_vector = vectorizer.transform([resume_text]).toarray()
    
    # Predict the job title based on the resume content
    predicted_class = model.predict(resume_vector)
    predicted_title = label_encoder.inverse_transform([predicted_class.argmax()])

    return predicted_title[0]

# Create a Gradio interface for the resume enhancement
iface = gr.Interface(fn=enhance_resume, inputs=["text", "text"], outputs="text", title="Resume Enhancement Tool")
iface.launch()
~~~

## 5. Data Analysis and Visualization
### Analysis Techniques:
- **K-Means Clustering**: Grouped job postings into 5 clusters based on skills and job descriptions.
- **PCA (Principal Component Analysis)**: Reduced dimensionality for visualization.
- **Elbow Curve Analysis**: Determined the optimal number of clusters.
- **Silhouette Scores**: Evaluated cluster compactness and separation.

### Example Code:
~~~python
# Perform K-Means clustering on the TF-IDF vectorized data
k_values = range(2, 10)
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Plot the silhouette scores for each value of k
plt.plot(k_values, silhouette_scores)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for K-Means Clustering')
plt.show()
~~~

## 6. Integrating Hugging Face Models
- We integrated **Hugging Face’s Meta-Llama 3.1** with our Keras-based model for deeper NLP and contextual understanding.
- This allowed us to leverage both the traditional deep learning approach and the power of state-of-the-art transformer models.

## 7. Results and Insights
The integration of multiple models allowed for a comprehensive system that dynamically enhances resumes based on job requirements. Key insights include:

- Cluster analysis showed that Data Science and Engineering roles have significant overlap in skill requirements.
- Resume enhancement showed measurable improvements in matching the job descriptions’ requirements.

## 8. How to Run the Project
1. Clone the repository.
2. Install the necessary dependencies from `requirements.txt`.
3. Run the main application using `python app.py`.
4. Upload a resume and specify a job title to see the enhanced resume.

## 9. Future Work
Future improvements include incorporating real-time labor market trends and expanding the enhancement system to cover a broader range of industries and job titles.

## 10. Conclusion
This project successfully demonstrates the capabilities of machine learning and NLP in enhancing resumes. By utilizing modern scraping techniques, advanced model training, and integration with powerful transformer models, we have developed a tool that can help job seekers better align their qualifications with market needs.

## 11. Collaborators
Evan Wall
Thay Chansy
Omid Khan

## 12. Acknowledgements
- OpenAI for their powerful models and libraries.
- The developers of the `Hugging Face Transformers` and `Selenium` libraries.
- Job seekers who provided valuable feedback during the development process.
