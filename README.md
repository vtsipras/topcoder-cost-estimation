# Topcoder Estimator Pro 🤖📊

**Undergraduate Thesis Project**  
*Aristotle University of Thessaloniki (AUTH) - Department of Informatics*

## Overview
**Topcoder Estimator Pro** is a data-driven Decision Support System (DSS) designed to predict software development costs in crowdsourcing environments. Traditional estimation methods (e.g., COCOMO, FPA) require structured requirements or lines of code, making them unsuitable for the unstructured nature of crowdsourcing platforms. 

This project bridges the gap between **Natural Language Processing (NLP)** and **Statistical Machine Learning**. It utilizes a local Large Language Model (**Llama 3.2**) to semantically analyze raw project descriptions and extract quantitative complexity metrics. These features are then fed into a **Multiple Linear Regression** model to predict the optimal financial reward (Prize in USD) with high accuracy and absolute "White-Box" transparency.

## Key Features
* **Automated Data Extraction:** Custom Selenium Web Scraper built to bypass API limitations and extract historical challenge data directly from Topcoder.
* **Local NLP Inference:** Utilizes Llama 3.2 via Ollama for zero-shot prompt feature extraction, ensuring 100% data privacy and zero API costs.
* **Semantic Feature Engineering:** Automatically extracts 5 core metrics from raw text: *Complexity, Tech Count, QA Effort, Deliverables Count,* and *Duration (Days)*.
* **White-Box ML Model:** Implements Multiple Linear Regression trained on filtered, high-quality historical data ($50 - $2500 range) achieving an R² of 99.8%.
* **Interactive UI:** A modern, responsive Single Page Application built with Streamlit, providing real-time feedback and dynamic visualization of the cost breakdown.

## Repository Structure
```text
topcoder-cost-estimation/
│
├── app.py                         # Main Streamlit web application & ML execution logic
├── fetch_data.py                  # Selenium Web Scraper for Topcoder challenges
├── llm_extractor_local.py         # Local LLM communication script via Ollama
├── topcoder_dataset_FINAL.csv     # Cleaned historical dataset used for model training
├── topcoder_dataset.csv           # Raw scraped dataset (baseline)
└── README.md                      # Project documentation
```

## Tech Stack
* **Language:** Python 3.10+
* **Frontend/UI:** Streamlit
* **Machine Learning:** Scikit-Learn (Linear Regression), Pandas
* **NLP / LLM:** Ollama, Llama 3.2
* **Data Scraping:** Selenium, Webdriver Manager

## Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.10+ installed. You also need to install [Ollama](https://ollama.ai/) to run the local LLM.

### 2. Clone the Repository
```bash
git clone [https://github.com/your-username/topcoder-cost-estimation.git](https://github.com/your-username/topcoder-cost-estimation.git)
cd topcoder-cost-estimation
```

### 3. Setup Virtual Environment & Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install streamlit pandas scikit-learn requests selenium webdriver-manager
```

### 4. Install and Run Llama 3.2
Start the Ollama service and pull the Llama 3.2 model:
```bash
ollama run llama3.2
```
*(Ensure the Ollama daemon is running in the background on port 11434)*

### 5. Launch the Application
```bash
streamlit run app.py
```

## Usage Example
1. Open the Streamlit web interface (`http://localhost:8501`).
2. Paste a raw project description into the text area. Example:
   > *"We need a frontend developer to build a responsive landing page using Next.js and Tailwind CSS. The page should include a contact form with email validation and a dark mode toggle. Delivery is expected in 5 days. Must include basic unit tests and the source code."*
3. Click **"Εκτέλεση Κοστολόγησης AI"**.
4. The system will asynchronously query Llama 3.2, extract the metrics `[Complexity: 2, Tech_Count: 2, QA_Effort: 2, Deliverables: 1, Duration: 5]`, and instantly return the calculated budget (e.g., **$105.63**) along with a visual breakdown.

## Academic Context
This software was developed as part of an undergraduate thesis at the **Aristotle University of Thessaloniki (AUTH)**, under the supervision of Prof. Ioannis Stamelos. The primary objective is to demonstrate that modern Open-Source LLMs can effectively replace subjective human effort in requirement quantification, providing a mathematically sound and explainable foundation for Software Cost Estimation.

## License
Distributed under the MIT License.