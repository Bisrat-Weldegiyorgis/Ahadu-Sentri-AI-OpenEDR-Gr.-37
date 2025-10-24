Ahadu Sentri-AI OpenEDR automation system 

Ahadu SentriAI is an open-source cybersecurity project designed to provide intelligent Endpoint Detection and Response (EDR) capabilities. It leverages machine learning, rules-based policies, and automated actions to detect, analyze, and respond to security threats in real time.

Features
* Event Ingestion – Collects raw security events from endpoints.
* Feature Extraction – Transforms raw data into meaningful features for analysis.
* ML-Based Detection – Uses trained models to identify anomalies or malicious patterns.
* Policy Engine – Applies security rules for decision-making.
* Automated Response – Supports actions such as notify, quarantine, or isolate endpoints.
* Modular Design – Easy to extend with new features, models, or integrations.

Requirements

* Python 3.9+
* FastAPI
* scikit-learn (or ML framework)
* Uvicorn

Installation process 
1. **Clone the repository**
     ```bash
     git clone https://github.com/Bisrat-Weldegiyorgis/Ahadu-Sentri-AI-OpenEDR-Gr.-37.git
     ```
     One the file exists use this directory
     ```bash
     cd Ahadu-Sentri-AI-OpenEDR-Gr.-37
     ```
2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the FastAPI application:**
   ```bash
   uvicorn app:app
    ```

6. **Access the API:**
   Open your browser and go to `http://127.0.0.1:8000/docs` to view the interactive API documentation.


   ## Usage

### Making Predictions

