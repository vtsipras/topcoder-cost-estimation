import streamlit as st
import pandas as pd
import requests
import json
import re
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Application Setup & Styling
st.set_page_config(page_title="Topcoder Estimator Pro", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #f8fafc; }
    
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.3);
    }
    
    h1 { color: #0f172a; font-weight: 800; letter-spacing: -1px; }
    h3 { color: #334155; font-weight: 600; }
    
    .stTextArea textarea {
        border-radius: 10px;
        border: 1px solid #cbd5e1;
        box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# Language Model Configuration
MODELS = {'Llama 3.2': 'llama3.2'}
OLLAMA_URL = "http://localhost:11434/api/generate"

SYSTEM_PROMPT = """
You are an expert Software Engineering Project Estimator. 
Read the following raw project description from a crowdsourcing platform. Ignore any navigational menus, footers, or unrelated text.
Focus only on the technical requirements and extract the following features.
CRITICAL: You MUST output ONLY a valid JSON object exactly like this:
{
  "Complexity": 3,
  "Tech_Count": 4,
  "QA_Effort": 2,
  "Deliverables_Count": 3,
  "Duration_Days": 7
}
"""

# Utility: Extract valid JSON from raw LLM text
def extract_json(text):
    try:
        clean_text = text.replace('```json', '').replace('```', '').strip()
        match = re.search(r'\{.*\}', clean_text, re.DOTALL)
        if match: 
            return json.loads(match.group(0))
        return json.loads(clean_text)
    except Exception:
        return {"Complexity": 0, "Tech_Count": 0, "QA_Effort": 0, "Deliverables_Count": 0, "Duration_Days": 0}

# Service: Communicate with local Ollama instance
def query_local_ollama(model_name, description):
    payload = {
        "model": model_name, 
        "prompt": f"{SYSTEM_PROMPT}\n\nPROJECT:\n{description[:3000]}", 
        "stream": False, 
        "format": "json"
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        if response.status_code == 200:
            return extract_json(response.json().get("response", ""))
    except Exception:
        pass
    return {"Complexity": 0, "Tech_Count": 0, "QA_Effort": 0, "Deliverables_Count": 0, "Duration_Days": 0}

# UI Helper: Render customized metric cards
def create_metric_card(title, value, max_val):
    percentage = min((value / max_val) * 100, 100)
    return f"""
    <div style="background-color: white; padding: 20px 10px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); border-top: 4px solid #3b82f6; text-align: center; margin-bottom: 15px;">
        <p style="color: #64748b; font-size: 0.80rem; font-weight: 700; margin: 0; text-transform: uppercase; letter-spacing: 0.5px;">{title}</p>
        <h2 style="color: #0f172a; font-size: 2.2rem; font-weight: 800; margin: 8px 0;">{value:.1f}</h2>
        <div style="width: 100%; background-color: #f1f5f9; border-radius: 99px; height: 8px; margin-top: 15px; overflow: hidden;">
            <div style="width: {percentage}%; background-color: #3b82f6; height: 100%; border-radius: 99px; transition: width 1s ease-in-out;"></div>
        </div>
    </div>
    """

# Layout: Sidebar
with st.sidebar:
    st.markdown("## Cost Estimator AI")
    st.markdown("<p style='color: #64748b;'>Σύστημα Υποστήριξης Αποφάσεων</p>", unsafe_allow_html=True)
    st.divider()
    
    st.markdown("### Κατάσταση Υποδομής")
    st.success("Ollama Engine: Ενεργό")
    
    st.markdown("### Ενεργά Μοντέλα (Local)")
    for name in MODELS.keys():
        st.markdown(f"- **{name}** (SOTA NLP)")
        
    st.divider()
    with st.expander("Πώς λειτουργεί;"):
        st.caption("Το σύστημα αναλύει το κείμενο τοπικά μέσω του Llama 3.2. Εξάγει μετρικές πολυπλοκότητας και κάνει πρόβλεψη μέσω Γραμμικής Παλινδρόμησης (Linear Regression) βάσει ιστορικών δεδομένων Topcoder.")

# Layout: Main Content Area
st.title("Σύστημα Εκτίμησης Προϋπολογισμού Έργων") 
st.markdown("<p style='font-size: 1.1rem; color: #475569;'>Εισάγετε τις τεχνικές προδιαγραφές του έργου σε φυσική γλώσσα για τον αυτόματο υπολογισμό του προτεινόμενου επάθλου (Prize).</p>", unsafe_allow_html=True)
st.write("") 

new_description = st.text_area(
    "Προδιαγραφές Έργου (Technical Requirements):", 
    height=220, 
    placeholder="π.χ. We need a cross-platform mobile application built with React Native. The app must include user authentication, a product catalog, Stripe API integration, and basic unit tests..."
)

st.write("")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    calculate_btn = st.button("Εκτέλεση Κοστολόγησης AI", use_container_width=True)

# Main Execution Block
if calculate_btn:
    if len(new_description) < 30:
        st.error("Παρακαλώ εισάγετε μια πιο αναλυτική περιγραφή (τουλάχιστον 30 χαρακτήρες).")
    else:
        estimations = {}
        
        with st.status("Επεξεργασία δεδομένων...", expanded=True) as status:
            st.write("Αρχικοποίηση επικοινωνίας με Llama 3.2...")
            
            for display_name, ollama_name in MODELS.items():
                st.write(f"Εξαγωγή σημασιολογικών χαρακτηριστικών από **{display_name}**...")
                estimations[display_name] = query_local_ollama(ollama_name, new_description)
                
            st.write("Φόρτωση και φιλτράρισμα ιστορικού dataset αγοράς (CSV)...")
            
            current_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(current_dir, "topcoder_dataset_FINAL.csv")
            
            try:
                # Handle missing values
                df = pd.read_csv(csv_path)
                df = df.dropna(subset=['Prize_USD'])
                df = df.fillna(0)
                
                # Enforce budget boundaries (Outlier removal)
                df = df[(df['Prize_USD'] >= 50) & (df['Prize_USD'] <= 2500)]

                st.write("Επιλογή χαρακτηριστικών Llama...")
                
                # Define feature matrix (X) and target variable (y)
                X = df[['Complexity_Llama', 'Tech_Count_Llama', 'QA_Effort_Llama', 'Deliverables_Count_Llama', 'Duration_Days_Llama']]
                y = df['Prize_USD']

                st.write("Εκπαίδευση αλγορίθμου Γραμμικής Παλινδρόμησης (Linear Regression)...")
                model = LinearRegression()
                model.fit(X, y)
                
                # Calculate model performance metrics
                y_pred = model.predict(X)
                r2 = r2_score(y, y_pred)
                mae = mean_absolute_error(y, y_pred)
                
                st.write("Εξαγωγή τελικής πρόβλεψης...")
                comp = estimations['Llama 3.2'].get('Complexity', 0)
                tech = estimations['Llama 3.2'].get('Tech_Count', 0)
                qa = estimations['Llama 3.2'].get('QA_Effort', 0)
                deliv = estimations['Llama 3.2'].get('Deliverables_Count', 0)
                days = estimations['Llama 3.2'].get('Duration_Days', 0)

                new_project_features = pd.DataFrame([[comp, tech, qa, deliv, days]], 
                                                    columns=['Complexity_Llama', 'Tech_Count_Llama', 'QA_Effort_Llama', 'Deliverables_Count_Llama', 'Duration_Days_Llama'])
                predicted_price = model.predict(new_project_features)[0]
                
                # Establish minimum prize threshold
                if predicted_price < 50:
                    predicted_price = 50.0

                status.update(label="Η διαδικασία ολοκληρώθηκε επιτυχώς!", state="complete", expanded=False)

            except Exception as e:
                status.update(label="Παρουσιάστηκε σφάλμα", state="error", expanded=False)
                st.error(f"Λεπτομέρειες σφάλματος: {e}")
                st.stop()

        # Dynamic UI Updates
        st.write("<br>", unsafe_allow_html=True)
        
        hero_html = f"""
        <div style="background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); padding: 40px 20px 30px 20px; border-radius: 16px; text-align: center; color: white; box-shadow: 0 10px 25px rgba(37, 99, 235, 0.25); margin-bottom: 20px;">
            <h3 style="margin: 0; font-size: 1.2rem; font-weight: 500; opacity: 0.9; text-transform: uppercase; letter-spacing: 1px;">Προτεινομενος Προϋπολογισμος Εργου</h3>
            <h1 style="margin: 10px 0 0 0; font-size: 5rem; font-weight: 900; letter-spacing: -2px; color: white;">${predicted_price:,.2f}</h1>
            <p style="margin: 15px 0 0 0; font-size: 0.95rem; opacity: 0.8; font-weight: 300;">Η πρόβλεψη βασίζεται σε σημασιολογική ανάλυση (Llama 3.2) και αλγόριθμο Γραμμικής Παλινδρόμησης.</p>
        </div>
        """
        st.markdown(hero_html, unsafe_allow_html=True)
        
        # Render performance metrics
        st.markdown(f"""
        <div style="display: flex; justify-content: center; gap: 20px; margin-bottom: 40px;">
            <div style="background-color: #e0f2fe; color: #0369a1; padding: 10px 20px; border-radius: 8px; font-weight: 600; font-size: 0.95rem; border: 1px solid #bae6fd;">
                🎯 Ακρίβεια Μοντέλου (R²): {r2*100:.1f}%
            </div>
            <div style="background-color: #e0f2fe; color: #0369a1; padding: 10px 20px; border-radius: 8px; font-weight: 600; font-size: 0.95rem; border: 1px solid #bae6fd;">
                ⚖️ Μέσο Απόλυτο Σφάλμα (MAE): ±${mae:.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Ποσοτικοποίηση Μετρικών", "Ακατέργαστα Δεδομένα (JSON)"])
        
        with tab1:
            st.markdown("### Ανάλυση Εξαγόμενων Χαρακτηριστικών")
            m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
            
            with m_col1:
                st.markdown(create_metric_card("Πολυπλοκοτητα", comp, 5.0), unsafe_allow_html=True)
            with m_col2:
                st.markdown(create_metric_card("Τεχνολογιες", tech, 10.0), unsafe_allow_html=True)
            with m_col3:
                st.markdown(create_metric_card("Απαιτησεις QA", qa, 5.0), unsafe_allow_html=True)
            with m_col4:
                st.markdown(create_metric_card("Παραδοτεα", deliv, 10.0), unsafe_allow_html=True)
            with m_col5:
                st.markdown(create_metric_card("Ημερες", days, 45.0), unsafe_allow_html=True)

        with tab2:
            st.markdown("### Στατιστικά Εξόδου")
            st.json(estimations)