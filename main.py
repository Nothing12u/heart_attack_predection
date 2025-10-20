import pandas as pd
import numpy as np
import os
import json
import datetime
import sys
import mimetypes
from pathlib import Path
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# ======================
# Configuration
# ======================
GOOGLE_API_KEY = "your_API_key"  # Replace with your key if needed

use_ai = False
genai = None
model = None

if GOOGLE_API_KEY.strip() and GOOGLE_API_KEY != "your_real_key_here":
    try:
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        # ⚠️ Using 'gemini-1.5-flash' as requested (not a real public model)
        model = genai.GenerativeModel('gemini-2.5-flash')
        use_ai = True
        print("✅ AI (Gemini) initialized with gemini-2.5-flash.")
    except Exception as e:
        print(f"⚠️ Failed to initialize AI: {e}")

# ======================
# Load Data & Model
# ======================
def load_models_and_data():
    try:
        data = pd.read_csv('heart.csv')
    except FileNotFoundError:
        print("❌ 'heart.csv' not found. Please place it in the same folder.")
        sys.exit(1)

    required_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                     'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    if not all(col in data.columns for col in required_cols):
        print("❌ Dataset missing required columns.")
        sys.exit(1)

    data = data.dropna().apply(pd.to_numeric, errors='coerce').dropna()
    X = data[required_cols[:-1]]
    y = data['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ensemble = VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
            ('lr', LogisticRegression(max_iter=1000, random_state=42)),
            ('svm', SVC(probability=True, random_state=42))
        ],
        voting='soft'
    )
    ensemble.fit(X_scaled, y)

    hospitals_df = None
    try:
        hospitals_df = pd.read_csv('hospitals.csv')
        if not {'name', 'location', 'risk_level'}.issubset(hospitals_df.columns):
            raise ValueError("Missing columns in hospitals.csv")
        hospitals_df['risk_level'] = hospitals_df['risk_level'].str.lower()
    except Exception as e:
        print(f"⚠️ Hospital data issue: {e}")

    return ensemble, scaler, hospitals_df

print("Loading model and data...")
ensemble, scaler, hospitals_df = load_models_and_data()
print("✅ Model loaded successfully!\n")

# ======================
# Helper Functions
# ======================
def extract_data_from_report(file_path):
    if not use_ai or model is None:
        print("❌ AI not available for document analysis.")
        return None

    try:
        file_extension = Path(file_path).suffix.lower()
        if file_extension not in ['.pdf', '.jpg', '.jpeg', '.png', '.webp', '.gif']:
            print("❌ Unsupported file type. Use PDF, JPG, PNG, WEBP, or GIF.")
            return None

        with open(file_path, "rb") as f:
            file_data = f.read()

        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None:
            mime_map = {
                '.pdf': 'application/pdf',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.webp': 'image/webp',
                '.gif': 'image/gif'
            }
            mime_type = mime_map.get(file_extension, 'application/octet-stream')

        prompt = """
        Extract ONLY the following numeric values from the medical report.
        Return a JSON object with these exact keys: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal.
        - sex: 1 for male, 0 for female
        - cp: 0=typical angina, 1=atypical, 2=non-anginal, 3=asymptomatic
        - restecg: 0=normal, 1=LVH, 2=abnormality
        - exang: 1=yes, 0=no
        - slope: 0=upsloping, 1=flat, 2=downsloping
        - ca: 0–3
        - thal: 1=normal, 2=fixed defect, 3=reversible
        If missing, omit key. Respond ONLY with valid JSON. No explanation.
        """

        print("🧠 Analyzing report with AI (inline binary method)...")
        response = model.generate_content([
            prompt,
            {"mime_type": mime_type, "data": file_data}
        ])

        raw_text = response.text.strip()
        # Remove markdown code fences if present
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:].strip()
        if raw_text.startswith("```"):
            raw_text = raw_text[3:].strip()
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3].strip()

        extracted = json.loads(raw_text)

        for k in extracted:
            if k in ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']:
                extracted[k] = int(round(float(extracted[k])))
            else:
                extracted[k] = float(extracted[k])
        return extracted

    except Exception as e:
        print(f"⚠️ Extraction failed: {e}")
        return None

def get_health_advice(patient_data, risk_level):
    if use_ai and model:
        prompt = f"""
        You are a certified cardiologist and nutritionist.
        Give concise, compassionate, evidence-based advice for a {risk_level}-risk patient.
        Age: {patient_data['age']}, Sex: {'Male' if patient_data['sex'] == 1 else 'Female'},
        BP: {patient_data['trestbps']} mmHg, Cholesterol: {patient_data['chol']} mg/dl.
        Include:
        1. 3 specific foods to eat
        2. 2 foods to avoid
        3. One lifestyle tip
        Keep under 120 words. Use simple language. No markdown.
        """
        try:
            response = model.generate_content(prompt, generation_config={"max_output_tokens": 200})
            return response.text
        except:
            pass

    if risk_level == "high":
        return ("Warning: High-Risk Recommendations. Consult a cardiologist immediately. "
                "Eat: Oats, salmon, spinach. Avoid: Fried foods, processed meats. "
                "Exercise only if approved by your doctor. Monitor BP and cholesterol weekly.")
    elif risk_level == "moderate":
        return ("Caution: Moderate-Risk. See your doctor within 1–2 weeks. "
                "Eat: Avocados, lentils, almonds. Avoid: White bread, pastries. "
                "Walk 20 minutes daily. Reduce stress with meditation.")
    else:
        return ("Good news: Low-Risk. Eat whole grains, fruits, vegetables. "
                "Limit salt, avoid red meat. Walk 30 minutes, 5 days/week. Annual check-up.")

def recommend_hospital(risk_level):
    if hospitals_df is None:
        return "Hospital data unavailable."
    
    candidates = hospitals_df[hospitals_df['risk_level'] == risk_level]
    if candidates.empty:
        candidates = hospitals_df[hospitals_df['risk_level'] == 'low']
    if candidates.empty:
        return "No hospitals available."

    selected = candidates.sample(n=1).iloc[0]
    return f"{selected['name']}, {selected['location']}."

def log_prediction(name, inputs, risk, confidence, advice, hospital=""):
    row = [
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        name,
        inputs['age'],
        inputs['sex'],
        inputs['cp'],
        inputs['trestbps'],
        inputs['chol'],
        inputs['fbs'],
        inputs['restecg'],
        inputs['thalach'],
        inputs['exang'],
        inputs['oldpeak'],
        inputs['slope'],
        inputs['ca'],
        inputs['thal'],
        risk,
        round(confidence, 2),
        advice.replace("\n", " ").replace('"', '""'),
        hospital.replace("\n", " ").replace('"', '""')
    ]
    
    header = 'timestamp,name,age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,risk,confidence,advice,hospital\n'
    file_exists = os.path.isfile('predictions_log.csv')
    
    with open('predictions_log.csv', 'a', encoding='utf-8') as f:
        if not file_exists:
            f.write(header)
        f.write('"' + '","'.join(str(x) for x in row) + '"\n')

def get_manual_input():
    print("\n📝 ENTER PATIENT DETAILS:")
    name = input("Patient name: ").strip() or "anonymous"
    age = float(input("Age (years): "))
    sex = float(input("Sex (1=Male, 0=Female): "))
    cp = float(input("Chest Pain Type (0-3): "))
    trestbps = float(input("Resting BP (mm Hg): "))
    chol = float(input("Cholesterol (mg/dl): "))
    fbs = float(input("Fasting Blood Sugar >120? (1/0): "))
    restecg = float(input("Resting ECG (0,1,2): "))
    thalach = float(input("Max Heart Rate: "))
    exang = float(input("Exercise Induced Angina? (1/0): "))
    oldpeak = float(input("ST Depression (0.0–6.0): "))
    slope = float(input("ST Slope (0=upsloping,1=flat,2=downsloping): "))
    ca = float(input("Major Vessels Colored (0–3): "))
    thal = float(input("Thalassemia (1=normal,2=fixed,3=reversible): "))

    return name, {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }

# ======================
# Main Program
# ======================
def main():
    print("🫀 HEART ATTACK RISK PREDICTOR")
    print("Choose input method:")
    print("1. 📝 Enter details manually")
    print("2. 📎 Upload medical report (PDF/Image)")
    
    choice = input("Enter 1 or 2: ").strip()
    
    if choice == "2":
        if not use_ai:
            print("❌ AI analysis requires a valid Google API key.")
            return
        
        file_path = input("Enter full path to medical report file: ").strip()
        if not os.path.exists(file_path):
            print("❌ File not found.")
            return
        
        extracted = extract_data_from_report(file_path)
        if not extracted:
            print("❌ Could not extract data. Switching to manual input.")
            name, patient_data = get_manual_input()
        else:
            print(f"✅ Extracted data: {extracted}")
            name = input("Patient name (for records): ").strip() or "anonymous"
            
            required = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
            missing = [f for f in required if f not in extracted]
            if missing:
                print(f"\n⚠️ Missing fields: {', '.join(missing)}")
                for field in missing:
                    val = float(input(f"Enter {field}: "))
                    extracted[field] = val
            patient_data = extracted

    elif choice == "1":
        name, patient_data = get_manual_input()
    else:
        print("❌ Invalid choice.")
        return

    # Prediction
    try:
        X_input = np.array([[patient_data[col] for col in ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]])
        X_input_scaled = scaler.transform(X_input)
        prob = ensemble.predict_proba(X_input_scaled)[0]
        prob_heart_disease = prob[1]
        confidence = prob_heart_disease * 100

        if prob_heart_disease >= 0.70:
            risk = "high"
            print(f"\n⚠️  HIGH RISK (Confidence: {confidence:.1f}%)")
        elif prob_heart_disease >= 0.35:
            risk = "moderate"
            print(f"\n🔶 MODERATE RISK (Confidence: {confidence:.1f}%)")
        else:
            risk = "low"
            print(f"\n✅ LOW RISK (Confidence: {confidence:.1f}%)")

        advice = get_health_advice(patient_data, risk)
        hospital = recommend_hospital(risk)

        print(f"\n💡 HEALTH ADVICE:\n{advice}")
        print(f"\n🏥 RECOMMENDED HOSPITAL:\n{hospital}")

        log_prediction(name, patient_data, risk, confidence, advice, hospital)
        print(f"\n📄 Prediction logged to 'predictions_log.csv'")

    except Exception as e:
        print(f"❌ Prediction error: {e}")

if __name__ == "__main__":
    main()

