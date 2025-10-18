import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import warnings
import random
warnings.filterwarnings('ignore')

GOOGLE_API_KEY = "ENTER_YOUR_API_KEY_HERE"

use_ai = False
model = None
if GOOGLE_API_KEY.strip() and GOOGLE_API_KEY != "your_real_key_here":
    try:
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        use_ai = True
    except Exception as e:
        print(f"⚠️ Failed to initialize AI: {e}")

try:
    import pyttsx3
    tts_engine = pyttsx3.init()
except Exception as e:
    tts_engine = None
    print(f"🔇 Voice disabled: {e}")

def speak(text):
    if tts_engine:
        clean = text.replace("**", "").replace("✅", "Good news:").replace("⚠️", "Warning:").replace("💡", "Advice:").replace("🔶", "Moderate risk:").replace("-", "•")
        tts_engine.say(clean)
        tts_engine.runAndWait()

# Load heart data
try:
    data = pd.read_csv('heart.csv')
except FileNotFoundError:
    error_msg = "❌ 'heart.csv' not found."
    print(error_msg)
    speak(error_msg)
    exit()

# Load hospitals
try:
    hospitals_df = pd.read_csv('hospitals.csv')
    # Ensure required columns exist
    if not {'name', 'location', 'risk_level'}.issubset(hospitals_df.columns):
        raise ValueError("Missing columns in hospitals.csv")
    # Normalize risk_level to lowercase
    hospitals_df['risk_level'] = hospitals_df['risk_level'].str.lower()
except Exception as e:
    error_msg = f"❌ Failed to load hospitals.csv: {e}"
    print(error_msg)
    speak(error_msg)
    hospitals_df = None

required_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
if not all(col in data.columns for col in required_cols):
    error_msg = "❌ Dataset missing required columns."
    print(error_msg)
    speak(error_msg)
    exit()

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
        except Exception as e:
            print(f"⚠️ AI failed: {e}")

    if risk_level == "high":
        return (
            "Warning: High-Risk Recommendations. "
            "Consult a cardiologist immediately. "
            "Eat: Oats, salmon, spinach, walnuts, and blueberries. "
            "Avoid: Fried foods, processed meats, sugary drinks, and excess salt. "
            "Exercise: 30 minutes of walking daily, only if approved by your doctor. "
            "Monitor your blood pressure and cholesterol weekly."
        )
    elif risk_level == "moderate":
        return (
            "Caution: Moderate-Risk Recommendations. "
            "See your doctor within 1–2 weeks for evaluation. "
            "Eat: Avocados, lentils, almonds, berries, and olive oil. "
            "Avoid: White bread, pastries, and salty snacks. "
            "Start light exercise like walking 20 minutes daily. "
            "Reduce stress with meditation or yoga."
        )
    else:  # low
        return (
            "Good news: Low-Risk Maintenance. "
            "Eat whole grains, fruits, vegetables, and lean proteins like chicken or beans. "
            "Limit salt to less than one teaspoon per day and avoid butter or red meat. "
            "Stay active: walk 30 minutes, five days a week. "
            "Avoid smoking and manage stress with deep breathing or meditation. "
            "Get an annual heart check-up."
        )

def recommend_hospital(risk_level):
    if hospitals_df is None:
        return "Hospital data unavailable."
    
    # Filter by risk level
    candidates = hospitals_df[hospitals_df['risk_level'] == risk_level]
    if candidates.empty:
        # Fallback to low-risk if no match
        candidates = hospitals_df[hospitals_df['risk_level'] == 'low']
    
    if candidates.empty:
        return "No hospitals available."

    # Pick random hospital
    selected = candidates.sample(n=1).iloc[0]
    name = selected['name']
    location = selected['location']
    desc = selected.get('description', '')
    return f"{name}, {location}. {desc}".strip()

import datetime

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
    
    try:
        with open('predictions_log.csv', 'r') as f:
            header_exists = True
    except FileNotFoundError:
        header_exists = False

    with open('predictions_log.csv', 'a', newline='', encoding='utf-8') as f:
        if not header_exists:
            f.write('timestamp,name,age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,risk,confidence,advice,hospital\n')
        f.write('"' + '","'.join(str(x) for x in row) + '"\n')

def predict_and_advise():
    intro = "Welcome to the Heart Attack Risk Predictor. Please enter your details."
    print("🫀 Heart Attack Risk Prediction + Hospital Recommendation\n")
    speak(intro)

    try:
        name = input("Patient name: ").strip()
        if not name:
            name = "anonymous"

        age = float(input("Age (years): "))
        sex = float(input("Sex (1 = Male, 0 = Female): "))
        cp = float(input("Chest Pain Type (0-3): "))
        trestbps = float(input("Resting Blood Pressure (mm Hg): "))
        chol = float(input("Serum Cholesterol (mg/dl): "))
        fbs = float(input("Fasting Blood Sugar > 120? (1/0): "))
        restecg = float(input("Resting ECG (0,1,2): "))
        thalach = float(input("Max Heart Rate Achieved: "))
        exang = float(input("Exercise Induced Angina? (1/0): "))
        oldpeak = float(input("ST Depression (0.0–6.0): "))
        slope = float(input("Slope of ST segment (0,1,2): "))
        ca = float(input("Major vessels colored (0–3): "))
        thal = float(input("Thalassemia (1,2,3): "))

        patient_data = {
            'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
            'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
            'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
        }

        X_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                             thalach, exang, oldpeak, slope, ca, thal]])
        X_input_scaled = scaler.transform(X_input)
        prob = ensemble.predict_proba(X_input_scaled)[0]
        prob_heart_disease = prob[1]
        confidence = prob_heart_disease * 100

        if prob_heart_disease >= 0.70:
            risk = "high"
            risk_display = "⚠️ HIGH RISK"
        elif prob_heart_disease >= 0.35:
            risk = "moderate"
            risk_display = "🔶 MODERATE RISK"
        else:
            risk = "low"
            risk_display = "✅ LOW RISK"

        result_display = f"{risk_display} (Confidence: {confidence:.1f}%)"
        result_voice = f"{risk.replace('moderate', 'Moderate').replace('high', 'High').replace('low', 'Low')} risk of heart disease. Confidence: {confidence:.1f} percent."

        print(f"\n{result_display}")
        speak(result_voice)

        advice = get_health_advice(patient_data, risk)
        hospital = recommend_hospital(risk)

        print("\n💡 Personalized Health & Diet Advice:")
        print(advice)
        speak("Here is your personalized health advice:")
        speak(advice)

        print(f"\n🏥 Recommended Hospital:\n{hospital}")
        speak("Recommended hospital:")
        speak(hospital)

        log_prediction(name, patient_data, risk, confidence, advice, hospital)

    except Exception as e:
        error_msg = f"Input error: {e}"
        print(f"❌ {error_msg}")
        speak(error_msg)

if __name__ == "__main__":
    predict_and_advise()

