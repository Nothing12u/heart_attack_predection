# heart_attack_predection


Here’s the **README.md** without any emojis — clean and professional for GitHub:

---

# Heart Attack Risk Predictor with Voice and AI Health Advice

An intelligent **Heart Attack Risk Prediction System** that uses **Machine Learning** and optional **Google Generative AI** for personalized health and diet recommendations.
It also provides **voice feedback** using `pyttsx3`, making it accessible and interactive.

---

## Features

* Predicts heart attack risk using an **ensemble ML model** (Random Forest + Logistic Regression + SVM).
* Provides **personalized lifestyle and diet advice** for low, moderate, and high-risk patients.
* Offers **voice-based interaction** with `pyttsx3`.
* Optional **AI-generated advice** using **Google Gemini API**.
* Automatically logs predictions to a CSV file for record-keeping.

---

## Tech Stack

* **Python 3.8+**
* **Libraries:**

  * `pandas`, `numpy`
  * `scikit-learn`
  * `pyttsx3`
  * `google-generativeai` *(optional for AI advice)*

---

## Project Structure

```
├── main.py                 # Main script for prediction and advice
├── heart.csv               # Dataset (must be in same folder)
├── predictions_log.csv     # Auto-generated prediction logs
└── README.md               # Project documentation
```

---

## Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/heart-risk-predictor.git
   cd heart-risk-predictor
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   *(If you don’t have a `requirements.txt`, create one with:)*

   ```bash
   pip install pandas numpy scikit-learn pyttsx3 google-generativeai
   pip freeze > requirements.txt
   ```

3. **Add your dataset**

   * Make sure `heart.csv` is present in the project folder.
   * The file must include these columns:

     ```
     age, sex, cp, trestbps, chol, fbs, restecg,
     thalach, exang, oldpeak, slope, ca, thal, target
     ```

4. **(Optional) Enable AI-generated advice**

   * Get a **Google API Key** from [Google AI Studio](https://aistudio.google.com/).
   * Replace this line in `main.py`:

     ```python
     GOOGLE_API_KEY = "your_real_key_here"
     ```

     with:

     ```python
     GOOGLE_API_KEY = "YOUR_API_KEY"
     ```

5. **Run the program**

   ```bash
   python main.py
   ```

---

## How It Works

1. The system asks for patient details like age, BP, cholesterol, etc.
2. It predicts heart disease risk using an ensemble of ML classifiers.
3. Based on the prediction, it provides **personalized advice**:

   * Low risk → Maintenance tips
   * Moderate risk → Early lifestyle corrections
   * High risk → Urgent doctor consultation
4. All results are **spoken aloud** and saved in `predictions_log.csv`.

---

## Output Example

```
Heart Attack Risk Prediction + Voice Advice

Patient name: John
Age (years): 45
Sex (1 = Male, 0 = Female): 1
...

LOW RISK (Confidence: 82.5%)

Personalized Health & Diet Advice:
Eat whole grains, fruits, vegetables, and lean proteins.
Avoid butter or red meat. Stay active and monitor your health regularly.
```

---

## Logs

Each prediction is saved automatically in `predictions_log.csv` with:

* Timestamp
* Patient name
* Input values
* Risk level
* Confidence
* Health advice

---

## Author

**mr.X**
Email: *[your.email@example.com](mailto:your.email@example.com)*
GitHub: [https://github.com/yourusername](https://github.com/yourusername)

---

## Disclaimer

> This tool is for **educational and informational purposes only**.
> It does **not replace medical diagnosis or professional advice**.
> Always consult a certified healthcare provider for medical concerns.

---

Would you like me to add a short **project description paragraph** for the top of your GitHub page (for the repository “About” section)?
