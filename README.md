# Heart Disease Predictor - Streamlit

Simple Streamlit app that loads a pickled model (`heartdisease.pkl`) and predicts heart disease.

## Files
- app.py         -> Streamlit application
- heartdisease.pkl -> Your trained model (place in same folder)
- requirements.txt

## Run locally
1. Create a virtual environment (optional)
2. Install dependencies:
   pip install -r requirements.txt
3. Start the app:
   streamlit run app.py

## Notes
- The app expects features in this order:
  ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeack','slope','ca','thal']
- Ensure the model was trained with the same ordering and encodings. If not, edit the preprocessing block in `app.py`.
- This is a demo only — not medical advice.
