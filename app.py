from flask import Flask, request, render_template,jsonify
import joblib 
import pandas as pd

model=joblib.load('Text_Classification_Model.pkl')
vect=joblib.load('vectorizerfortextclassification.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    # Process the data (for this example, just return it)
    data=model.predict(vect.transform(pd.Series((request.form.get(('user-input'))))))
    # Render the result in the respons
    return str(data)

if __name__ == '__main__':
    app.run(debug=True)
