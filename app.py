from flask import Flask, jsonify,request,send_from_directory
from gtts import gTTS
from flask_cors import CORS
import os
import pickle
import statistics
import joblib
import numpy as np
from googletrans import Translator, LANGUAGES
from werkzeug.exceptions import BadRequest
import asyncio
import nest_asyncio


# Flask constructor takes the name of 
# current module (__name__) as argument.
app = Flask(__name__)
CORS(app)  

model = pickle.load(open('model.pkl', 'rb'))
final_rf_model = joblib.load("final_rf_model.pkl")
final_nb_model = joblib.load("final_nb_model.pkl")
final_svm_model = joblib.load("final_svm_model.pkl")
encoder = joblib.load("label_encoder.pkl")
# The route() function of the Flask class is a decorator, 
# which tells the application which URL should call 
# the associated function.
@app.route('/')
def hello_world():
    return 'Hello World'



# Directory to store temporary audio files

UPLOAD_FOLDER = 'audio_files'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

LANGUAGE_CODES = {
    'en': 'en',    # English
    'hi': 'hi',    # Hindi
    'kn': 'kn',    # Kannada
    'es': 'es',    # Spanish
    'fr': 'fr',    # French
    'de': 'de'     # German
}

@app.route('/generate-speech', methods=['POST'])
def generate_speech():
    try:
        # Get text from request body
        data = request.get_json()
        text = data['text']
        language = data.get('language', 'en')  # Default to English if no language specified
        
        # Validate language
        if language not in LANGUAGE_CODES:
            return jsonify({'error': f'Unsupported language: {language}'}), 400

        # Generate speech using GTTS
        tts = gTTS(text=text, lang=LANGUAGE_CODES[language], slow=False)
        
        # Create a temporary file to save the speech
        temp_audio_path = os.path.join(UPLOAD_FOLDER, 'speech.mp3')
        tts.save(temp_audio_path)
        
        # Return the path of the generated MP3 file as a URL
        return jsonify({'audioUrl': f'/audio/{os.path.basename(temp_audio_path)}'}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Serve the audio file as a static file
AUDIO_DIRECTORY = os.path.join(os.getcwd(), 'audio_files')

@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_from_directory(AUDIO_DIRECTORY, filename)


# Symptom dictionary
symptom_index = {'itching': 0, 'skin rash': 1, 'nodal skin eruptions': 2, 'continuous sneezing': 3, 'shivering': 4, 'chills': 5, 'joint pain': 6, 'stomach pain': 7, 'acidity': 8, 'ulcers on tongue': 9, 'muscle wasting': 10, 'vomiting': 11, 'burning micturition': 12, 'spotting  urination': 13, 'fatigue': 14, 'weight gain': 15, 'anxiety': 16, 'cold hands and feets': 17, 'mood swings': 18, 'weight loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches in throat': 22, 'irregular sugar level': 23, 'cough': 24, 'fever': 25, 'sunken eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish skin': 32, 'dark urine': 33, 'nausea': 34, 'loss of appetite': 35, 'pain behind the eyes': 36, 'back pain': 37, 'constipation': 38, 'abdominal pain': 39, 'diarrhoea': 40, 'high fever': 41, 'yellow urine': 42, 'yellowing of eyes': 43, 'acute liver failure': 44, 'fluid overload': 45, 'swelling of stomach': 46, 'swelled lymph nodes': 47, 'malaise': 48, 'blurred and distorted vision': 49, 'phlegm': 50, 'throat irritation': 51, 'redness of eyes': 52, 'sinus pressure': 53, 'runny nose': 54, 'congestion': 55, 'chest pain': 56, 'weakness in limbs': 57, 'fast heart rate': 58, 'pain during bowel movements': 59, 'pain in anal region': 60, 'bloody stool': 61, 'irritation in anus': 62, 'neck pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen legs': 68, 'swollen blood vessels': 69, 'puffy face and eyes': 70, 'enlarged thyroid': 71, 'brittle nails': 72, 'swollen extremeties': 73, 'excessive hunger': 74, 'extra marital contacts': 75, 'drying and tingling lips': 76, 'slurred speech': 77, 'knee pain': 78, 'hip joint pain': 79, 'muscle weakness': 80, 'stiff neck': 81, 'swelling joints': 82, 'movement stiffness': 83, 'spinning movements': 84, 'loss of balance': 85, 'unsteadiness': 86, 'weakness of one body side': 87, 'loss of smell': 88, 'bladder discomfort': 89, 'foul smell of urine': 90, 'continuous feel of urine': 91, 'passage of gases': 92, 'internal itching': 93, 'toxic look (typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle pain': 97, 'altered sensorium': 98, 'red spots over body': 99, 'belly pain': 100, 'abnormal menstruation': 101, 'dischromic  patches': 102, 'watering from eyes': 103, 'increased appetite': 104, 'polyuria': 105, 'family history': 106, 'mucoid sputum': 107, 'rusty sputum': 108, 'lack of concentration': 109, 'visual disturbances': 110, 'receiving blood transfusion': 111, 'receiving unsterile injections': 112, 'coma': 113, 'stomach bleeding': 114, 'distention of abdomen': 115, 'history of alcohol consumption': 116, 'fluid overload.1': 117, 'blood in sputum': 118, 'prominent veins on calf': 119, 'palpitations': 120, 'painful walking': 121, 'pus filled pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin peeling': 125, 'silver like dusting': 126, 'small dents in nails': 127, 'inflammatory nails': 128, 'blister': 129, 'red sore around nose': 130, 'yellow crust ooze': 131}


data_dict = {
    "symptom_index": symptom_index,
    "predictions_classes": encoder.classes_
}

def predictDisease(symptoms):
    try:
       
        # Split symptoms string into list
        symptom_list = [s.strip().lower() for s in symptoms.split(",")]
        
        # Create input data array
        input_data = [0] * len(data_dict["symptom_index"])
        for symptom in symptom_list:
            if symptom in data_dict["symptom_index"]:
                index = data_dict["symptom_index"][symptom]
                input_data[index] = 1
            
        
        # Reshape input data
        input_data = np.array(input_data).reshape(1, -1)
        
        # Generate predictions from all models
        rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
        nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
        svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]
        
        # Make final prediction using mode
        final_prediction = statistics.mode([rf_prediction, nb_prediction, svm_prediction])
        
        return {
            "rf_model_prediction": rf_prediction,
            "naive_bayes_prediction": nb_prediction,
            "svm_model_prediction": svm_prediction,
            "final_prediction": final_prediction
        }
    except Exception as e:
        raise Exception(f"Error in prediction: {str(e)}")


@app.route('/predict-disease', methods=['POST'])
def predict_disease():
    # Check if symptoms were provided in the JSON request
    try:
        # Get data from request
        data = request.get_json()
        if not data or 'symptoms' not in data:
            return jsonify({'error': 'No symptoms provided'}), 400
            
        # Get symptoms string
        symptoms = data['symptoms']
        print("sum", symptoms,data['symptoms'])
        # Make prediction
        predictions = predictDisease(symptoms)
        
        # Return response
        return jsonify({
            'disease': predictions['final_prediction'],
            'details': {
                'random_forest': predictions['rf_model_prediction'],
                'naive_bayes': predictions['naive_bayes_prediction'],
                'svm': predictions['svm_model_prediction']
            }
        })
        
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500





translator = Translator()
nest_asyncio.apply()

@app.route('/translate', methods=['POST'])
def translate_text():
    """
    Translate text to a specified target language.
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        text = data['text']
        target_lang = data['target_language'].lower()
        # from_lang = data.get('source', 'auto').lower()

        # Run the coroutine using asyncio
        translation = asyncio.run(translator.translate(text, dest=target_lang))
        print("dd",translation)
        # Prepare response
        response = {
            'translated_text': translation.text,
            'source_language': translation.src,
            'target_language': translation.dest
        }
        return jsonify(response), 200

    except BadRequest as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Main driver function
if __name__ == '__main__':
    # Run the application on the local development server.
    app.run(debug=True)
