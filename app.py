from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import shap  # Import SHAP

# Initialize Flask App
app = Flask(__name__)
CORS(app)

# --- Load Artifacts ---
model = load_model(r'D:/DL_Project/loan_default_model1.h5')
tokenizer = joblib.load(r'D:/DL_Project/tokenizer1.pkl')
preprocessor = joblib.load(r'D:/DL_Project/preprocessor1.pkl')
print("Model and artifacts loaded successfully!")

# --- Create SHAP Explainer (runs once at startup) ---
# --- Create SHAP Explainer (runs once at startup) ---

# Get the list of categorical and numerical feature names from the preprocessor
categorical_features = preprocessor.named_transformers_['cat'].feature_names_in_
numerical_features = preprocessor.named_transformers_['num'].feature_names_in_

# Create a dummy DataFrame with one row of placeholder data
background_data = {}
for col in numerical_features:
    background_data[col] = [0]  # Placeholder for numerical columns
for col in categorical_features:
    background_data[col] = ['missing'] # Placeholder for categorical columns

background_static_df = pd.DataFrame(background_data)

# Now, transform this correctly-typed DataFrame
background_static = preprocessor.transform(background_static_df)
background_text = np.zeros((1, 50))

# Define a prediction function that SHAP can use
def predict_for_shap(x_static):
    num_samples = x_static.shape[0]
    text_data_tiled = np.tile(background_text, (num_samples, 1))
    return model.predict({'text_input': text_data_tiled, 'static_input': x_static})

explainer = shap.KernelExplainer(predict_for_shap, background_static)
print("SHAP explainer created successfully!")

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to predict loan default and provide explanations."""
    json_data = request.get_json()
    new_data = pd.DataFrame(json_data, index=[0])

    # --- Pre-process Data ---
    new_text_data = new_data['title'].fillna('') + ' ' + new_data['purpose']
    static_features_cols = preprocessor.feature_names_in_
    new_static_data = new_data[static_features_cols]
    
    new_sequences = tokenizer.texts_to_sequences(new_text_data)
    padded_new_sequences = pad_sequences(new_sequences, maxlen=50, padding='post', truncating='post')
    processed_new_static = preprocessor.transform(new_static_data)
    
    # --- Make Prediction ---
    prediction_proba = model.predict({
        'text_input': padded_new_sequences,
        'static_input': processed_new_static
    })

    # --- Get SHAP Explanation ---
    shap_values = explainer.shap_values(processed_new_static)
    
    # Get original feature names from the preprocessor
    feature_names = preprocessor.get_feature_names_out()

    # Format the explanation
    explanation = []
    for feature, shap_val in zip(feature_names, shap_values[0][0]):
         explanation.append({"feature": feature.replace('cat__', '').replace('num__', ''), "value": shap_val})

    # Sort by the absolute SHAP value to find the most influential features
    explanation.sort(key=lambda x: abs(x['value']), reverse=True)
    top_explanations = explanation[:5]

    # --- Format and Return Response ---
    output = {
        'default_probability': float(prediction_proba[0][0]),
        'explanation': top_explanations  # Add explanations to the output
    }
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True, port=5000)



# Extra code 


# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import joblib
# import pandas as pd
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import shap

# # Initialize Flask App
# app = Flask(__name__)
# CORS(app)

# # --- Load Artifacts ---
# model = load_model(r'D:/DL_Project/loan_default_model.h5')
# tokenizer = joblib.load(r'D:/DL_Project/tokenizer.pkl')
# preprocessor = joblib.load(r'D:/DL_Project/preprocessor.pkl')
# print("Model and artifacts loaded successfully!")

# # --- Create SHAP Explainer using a real data sample ---
# # Load the background data you saved from your notebook
# background_df = pd.read_csv('shap_background_data.csv')
# # Pre-process it just like the training data
# background_static = preprocessor.transform(background_df)
# # We still need a placeholder for the text data
# background_text = np.zeros((1, 50))

# # To make the explainer faster, we can summarize the background data
# # We'll use SHAP's kmeans to create a smaller, weighted summary
# background_summary = shap.kmeans(background_static, 10)

# # Define a prediction function that SHAP can use
# def predict_for_shap(x_static):
#     num_samples = x_static.shape[0]
#     # For each static sample from SHAP, we need a corresponding text sample.
#     text_data_tiled = np.tile(background_text, (num_samples, 1))
#     return model.predict({'text_input': text_data_tiled, 'static_input': x_static})

# # Create the explainer with the summarized background data
# explainer = shap.KernelExplainer(predict_for_shap, background_summary)
# print("SHAP explainer created successfully with real background data!")


# @app.route('/predict', methods=['POST'])
# def predict():
#     """API endpoint to predict loan default and provide explanations."""
#     json_data = request.get_json()
#     new_data = pd.DataFrame(json_data, index=[0])

#     # --- Pre-process Data ---
#     new_text_data = new_data['title'].fillna('') + ' ' + new_data['purpose']
#     static_features_cols = preprocessor.feature_names_in_
#     new_static_data = new_data[static_features_cols]
    
#     new_sequences = tokenizer.texts_to_sequences(new_text_data)
#     padded_new_sequences = pad_sequences(new_sequences, maxlen=50, padding='post', truncating='post')
#     processed_new_static = preprocessor.transform(new_static_data)
    
#     # --- Make Prediction ---
#     prediction_proba = model.predict({
#         'text_input': padded_new_sequences,
#         'static_input': processed_new_static
#     })

#     # --- Get SHAP Explanation ---
#     shap_values = explainer.shap_values(processed_new_static)
    
#     feature_names = preprocessor.get_feature_names_out()

#     explanation = []
#     for feature, shap_val in zip(feature_names, shap_values[0][0]):
#          explanation.append({"feature": feature.replace('cat__', '').replace('num__', ''), "value": shap_val})

#     explanation.sort(key=lambda x: abs(x['value']), reverse=True)
#     top_explanations = explanation[:5]

#     # --- Format and Return Response ---
#     output = {
#         'default_probability': float(prediction_proba[0][0]),
#         'explanation': top_explanations
#     }
#     return jsonify(output)

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)