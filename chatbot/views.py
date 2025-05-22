from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods, require_GET
from django.shortcuts import render
from django.http import JsonResponse
import json
import numpy as np
import tensorflow as tf
import os
import re
import random
import logging
import traceback

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==== LOAD MODEL AND DATA ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    # Load symptom_index.json
    symptom_index_path = os.path.join(BASE_DIR, "model/symptom_index.json")
    if not os.path.exists(symptom_index_path):
        raise FileNotFoundError(f"Could not find symptom_index.json at {symptom_index_path}")
    
    with open(symptom_index_path, "r", encoding="utf-8") as f:
        symptom_index = json.load(f)
    logger.info(f"Successfully loaded symptom_index.json with {len(symptom_index)} symptoms")

    # Load label_encoder.json
    label_encoder_path = os.path.join(BASE_DIR, "model/label_encoder.json")
    if not os.path.exists(label_encoder_path):
        raise FileNotFoundError(f"Could not find label_encoder.json at {label_encoder_path}")
    
    with open(label_encoder_path, "r", encoding="utf-8") as f:
        disease_classes = json.load(f)
    logger.info(f"Successfully loaded label_encoder.json with {len(disease_classes)} diseases")

    # Load disease symptoms
    disease_data_path = os.path.join(BASE_DIR, "data/diseases_50_vietnam.json")
    with open(disease_data_path, "r", encoding="utf-8") as f:
        disease_data = json.load(f)
        disease_symptoms = {entry["name"]: entry["symptoms"] for entry in disease_data}
    logger.info("Đã load danh sách triệu chứng của các bệnh thành công")

    # Load model
    model_path = os.path.join(BASE_DIR, "model/medical_model.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Could not find model file (.h5)")
    
    logger.info(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)
    logger.info("Model loaded successfully")

except Exception as e:
    logger.error(f"Error during initialization: {str(e)}")
    logger.error(traceback.format_exc())
    raise

# ==== NATURAL CONVERSATION ====
conversation_phrases = {
    "hello": ["Hello! What symptoms are you experiencing?", "Hi there! I'm your AI medical assistant. How can I help you?"],
    "hi": ["Hello! What symptoms are you experiencing?", "Hi! How can I assist you today?"],
    "introduction": ["I am an AI medical chatbot. I can help diagnose common diseases based on symptoms."],
    "goodbye": ["Take care! Hope you feel better soon!", "Goodbye! Have a great day!"],
    "thanks": ["You're welcome! I'm here to help.", "Happy to help!"]
}

def is_smalltalk(user_input):
    user_input = user_input.lower().strip()
    for key in conversation_phrases:
        if key in user_input:
            return random.choice(conversation_phrases[key])
    return None

def is_positive_response(text):
    """Check if the response is positive"""
    text = text.lower().strip()
    positive_patterns = [
        r"\byes\b",
        r"\by\b",
        r"\bok\b",
        r"\bcorrect\b",
        r"\bright\b",
        r"\btrue\b",
        r"^yes$",
        r"^y$",
        r"^ok$"
    ]
    return any(re.search(pattern, text) for pattern in positive_patterns)

def is_negative_response(text):
    """Check if the response is negative"""
    text = text.lower().strip()
    negative_patterns = [
        r"\bno\b",
        r"\bn\b",
        r"\bnot\b",
        r"\bwrong\b",
        r"\bfalse\b",
        r"^no$",
        r"^n$"
    ]
    return any(re.search(pattern, text) for pattern in negative_patterns)

# ==== PREPROCESSING ====
def extract_symptoms_from_text(user_input, symptom_index):
    try:
        clean_text = user_input.lower()
        clean_text = re.sub(r"[\.,;:!?]", " ", clean_text)  # Remove punctuation
        clean_text = re.sub(r"[^\w\s]", "", clean_text)  # Remove special characters
        found = []
        vector = [0] * len(symptom_index)

        # Support fuzzy matching
        for symptom, index in symptom_index.items():
            if symptom in clean_text or any(word in clean_text for word in symptom.split()):
                found.append(symptom)
                vector[index] = 1
        return vector, list(set(found))
    except Exception as e:
        logger.error(f"Error extracting symptoms: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# ==== HTML INTERFACE ====
def chat_web(request):
    return render(request, "chatbot_chatbox.html")

def get_next_symptom(remaining_symptoms, collected_symptoms):
    """Get next symptom to ask about, skipping already asked symptoms"""
    for symptom in remaining_symptoms:
        if symptom not in collected_symptoms:
            return symptom
    return None

# ==== CONVERSATION HANDLING ====
@csrf_exempt
@require_http_methods(["POST"])
def chatbot_predict(request):
    try:
        # Parse request data
        try:
            data = json.loads(request.body)
            user_input = data.get("message", "").strip()
            if not user_input:
                return JsonResponse({
                    "response": "Sorry, I didn't receive your message. Please try again."
                })
            logger.info(f"Received message: {user_input}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {str(e)}")
            return JsonResponse({
                "response": "Invalid data. Please try again."
            }, status=400)

        # Initialize or get session
        session = request.session
        if "symptoms_collected" not in session:
            session["symptoms_collected"] = []
        if "current_disease" not in session:
            session["current_disease"] = None
        if "remaining_symptoms" not in session:
            session["remaining_symptoms"] = []
        if "last_asked_symptom" not in session:
            session["last_asked_symptom"] = None
        if "asked_symptoms" not in session:
            session["asked_symptoms"] = []

        # Check for small talk
        smalltalk_response = is_smalltalk(user_input)
        if smalltalk_response:
            return JsonResponse({"response": f"🤖 {smalltalk_response}"})

        # Handle yes/no responses for current symptom
        if session["remaining_symptoms"] and session["current_disease"] and session["last_asked_symptom"]:
            current_symptom = session["last_asked_symptom"]
            
            if is_positive_response(user_input):
                logger.info(f"Received YES response for symptom: {current_symptom}")
                if current_symptom not in session["symptoms_collected"]:
                    session["symptoms_collected"].append(current_symptom)
                session["asked_symptoms"].append(current_symptom)
            elif is_negative_response(user_input):
                logger.info(f"Received NO response for symptom: {current_symptom}")
                session["asked_symptoms"].append(current_symptom)
            else:
                return JsonResponse({
                    "response": f"🤖 I'm not sure about your response. Please answer 'yes' or 'no'.\nDo you have **{current_symptom}**?"
                })

            # Get next symptom to ask about
            next_symptom = get_next_symptom(session["remaining_symptoms"], session["asked_symptoms"])
            if next_symptom:
                session["last_asked_symptom"] = next_symptom
                return JsonResponse({
                    "response": f"🤖 Do you have **{next_symptom}**?"
                })
            else:
                session["last_asked_symptom"] = None

        # Extract symptoms from user input
        input_vector, matched = extract_symptoms_from_text(user_input, symptom_index)
        logger.info(f"Found symptoms: {matched}")

        # Add new symptoms to the list
        for sym in matched:
            if sym not in session["symptoms_collected"]:
                session["symptoms_collected"].append(sym)
                session["asked_symptoms"].append(sym)

        if not session["symptoms_collected"]:
            # Suggest common symptoms if none recognized
            common_symptoms = list(symptom_index.keys())[:5]
            return JsonResponse({
                "response": "🤖 I don't see any clear symptoms in your message. "
                            "Could you describe them more specifically? For example: 'I have fever and sore throat'.\n"
                            f"Some common symptoms: {', '.join(common_symptoms)}"
            })

        # Create vector from all collected symptoms
        vector = [0] * len(symptom_index)
        for sym in session["symptoms_collected"]:
            idx = symptom_index.get(sym)
            if idx is not None:
                vector[idx] = 1

        # Make prediction
        try:
            x = np.array([vector], dtype=np.float32)
            predictions = model(x, training=False).numpy()[0]
            # Get top 3 diseases with highest probability
            top_indices = predictions.argsort()[-3:][::-1]
            confidences = [float(predictions[i]) for i in top_indices]
            diseases = [disease_classes[str(i)] for i in top_indices]
            logger.info(f"Top predicted diseases: {diseases} with confidence: {confidences}")
            
            # Format response
            response = "🤖 Based on your symptoms, here are the possible conditions:\n\n"
            response += f"Symptoms detected: {', '.join(session['symptoms_collected'])}\n\n"
            for idx, (disease, confidence) in enumerate(zip(diseases, confidences)):
                if idx == 0:
                    response += f"• <b>{disease}</b> (Confidence: {confidence:.1%})\n"
                else:
                    response += f"• {disease} (Confidence: {confidence:.1%})\n"
            
            response += "\nPlease note: This is just a preliminary assessment. Please consult a healthcare professional for proper diagnosis."
            
            return JsonResponse({"response": response})
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return JsonResponse({
                "response": "🤖 I'm having trouble analyzing your symptoms. Please try again."
            })

    except Exception as e:
        logger.error(f"Error in chatbot_predict: {str(e)}")
        logger.error(traceback.format_exc())
        return JsonResponse({
            "response": "🤖 An error occurred. Please try again."
        })

@csrf_exempt
@require_GET
def reset_session(request):
    """Reset the chat session"""
    request.session.flush()
    return JsonResponse({"response": "🤖 Session reset. How can I help you today?"})
