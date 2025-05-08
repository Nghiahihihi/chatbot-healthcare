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

# ==== LOAD MÔ HÌNH VÀ DỮ LIỆU ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    # Load symptom_index.json
    symptom_index_path = os.path.join(BASE_DIR, "model/symptom_index.json")
    if not os.path.exists(symptom_index_path):
        raise FileNotFoundError(f"Không tìm thấy file symptom_index.json tại {symptom_index_path}")
    
    with open(symptom_index_path, "r", encoding="utf-8") as f:
        symptom_index = json.load(f)
    logger.info(f"Đã load symptom_index.json thành công với {len(symptom_index)} triệu chứng")

    # Load label_encoder.json
    label_encoder_path = os.path.join(BASE_DIR, "model/label_encoder.json")
    if not os.path.exists(label_encoder_path):
        raise FileNotFoundError(f"Không tìm thấy file label_encoder.json tại {label_encoder_path}")
    
    with open(label_encoder_path, "r", encoding="utf-8") as f:
        disease_classes = json.load(f)
    logger.info(f"Đã load label_encoder.json thành công với {len(disease_classes)} bệnh")

    # Load disease symptoms
    disease_data_path = os.path.join(BASE_DIR, "data/diseases_50_vietnam.json")
    with open(disease_data_path, "r", encoding="utf-8") as f:
        disease_data = json.load(f)
        disease_symptoms = {entry["name"]: entry["symptoms"] for entry in disease_data}
    logger.info("Đã load danh sách triệu chứng của các bệnh thành công")

    # Load model
    model_path = os.path.join(BASE_DIR, "model/medical_model.keras")
    if not os.path.exists(model_path):
        model_path = os.path.join(BASE_DIR, "model/medical_model.h5")
        if not os.path.exists(model_path):
            raise FileNotFoundError("Không tìm thấy file model (.keras hoặc .h5)")
    
    logger.info(f"Đang load model từ {model_path}")
    model = tf.keras.models.load_model(model_path)
    logger.info("Đã load model thành công")

except Exception as e:
    logger.error(f"Lỗi khi khởi tạo: {str(e)}")
    logger.error(traceback.format_exc())
    raise

# ==== TRÒ CHUYỆN TỰ NHIÊN ====
conversation_phrases = {
    "chào": ["Xin chào! Bạn đang gặp triệu chứng gì vậy?", "Chào bạn, mình là trợ lý y tế AI, bạn cần mình giúp gì?"],
    "giới thiệu": ["Tôi là chatbot AI y tế. Tôi có thể hỗ trợ bạn chẩn đoán các bệnh phổ biến dựa trên triệu chứng."],
    "tạm biệt": ["Chúc bạn nhiều sức khỏe. Hẹn gặp lại!", "Tạm biệt! Chúc bạn một ngày tốt lành."],
    "cảm ơn": ["Bạn không cần cảm ơn đâu. Mình ở đây để giúp bạn.", "Rất vui được giúp bạn!"],
    "hello": ["Xin chào! Bạn đang gặp triệu chứng gì vậy?", "Chào bạn! Mình có thể giúp gì cho bạn?"],
    "hi": ["Xin chào! Bạn đang gặp triệu chứng gì vậy?", "Chào bạn! Mình có thể giúp gì cho bạn?"]
}

def is_smalltalk(user_input):
    user_input = user_input.lower().strip()
    for key in conversation_phrases:
        if key in user_input:
            return random.choice(conversation_phrases[key])
    return None

def is_positive_response(text):
    """Kiểm tra xem câu trả lời có phải là "có" hay không"""
    text = text.lower().strip()
    positive_patterns = [
        r"\bcó\b",
        r"\bvâng\b",
        r"\bừ\b",
        r"\bđúng\b",
        r"\bphải\b",
        r"\byes\b",
        r"\by\b",
        r"\bok\b",
        r"\brồi\b",
        r"^có$",
        r"^ừ$",
        r"^ok$"
    ]
    return any(re.search(pattern, text) for pattern in positive_patterns)

def is_negative_response(text):
    """Kiểm tra xem câu trả lời có phải là "không" hay không"""
    text = text.lower().strip()
    negative_patterns = [
        r"\bkhông\b",
        r"\bko\b",
        r"\bkhong\b",
        r"\bkh\b",
        r"\bhông\b",
        r"\bno\b",
        r"\bn\b",
        r"^no$",
        r"^k$",
        r"^ko$"
    ]
    return any(re.search(pattern, text) for pattern in negative_patterns)

# ==== TIỀN XỬ LÝ ====
def extract_symptoms_from_text(user_input, symptom_index):
    try:
        clean_text = user_input.lower()
        clean_text = re.sub(r"[\.,;:!?]", " ", clean_text)  # Loại bỏ dấu câu
        # Loại bỏ ký tự đặc biệt unicode và null byte
        clean_text = re.sub(r"[^\w\s]", "", clean_text)
        found = []
        vector = [0] * len(symptom_index)

        # Hỗ trợ tìm kiếm gần đúng (fuzzy match)
        for symptom, index in symptom_index.items():
            # Tìm chính xác hoặc gần đúng (ví dụ: "mệt mỏi" ~ "mệt")
            if symptom in clean_text or any(word in clean_text for word in symptom.split()):
                found.append(symptom)
                vector[index] = 1
        return vector, list(set(found))
    except Exception as e:
        logger.error(f"Lỗi khi extract symptoms: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# ==== GIAO DIỆN HTML ====
def chat_web(request):
    return render(request, "chatbot_chatbox.html")

def get_next_symptom(remaining_symptoms, collected_symptoms):
    """Lấy triệu chứng tiếp theo cần hỏi, bỏ qua các triệu chứng đã hỏi"""
    for symptom in remaining_symptoms:
        if symptom not in collected_symptoms:
            return symptom
    return None

# ==== XỬ LÝ HỘI THOẠI ====
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
                    "response": "Xin lỗi, tôi không nhận được tin nhắn của bạn. Vui lòng thử lại."
                })
            logger.info(f"Nhận được tin nhắn: {user_input}")
        except json.JSONDecodeError as e:
            logger.error(f"Lỗi parse JSON: {str(e)}")
            return JsonResponse({
                "response": "Dữ liệu không hợp lệ. Vui lòng thử lại."
            }, status=400)

        # Khởi tạo hoặc lấy session
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

        # Kiểm tra small talk
        smalltalk_response = is_smalltalk(user_input)
        if smalltalk_response:
            return JsonResponse({"response": f"🤖 {smalltalk_response}"})

        # Xử lý câu trả lời có/không cho triệu chứng đang hỏi
        if session["remaining_symptoms"] and session["current_disease"] and session["last_asked_symptom"]:
            current_symptom = session["last_asked_symptom"]
            
            if is_positive_response(user_input):
                logger.info(f"Nhận được câu trả lời CÓ cho triệu chứng: {current_symptom}")
                if current_symptom not in session["symptoms_collected"]:
                    session["symptoms_collected"].append(current_symptom)
                session["asked_symptoms"].append(current_symptom)
            elif is_negative_response(user_input):
                logger.info(f"Nhận được câu trả lời KHÔNG cho triệu chứng: {current_symptom}")
                session["asked_symptoms"].append(current_symptom)
            else:
                return JsonResponse({
                    "response": f"🤖 Xin lỗi, tôi chưa rõ câu trả lời của bạn. Bạn có thể trả lời 'có' hoặc 'không'.\nBạn có bị **{current_symptom}** không?"
                })

            # Lấy triệu chứng tiếp theo cần hỏi
            next_symptom = get_next_symptom(session["remaining_symptoms"], session["asked_symptoms"])
            if next_symptom:
                session["last_asked_symptom"] = next_symptom
                return JsonResponse({
                    "response": f"🤖 Bạn có bị **{next_symptom}** không?"
                })
            else:
                session["last_asked_symptom"] = None

        # Extract triệu chứng từ câu nhập của người dùng
        input_vector, matched = extract_symptoms_from_text(user_input, symptom_index)
        logger.info(f"Đã tìm thấy các triệu chứng: {matched}")

        # Thêm các triệu chứng mới vào danh sách
        for sym in matched:
            if sym not in session["symptoms_collected"]:
                session["symptoms_collected"].append(sym)
                session["asked_symptoms"].append(sym)

        if not session["symptoms_collected"]:
            # Gợi ý triệu chứng phổ biến nếu không nhận diện được
            common_symptoms = list(symptom_index.keys())[:5]
            return JsonResponse({
                "response": "🤖 Tôi chưa thấy triệu chứng rõ ràng trong câu bạn nói. "
                            "Bạn có thể mô tả cụ thể hơn không? Ví dụ: 'Tôi bị sốt và đau họng'.\n"
                            f"Một số triệu chứng phổ biến: {', '.join(common_symptoms)}"
            })

        # Tạo vector từ tất cả triệu chứng đã thu thập
        vector = [0] * len(symptom_index)
        for sym in session["symptoms_collected"]:
            idx = symptom_index.get(sym)
            if idx is not None:
                vector[idx] = 1

        # Dự đoán
        try:
            x = np.array([vector], dtype=np.float32)
            predictions = model(x, training=False).numpy()[0]
            # Lấy top 3 bệnh có xác suất cao nhất
            top_indices = predictions.argsort()[-3:][::-1]
            confidences = [float(predictions[i]) for i in top_indices]
            diseases = [disease_classes[str(i)] for i in top_indices]
            logger.info(f"Top bệnh dự đoán: {diseases} với độ tin cậy: {confidences}")
        except Exception as e:
            logger.error(f"Lỗi khi dự đoán: {str(e)}")
            logger.error(traceback.format_exc())
            raise

        # Nếu chưa đủ tin cậy, hỏi thêm triệu chứng của top bệnh
        if confidences[0] < 0.6:
            for idx, disease in zip(top_indices, diseases):
                disease_specific_symptoms = disease_symptoms.get(disease, [])
                remaining = [s for s in disease_specific_symptoms if s not in session["asked_symptoms"]]
                if remaining:
                    session["current_disease"] = disease
                    session["remaining_symptoms"] = remaining
                    next_symptom = get_next_symptom(remaining, session["asked_symptoms"])
                    if next_symptom:
                        session["last_asked_symptom"] = next_symptom
                        return JsonResponse({
                            "response": f"🤖 Tôi nghi ngờ bạn có thể mắc **{disease}**.\n"
                                        f"Bạn có bị **{next_symptom}** không?"
                        })
            # Nếu không còn triệu chứng nào để hỏi
            session["current_disease"] = None
            session["remaining_symptoms"] = []
            session["last_asked_symptom"] = None
            session["asked_symptoms"] = []

        # Nếu đủ tin cậy hoặc đã hỏi hết triệu chứng
        response = f"🤖 Dựa trên thông tin bạn cung cấp, tôi dự đoán bạn có thể đang mắc **{diseases[0]}** (độ tin cậy: {confidences[0]:.2f}).\n"
        response += f"Tôi khuyên bạn nên đến cơ sở y tế để được kiểm tra kỹ hơn."

        # Reset session sau khi đưa ra kết luận
        session["symptoms_collected"] = []
        session["current_disease"] = None
        session["remaining_symptoms"] = []
        session["last_asked_symptom"] = None
        session["asked_symptoms"] = []

        return JsonResponse({"response": response})

    except Exception as e:
        logger.error(f"Lỗi không mong đợi: {str(e)}")
        logger.error(traceback.format_exc())
        return JsonResponse({
            "response": "🚨 Xin lỗi, đã có lỗi xảy ra. Vui lòng thử lại sau.",
            "error": str(e)
        }, status=500)

@csrf_exempt
@require_GET
def reset_session(request):
    keys = [
        "symptoms_collected",
        "current_disease",
        "remaining_symptoms",
        "last_asked_symptom",
        "asked_symptoms"
    ]
    for k in keys:
        if k in request.session:
            del request.session[k]
    request.session.flush()  # Xóa toàn bộ session nếu muốn
    return JsonResponse({"response": "Đã reset cuộc hội thoại!"})
