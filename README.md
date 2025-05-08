# Hướng dẫn chạy dự án Chatbot Y tế AI

## 1. Yêu cầu hệ thống
- Python 3.8+
- pip (Python package manager)
- (Khuyến nghị) Sử dụng môi trường ảo (venv)

## 2. Cài đặt môi trường

### Bước 1: Tạo và kích hoạt môi trường ảo
```bash
python -m venv env
# Windows:
.\env\Scripts\activate
# Linux/Mac:
source env/bin/activate
```

### Bước 2: Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```

## 3. Chuẩn bị dữ liệu
- Đảm bảo các file dữ liệu đã có trong các thư mục:
  - `chatbot/data/diseases_50_vietnam.json`
  - `chatbot/model/symptom_index.json` (có thể tạo lại bằng script nếu cần)
  - `chatbot/model/label_encoder.json` (tự động sinh khi training)

## 4. Training lại mô hình AI (nếu muốn)
Nếu bạn đã cập nhật dữ liệu hoặc muốn cải thiện độ chính xác:
```bash
cd chatbot
python ai_training.py
cd ..
```
Sau khi training, file model mới sẽ được lưu tại `chatbot/model/medical_model.keras`.

## 5. Chạy server Django
```bash
python manage.py runserver
```

## 6. Sử dụng chatbot
- Truy cập giao diện chat tại: [http://127.0.0.1:8000/api/chatbot/chatbox/](http://127.0.0.1:8000/api/chatbot/chatbox/)
- Nhập triệu chứng hoặc câu hỏi về sức khỏe để được tư vấn.
- Bấm "Kết thúc hội thoại" để làm mới cuộc chat và reset bộ nhớ.

## 7. Một số lưu ý
- Nếu gặp lỗi thiếu file `symptom_index.json`, hãy tạo lại bằng script Python dựa trên dữ liệu bệnh.
- Nếu muốn bổ sung bệnh/triệu chứng, hãy cập nhật file `diseases_50_vietnam.json` rồi training lại model.
- Nếu chatbot dự đoán sai, hãy kiểm tra lại dữ liệu và tăng số lượng mẫu cho các bệnh phổ biến.

## 8. Liên hệ & đóng góp
- Nếu có thắc mắc hoặc muốn đóng góp, hãy liên hệ nhóm phát triển hoặc tạo issue trên repository. 