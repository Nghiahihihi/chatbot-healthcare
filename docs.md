# Tài liệu Hệ thống Chatbot Y tế AI

## 1. Tổng quan
Hệ thống Chatbot Y tế AI là một ứng dụng web được xây dựng bằng Django, sử dụng mô hình deep learning để chẩn đoán bệnh dựa trên triệu chứng. Hệ thống có khả năng trò chuyện tự nhiên với người dùng và đưa ra các chẩn đoán ban đầu dựa trên các triệu chứng được cung cấp.

## 2. Cấu trúc dự án
```
healthsystem/
│
├── manage.py                 # File quản lý Django
├── requirements.txt          # Danh sách thư viện cần thiết
├── db.sqlite3               # Database SQLite
│
├── healthsystem/            # Thư mục cấu hình chính
│   ├── __init__.py
│   ├── settings.py          # Cấu hình dự án
│   ├── urls.py             # Cấu hình routing chính
│   ├── wsgi.py             # Cấu hình WSGI
│   └── asgi.py             # Cấu hình ASGI
│
└── chatbot/                 # Ứng dụng chatbot
    ├── __init__.py
    ├── views.py            # Logic xử lý request/response
    ├── urls.py             # Cấu hình routing của app
    ├── ai_training.py      # Code training model AI
    │
    ├── model/              # Thư mục chứa model AI
    │   ├── medical_model.keras    # Model đã train
    │   ├── symptom_index.json     # Danh sách triệu chứng
    │   └── label_encoder.json     # Danh sách bệnh
    │
    ├── data/               # Thư mục dữ liệu
    │   └── diseases_50_vietnam.json  # Dữ liệu bệnh và triệu chứng
    │
    └── templates/          # Thư mục chứa giao diện
        └── chatbot_chatbox.html    # Template giao diện chat
```

## 3. Các thành phần chính

### 3.1. Backend (Django)
- **Framework**: Django 4.2.x
- **Database**: SQLite
- **API**: RESTful API endpoints
- **Session Management**: Django session framework

### 3.2. AI Model
- **Framework**: TensorFlow/Keras
- **Input**: Vector triệu chứng (one-hot encoding)
- **Output**: Dự đoán bệnh
- **Training**: File `ai_training.py`

### 3.3. Dữ liệu
- **Triệu chứng**: `symptom_index.json`
- **Bệnh**: `label_encoder.json`
- **Mapping bệnh-triệu chứng**: `diseases_50_vietnam.json`

### 3.4. Giao diện
- **Frontend**: HTML, CSS, JavaScript
- **Template**: `chatbot_chatbox.html`
- **Responsive Design**: Hỗ trợ đa thiết bị

## 4. Chức năng chính

### 4.1. Chatbot Y tế
- Trò chuyện tự nhiên với người dùng
- Nhận diện và xử lý triệu chứng
- Dự đoán bệnh dựa trên triệu chứng
- Hỏi thêm triệu chứng khi cần thiết

### 4.2. Xử lý ngôn ngữ tự nhiên
- Nhận diện câu chào hỏi
- Xử lý câu trả lời có/không
- Trích xuất triệu chứng từ văn bản

### 4.3. Chẩn đoán bệnh
- Phân tích triệu chứng
- Dự đoán bệnh bằng AI
- Đưa ra kết quả chẩn đoán

## 5. Luồng hoạt động

### 5.1. Khởi động hệ thống
1. Load model AI từ file
2. Load dữ liệu triệu chứng và bệnh
3. Khởi tạo Django server

### 5.2. Quy trình chat
1. **Bắt đầu hội thoại**
   - Người dùng gửi tin nhắn
   - Hệ thống kiểm tra câu chào hỏi
   - Yêu cầu triệu chứng nếu cần

2. **Xử lý triệu chứng**
   - Trích xuất triệu chứng từ văn bản
   - Lưu vào session
   - Hỏi thêm triệu chứng nếu cần

3. **Chẩn đoán**
   - Chuyển đổi triệu chứng thành vector
   - Dự đoán bệnh bằng model AI
   - Trả về kết quả cho người dùng

## 6. API Endpoints

### 6.1. Giao diện chat
- **URL**: `/api/chatbot/chatbox/`
- **Method**: GET
- **Chức năng**: Hiển thị giao diện chat

### 6.2. API dự đoán
- **URL**: `/api/chatbot/predict/`
- **Method**: POST
- **Input**: JSON với trường "message"
- **Output**: JSON với kết quả dự đoán

## 7. Cài đặt và Chạy

### 7.1. Yêu cầu hệ thống
- Python 3.8+
- Django 4.2+
- TensorFlow 2.12+
- Các thư viện khác trong requirements.txt

### 7.2. Cài đặt
```bash
# Tạo môi trường ảo
python -m venv env

# Kích hoạt môi trường
# Windows
.\env\Scripts\activate
# Linux/Mac
source env/bin/activate

# Cài đặt dependencies
pip install -r requirements.txt
```

### 7.3. Chạy server
```bash
python manage.py runserver
```

## 8. Bảo trì và Phát triển

### 8.1. Training model mới
- Chỉnh sửa file `ai_training.py`
- Chạy script training
- Lưu model mới vào thư mục `model/`

### 8.2. Thêm bệnh mới
- Cập nhật file `diseases_50_vietnam.json`
- Cập nhật `label_encoder.json`
- Training lại model nếu cần

### 8.3. Thêm triệu chứng mới
- Cập nhật file `symptom_index.json`
- Training lại model nếu cần 