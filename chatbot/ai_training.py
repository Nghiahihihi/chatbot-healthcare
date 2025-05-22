import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import pandas as pd

# Đảm bảo thư mục model tồn tại
os.makedirs("chatbot/model", exist_ok=True)

# Load dữ liệu từ CSV
df = pd.read_csv("chatbot/data/dataset.csv")

# Tạo danh sách triệu chứng duy nhất
all_symptoms = set()
for i in range(1, 18):  # Symptom_1 to Symptom_17
    col = f'Symptom_{i}'
    symptoms = df[col].dropna().unique()
    all_symptoms.update(symptoms)

# Tạo symptom index
symptom_index = {symptom: idx for idx, symptom in enumerate(sorted(all_symptoms))}

# Lưu symptom index
with open("chatbot/model/symptom_index.json", "w", encoding="utf-8") as f:
    json.dump(symptom_index, f, ensure_ascii=False, indent=2)

# Chuẩn bị dữ liệu
X = []
y = []
label_encoder = {}
label_index = 0

# Xử lý từng dòng trong dataset
for _, row in df.iterrows():
    disease = row['Disease']
    if disease not in label_encoder:
        label_encoder[disease] = label_index
        label_index += 1

    # Tạo vector triệu chứng
    vec = [0] * len(symptom_index)
    for i in range(1, 18):
        symptom = row[f'Symptom_{i}']
        if pd.notna(symptom):  # Kiểm tra nếu không phải NaN
            idx = symptom_index.get(symptom)
            if idx is not None:
                vec[idx] = 1
    
    X.append(vec)
    y.append(label_encoder[disease])

X = np.array(X, dtype=np.float32)
y = tf.keras.utils.to_categorical(y, num_classes=len(label_encoder))

# Chia dữ liệu
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây dựng model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(symptom_index),)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(len(label_encoder), activation='softmax')
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath='chatbot/model/best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Training
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# Lưu model cuối cùng
model_path = "chatbot/model/medical_model.h5"
model.save(model_path)
print(f"\nĐã lưu model cuối cùng tại: {model_path}")

# Lưu label encoder
with open("chatbot/model/label_encoder.json", "w", encoding="utf-8") as f:
    json.dump({v: k for k, v in label_encoder.items()}, f, ensure_ascii=False, indent=2)

# In kết quả training
print("\nTraining Results:")
for i, (acc, val_acc) in enumerate(zip(history.history['accuracy'], history.history['val_accuracy'])):
    print(f"Epoch {i+1}: accuracy={acc:.4f}, val_accuracy={val_acc:.4f}")
print(f"Highest validation accuracy: {max(history.history['val_accuracy']):.4f}")
print(f"Lowest validation loss: {min(history.history['val_loss']):.4f}")
