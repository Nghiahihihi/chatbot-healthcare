import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# Đảm bảo thư mục model tồn tại
os.makedirs("chatbot/model", exist_ok=True)

# Load dữ liệu
with open("chatbot/model/symptom_index.json", "r", encoding="utf-8") as f:
    symptom_index = json.load(f)

with open("chatbot/data/diseases_50_vietnam.json", "r", encoding="utf-8") as f:
    disease_data = json.load(f)

# Chuẩn bị dữ liệu
X = []
y = []
label_encoder = {}
label_index = 0

for entry in disease_data:
    disease = entry["name"]
    if disease not in label_encoder:
        label_encoder[disease] = label_index
        label_index += 1

    for symptom_list in [entry["symptoms"]]:
        vec = [0] * len(symptom_index)
        for symptom in symptom_list:
            idx = symptom_index.get(symptom)
            if idx is not None:
                vec[idx] = 1
        X.append(vec)
        y.append(label_encoder[disease])

X = np.array(X, dtype=np.float32)
y = tf.keras.utils.to_categorical(y, num_classes=len(label_encoder))

# Chia dữ liệu
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây dựng model (nâng cấp)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(symptom_index),)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(label_encoder), activation='softmax')
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks (thêm ModelCheckpoint)
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath='chatbot/model/best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Training
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=300,
    batch_size=16,
    callbacks=callbacks,
    verbose=1
)

# Lưu model cuối cùng
model_path = "chatbot/model/medical_model.keras"
model.save(model_path, save_format='keras')
print(f"\nĐã lưu model cuối cùng tại: {model_path}")

# Lưu label encoder
with open("chatbot/model/label_encoder.json", "w", encoding="utf-8") as f:
    json.dump({v: k for k, v in label_encoder.items()}, f, ensure_ascii=False, indent=2)

# In kết quả training
print("\nKết quả training:")
for i, (acc, val_acc) in enumerate(zip(history.history['accuracy'], history.history['val_accuracy'])):
    print(f"Epoch {i+1}: accuracy={acc:.4f}, val_accuracy={val_acc:.4f}")
print(f"Độ chính xác cao nhất trên tập validation: {max(history.history['val_accuracy']):.4f}")
print(f"Loss thấp nhất trên tập validation: {min(history.history['val_loss']):.4f}")
