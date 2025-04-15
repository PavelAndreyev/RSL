import cv2
import numpy as np
import onnxruntime as ort
from collections import deque

# === Параметры ===
model_path = 'tsm_bukva.onnx'
alphabet = [f'Class_{i}' for i in range(34)]  # или замени на реальные буквы
clip_len = 8
frame_interval = 2
buffer_size = clip_len * frame_interval
input_size = (224, 224)
mean = np.array([123.675, 116.28, 103.53])
std = np.array([58.395, 57.12, 57.375])

# === Загрузка модели ===
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name

# === Веб-камера ===
cap = cv2.VideoCapture(0)
frame_buffer = deque(maxlen=buffer_size)

print("🟢 ONNX Live started. Нажмите 'q' для выхода.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("🔴 Не удалось считать кадр.")
        break

    resized = cv2.resize(frame, (340, 256))
    frame_buffer.append(resized)

    if len(frame_buffer) == buffer_size:
        sampled = [frame_buffer[i] for i in range(0, buffer_size, frame_interval)]
        processed = []

        for img in sampled:
            img = cv2.resize(img, input_size)
            img = (img - mean) / std
            img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
            processed.append(img)

        input_tensor = np.stack(processed)[None].astype(np.float32)  # (1, 8, 3, 224, 224)

        pred = session.run(None, {input_name: input_tensor})[0]
        pred = pred[0]  # shape: (34,)

        pred_label = int(np.argmax(pred))
        pred_score = float(np.max(pred))

        print(f'📍 Распознано: {alphabet[pred_label]} (score: {pred_score:.2f})')

        label_text = f'{alphabet[pred_label]}: {pred_score:.2f}'
        cv2.putText(frame, label_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("ONNX Live Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
