import cv2
import numpy as np
from collections import deque
from mmaction.apis import init_recognizer, inference_recognizer
from mmengine.config import Config

# === Параметры ===
config_file = '/home/ppds/PycharmProjects/gluh/tsm_mobilenetv2_bukva.py'
checkpoint_file = '/home/ppds/PycharmProjects/mmaction2/work_dirs/tsm_bukva/epoch_50.pth'
device = 'cpu'
clip_len = 8
frame_interval = 2
buffer_size = clip_len * frame_interval

# === Инициализация модели ===
config = Config.fromfile(config_file)
model = init_recognizer(config, checkpoint_file, device=device)

# === Алфавит (из train.txt)
alphabet = list("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ ")


# === Кадровый буфер ===
frame_buffer = deque(maxlen=buffer_size)

# === Веб-камера ===
cap = cv2.VideoCapture(0)
print("🟢 Запущено. Нажмите 'q' для выхода.")

# === Кадровый буфер ===
frame_buffer = deque(maxlen=buffer_size)

# Инициализируем переменные для отображения, чтобы не падать до первого инференса
pred_score = 0.0
pred_label = 0


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("🔴 Не удалось считать кадр с камеры.")
        break

    resized = cv2.resize(frame, (340, 256))
    frame_buffer.append(resized)

    if len(frame_buffer) == buffer_size:
        sampled_frames = [frame_buffer[i] for i in range(0, buffer_size, frame_interval)]

        input_data = dict(
            imgs=sampled_frames,
            total_frames=len(sampled_frames),
            label=-1,
            modality='RGB'
        )

        # Предсказание
        result = inference_recognizer(model, input_data)
        pred_scores = result.pred_score.cpu().numpy().flatten()
        pred_label = int(np.argmax(pred_scores))
        pred_score = float(np.max(pred_scores))

        print(f'📍 Распознано: {alphabet[pred_label]} (score: {pred_score:.2f})')
        print("🔎 Топ-5:")
        top5_indices = np.argsort(pred_scores)[-5:][::-1]
        for i in top5_indices:
            print(f'   {alphabet[i]}: {pred_scores[i]:.2f}')

    # === Отображение предсказания на кадре ===
    display_text = f'{alphabet[pred_label]} ({pred_score:.2f})'
    (text_width, text_height), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 4)
    x, y = 20, 60

    # Фон
    cv2.rectangle(frame, (x - 10, y - text_height - 10), (x + text_width + 10, y + 10), (0, 0, 0), -1)
    # Текст
    cv2.putText(frame, display_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

    # Отображение окна
    cv2.imshow('Live Gesture Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Информация по завершению
print(f'pred_scores shape: {pred_scores.shape}')
print(f'alphabet length: {len(alphabet)}')

cap.release()
cv2.destroyAllWindows()
