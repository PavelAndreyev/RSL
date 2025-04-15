import cv2
import os
import tempfile
import numpy as np
from mmaction.apis import inference_recognizer, init_recognizer
from mmengine.config import Config

# 🧾 Путь к конфигу и чекпойнту
config_file = '/home/ppds/PycharmProjects/gluh/tsm_mobilenetv2_bukva.py'
checkpoint_file = '/home/ppds/PycharmProjects/mmaction2/work_dirs/tsm_bukva/best_acc_top1_epoch_1.pth'
device = 'cpu'

# 🔤 Алфавит
label_map = {i: chr(1040 + i) for i in range(32)}  # А-Я
label_map[32] = 'Ё'
label_map[33] = ' '

# 🎥 Путь к видео
video_path = 'test_data/b.mp4'

# 📥 Выгружаем кадры во временную папку
tmp_dir = tempfile.TemporaryDirectory()
frame_paths = []

cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), f'Не удалось открыть видео: {video_path}'

frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_path = os.path.join(tmp_dir.name, f'{frame_id:06d}.jpg')
    cv2.imwrite(frame_path, frame)
    frame_paths.append(frame_path)
    frame_id += 1

cap.release()

# ⚙️ Загружаем модель
model = init_recognizer(config_file, checkpoint_file, device=device)

# 🧠 Предсказание
result = inference_recognizer(model, frame_paths)

# 🎯 Обработка результата
pred_scores = result.pred_score.cpu().numpy()

if len(pred_scores) == 1:
    pred_label = 0
    pred_score = float(pred_scores[0])
    print("⚠️ Модель обучена только на 1 класс. Используем индекс 0.")
else:
    pred_label = np.argmax(pred_scores)
    pred_score = float(pred_scores[pred_label])

pred_letter = label_map.get(pred_label, f"[{pred_label}]")
print(f'📍 Распознано: {pred_letter} (score: {pred_score:.2f})')

# 🧹 Удаляем временные кадры
tmp_dir.cleanup()
