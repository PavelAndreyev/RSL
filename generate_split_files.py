import pandas as pd
import os

# Пути
ANNOTATIONS_PATH = "/home/ppds/PycharmProjects/gluh/bukva/annotations.tsv"
VIDEOS_DIR = "bukva/videos_cropped"
OUTPUT_DIR = "/home/ppds/PycharmProjects/gluh"

# Загрузка аннотаций
df = pd.read_csv(ANNOTATIONS_PATH, sep='\t')

# Создаем словарь: символ (буква) → индекс
unique_letters = sorted(df["text"].unique())
label2idx = {letter: idx for idx, letter in enumerate(sorted(set(df['text'])))}
print(f"🔤 Всего классов: {len(label2idx)}")

# Сопоставляем буквы с числовыми метками
df["label"] = df["text"].map(label2idx)

# Теперь можно вывести макс. метку
print(f"📦 Макс. метка: {df['label'].max()}")


# Генерируем имя файла: videos_cropped/00001.mp4
df["file_name"] = df["attachment_id"].astype(str).str.zfill(5) + ".mp4"
df["file_path"] = "/home/ppds/PycharmProjects/gluh/bukva/videos_cropped/" + df["file_name"]




# Создаем строку: путь label
df["entry"] = df["file_path"] + " " + df["label"].astype(str)

# train/test деление по флагу train
df_train = df[df["train"] == 1]
df_test = df[df["train"] == 0]

# Сохраняем
df_train["entry"].to_csv(os.path.join(OUTPUT_DIR, "train.txt"), index=False, header=False)
df_test["entry"].to_csv(os.path.join(OUTPUT_DIR, "val.txt"), index=False, header=False)

print("✅ Готово: train.txt и val.txt созданы.")
