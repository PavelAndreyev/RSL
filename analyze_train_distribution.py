from collections import Counter
import matplotlib.pyplot as plt

# Путь к train.txt
train_file = 'train.txt'

# Считываем метки классов
with open(train_file, 'r') as f:
    labels = [int(line.strip().split()[1]) for line in f.readlines()]

# Подсчёт количества примеров каждого класса
label_counts = Counter(labels)

# Вывод в консоль
print("Распределение классов:")
for label, count in sorted(label_counts.items()):
    print(f"Класс {label}: {count}")

# Визуализация
plt.figure(figsize=(12, 6))
plt.bar(label_counts.keys(), label_counts.values())
plt.xlabel("Класс")
plt.ylabel("Количество видео")
plt.title("Распределение классов в train.txt")
plt.xticks(range(min(label_counts.keys()), max(label_counts.keys()) + 1))
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
