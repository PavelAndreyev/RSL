import re
import matplotlib.pyplot as plt

# Укажи путь к лог-файлу
log_file = '/home/ppds/PycharmProjects/mmaction2/work_dirs/tsm_bukva/20250408_*.log'

# Автоматически находит файл, если имя начинается с даты
import glob
log_path = glob.glob(log_file)
if not log_path:
    print("❌ Лог-файл не найден")
    exit()
log_path = log_path[0]

# Парсим строки
losses, top1_accs, top5_accs, steps = [], [], [], []

with open(log_path, 'r') as f:
    for line in f:
        if 'Epoch(train)' in line and 'loss:' in line:
            step_match = re.search(r'\[(\d+)\]\[\s*(\d+)/\d+\]', line)
            loss_match = re.search(r'loss: ([0-9.]+)', line)
            top1_match = re.search(r'top1_acc: ([0-9.]+)', line)
            top5_match = re.search(r'top5_acc: ([0-9.]+)', line)

            if step_match and loss_match and top1_match and top5_match:
                epoch = int(step_match.group(1))
                step = int(step_match.group(2))
                global_step = epoch + step / 1000  # для гладкости графика
                steps.append(global_step)
                losses.append(float(loss_match.group(1)))
                top1_accs.append(float(top1_match.group(1)))
                top5_accs.append(float(top5_match.group(1)))

# График
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(steps, losses, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(steps, top1_accs, label='Top-1 Acc')
plt.plot(steps, top5_accs, label='Top-5 Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
