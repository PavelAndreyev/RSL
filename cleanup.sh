#!/bin/bash

# 1. Проверка, что ты в Git-репозитории
if [ ! -d .git ]; then
  echo "❌ Здесь нет .git — вы не в корне Git-репозитория"
  exit 1
fi

# 2. Удаление бинарных файлов из истории
echo "🧹 Удаляем mp4, onnx, pth из истории..."
java -jar ~/bfg.jar --delete-files '*.{mp4,onnx,pth}' .

# 3. Очистка reflog и удаление мусора
echo "🧼 Очистка reflog и удаление мусора..."
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# 4. Проверка размера
echo "📦 Текущий размер .git:"
du -sh .git

# 5. Форсированный push
echo "🚀 Выполняем форсированный push..."
git push --force origin v1

echo "✅ Готово!"
