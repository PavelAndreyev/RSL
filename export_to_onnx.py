import torch
from mmaction.apis import init_recognizer
from mmengine.config import Config
import torch.nn as nn

# === Параметры ===
config_file = '/home/ppds/PycharmProjects/gluh/tsm_mobilenetv2_bukva.py'
checkpoint_file = '/home/ppds/PycharmProjects/mmaction2/work_dirs/tsm_bukva/epoch_50.pth'
output_file = '/home/ppds/PycharmProjects/gluh/tsm_bukva.onnx'
device = 'cpu'
num_segs = 8

# === Инициализация модели ===
config = Config.fromfile(config_file)
model = init_recognizer(config, checkpoint_file, device=device)
model.eval()

# === Обёртка, добавляющая num_segs внутрь forward ===
class WrappedRecognizer(nn.Module):
    def __init__(self, recognizer, num_segs):
        super().__init__()
        self.recognizer = recognizer
        self.num_segs = num_segs

    def forward(self, x):
        # x: [1, 8, 3, 224, 224] -> [batch_size * num_segs, 3, 224, 224]
        b, t, c, h, w = x.shape
        x = x.view(-1, c, h, w)
        feat = self.recognizer.backbone(x)

        # Защита от отсутствия neck
        if hasattr(self.recognizer, 'neck') and self.recognizer.neck is not None:
            feat = self.recognizer.neck(feat)

        out = self.recognizer.cls_head(feat, self.num_segs)
        return out


# === Оборачиваем и экспортируем ===
wrapped_model = WrappedRecognizer(model, num_segs=num_segs)

dummy_input = torch.randn(1, num_segs, 3, 224, 224)

torch.onnx.export(
    wrapped_model,
    dummy_input,
    output_file,
    input_names=['input'],
    output_names=['logits'],
    dynamic_axes={'input': {0: 'batch_size'}, 'logits': {0: 'batch_size'}},
    opset_version=11
)

print(f"✅ ONNX-модель экспортирована в {output_file}")
