import argparse
import sys
import time
from collections import deque
from multiprocessing import Manager, Process, Value
from typing import Optional, Tuple

import onnxruntime as ort
import cv2
import numpy as np
from loguru import logger
from omegaconf import OmegaConf

from PIL import Image, ImageDraw, ImageFont

classes = [
    "no_event", "Ё", "А", "Б", "В", "Г", "Д", "Е", "Ж", "З", "И", "Й", "К", "Л",
    "М", "Н", "О", "П", "Р", "С", "Т", "У", "Ф", "Х", "Ц", "Ч", "Ш", "Щ", "Ъ",
    "Ы", "Ь", "Э", "Ю", "Я"
]

ort.set_default_logger_severity(4)
logger.add(sys.stdout, format="{level} | {message}")
logger.remove(0)

def draw_text(frame, text):
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()
    draw.text((10, 10), text, font=font, fill=(255, 255, 255))
    return np.array(img_pil)


class BaseRecognition:
    def __init__(self, model_path: str, tensors_list, prediction_list, verbose):
        self.verbose = verbose
        self.started = None
        self.output_names = None
        self.input_shape = None
        self.input_name = None
        self.session = None
        self.model_path = model_path
        self.window_size = None
        self.tensors_list = tensors_list
        self.prediction_list = prediction_list

    def clear_tensors(self):
        for _ in range(self.window_size):
            self.tensors_list.pop(0)

    def run(self):
        if self.session is None:
            self.session = ort.InferenceSession(self.model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.window_size = self.input_shape[1]
            self.output_names = [output.name for output in self.session.get_outputs()]

        if len(self.tensors_list) >= self.window_size:
            input_tensor = np.stack(self.tensors_list[:self.window_size], axis=0)[None]
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor.astype(np.float32)})[0]
            pred_idx = int(np.argmax(outputs))
            gloss = classes[pred_idx] if pred_idx < len(classes) else f"Class_{pred_idx}"
            print(f"📊 Индекс: {pred_idx}, Гласс: {gloss}, Вероятность: {np.max(outputs):.2f}")
            if gloss != self.prediction_list[-1] and len(self.prediction_list):
                if gloss != "no_event":
                    self.prediction_list.append(gloss)
            self.clear_tensors()
            if self.verbose:
                logger.info(f"--- Предсказание: {gloss}")


class Recognition(BaseRecognition):
    def __init__(self, model_path: str, tensors_list: list, prediction_list: list, verbose: bool):
        super().__init__(model_path=model_path, tensors_list=tensors_list, prediction_list=prediction_list, verbose=verbose)
        self.started = True

    def start(self):
        self.run()


class RecognitionMP(Process, BaseRecognition):
    def __init__(self, model_path: str, tensors_list, prediction_list, verbose):
        super().__init__()
        BaseRecognition.__init__(self, model_path=model_path, tensors_list=tensors_list, prediction_list=prediction_list, verbose=verbose)
        self.started = Value("i", False)

    def run(self):
        while True:
            BaseRecognition.run(self)
            self.started = True


class Runner:
    STACK_SIZE = 8

    def __init__(self, model_path: str, config, mp: bool = False, verbose: bool = False, length: int = 4):
        self.multiprocess = mp
        self.cap = cv2.VideoCapture(0)
        self.manager = Manager() if self.multiprocess else None
        self.tensors_list = self.manager.list() if self.multiprocess else []
        self.prediction_list = self.manager.list() if self.multiprocess else []
        self.prediction_list.append("---")
        self.frame_counter = 0
        self.frame_interval = config.frame_interval
        self.length = length
        self.mean = np.array(config.mean)
        self.std = np.array(config.std)

        self.recognizer = RecognitionMP(model_path, self.tensors_list, self.prediction_list, verbose) if self.multiprocess \
            else Recognition(model_path, self.tensors_list, self.prediction_list, verbose)

    def add_frame(self, image):
        self.frame_counter += 1
        if self.frame_counter == self.frame_interval:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            image = (image - self.mean) / self.std
            image = np.transpose(image, [2, 0, 1])
            self.tensors_list.append(image)
            self.frame_counter = 0

    def run(self):
        if self.multiprocess:
            self.recognizer.start()

        print("🟢 ONNX Live started. Нажмите 'q' для выхода.")
        while self.cap.isOpened():
            if self.recognizer.started:
                _, frame = self.cap.read()
                self.add_frame(frame)

                if not self.multiprocess:
                    self.recognizer.start()

                if self.prediction_list:
                    text = "  ".join(self.prediction_list)
                    frame = draw_text(frame, text)

                cv2.imshow("Live ONNX", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    if self.multiprocess:
                        self.recognizer.kill()
                    self.cap.release()
                    cv2.destroyAllWindows()
                    break


def parse_arguments(params: Optional[Tuple] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ONNX Russian Sign Language Demo")
    parser.add_argument("-p", "--config", required=True, type=str, help="Path to config.yaml")
    parser.add_argument("--mp", action="store_true", help="Enable multiprocessing")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")
    parser.add_argument("-l", "--length", type=int, default=4, help="Number of predictions in view")
    return parser.parse_args(params)


if __name__ == "__main__":
    args = parse_arguments()
    conf = OmegaConf.load(args.config)
    runner = Runner(conf.model_path, conf, args.mp, args.verbose, args.length)
    runner.run()