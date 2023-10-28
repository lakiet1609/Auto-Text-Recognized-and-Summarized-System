import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

import cv2
import numpy as np
import OCR.common.utility as utility
import OCR.components.predict_rec as predict_rec
import OCR.components.predict_det as predict_det


class TextSystem(object):
    def __init__(self):
        self.text_detector = predict_det.TextDetector()
        self.text_recognizer = predict_rec.TextRecognizer()
    
    def __call__(self, img):
        img = cv2.imread(img)
        img = np.expand_dims(img, axis=0)
        dt_boxes = self.text_detector(img)
        texts, scores = self.text_recognizer(dt_boxes, img)
        return texts, scores

if __name__ == "__main__":
    text_sys = TextSystem()
    img = 'OCR/test/text1.jpg'
    texts, scores = text_sys(img)
    print(texts)
