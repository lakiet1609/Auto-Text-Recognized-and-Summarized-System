import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

import cv2
import numpy as np
import src.common.utility as utility
import src.components.predict_rec as predict_rec
import src.components.predict_det as predict_det


class TextSystem(object):
    def __init__(self, args):
        self.text_detector = predict_det.TextDetector()
        self.text_recognizer = predict_rec.TextRecognizer()
        self.args = args
    
    def __call__(self, img):
        if img is None:
            return None
        
        dt_boxes = self.text_detector(img)
        text, score = self.text_recognizer(dt_boxes, img)

def main(args):
    text_sys = TextSystem(args)
    img = cv2.imread(args.image_dir)
    img = np.expand_dims(img, axis=0)
    text_sys(img)


if __name__ == "__main__":
    args = utility.parse_args()
    main(args)