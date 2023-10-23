import os
import sys
import subprocess

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import copy
import numpy as np
import src.common.utility as utility
import src.components.predict_rec as predict_rec
import src.components.predict_det as predict_det
from src.common.utility import get_rotate_crop_image


class TextSystem(object):
    def __init__(self, args):
        self.text_detector = predict_det.TextDetector()
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        self.args = args
        self.crop_image_res_index = 0
    

    def __call__(self, img):
        if img is None:
            return None
        
        dt_boxes = self.text_detector(img)

        rec_res = self.text_recognizer(dt_boxes, img)
        
        filter_boxes, filter_rec_res = [], []
        
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)
        
        return filter_boxes, filter_rec_res


def main(args):
    text_sys = TextSystem(args)
    img = cv2.imread(args.image_dir)
    img = np.expand_dims(img, axis=0)
    dt_boxes, rec_res = text_sys(img)
    return dt_boxes, rec_res


if __name__ == "__main__":
    args = utility.parse_args()
    dt_boxes, rec_res = main(args)