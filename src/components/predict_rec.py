import os
import sys
from PIL import Image
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import numpy as np
import paddle
from src.common.ppocr.postprocess import build_post_process
import tritonclient.grpc as grpcclient


class TextRecognizer(object):
    def __init__(self, args):
        self.dictionary = (open('common/ppocr/utils/en_dict.txt', 'r').read().split('\n'))

        self.url = '192.168.1.10:8001'
        self.triton_client = grpcclient.InferenceServerClient(url=self.url, verbose=False)
        self.model_name = 'text_rec'

    def decode(self, encoded_text):
        text = ''
        for item in encoded_text:
            if item == -1:
                continue
            char = self.dictionary[int(item)]
            text += char
        return text
    

    def __call__(self, dt_boxes, ori_img):
        dt_boxes_shape = list(dt_boxes.shape)
        images_shape = list(ori_img.shape)
        inputs = []
        outputs = []
        
        inputs.append(grpcclient.InferInput("dt_boxes", dt_boxes_shape, "FP32"))
        inputs.append(grpcclient.InferInput("images", images_shape, "UINT8"))
        inputs[0].set_data_from_numpy(dt_boxes)
        inputs[1].set_data_from_numpy(ori_img)

        outputs.append(grpcclient.InferRequestedOutput("infer_text_rec_output"))

        results = self.triton_client.infer(model_name=self.model_name, inputs=inputs, outputs=outputs)
        
        infer_text_rec_output = results.as_numpy("infer_text_rec_output")
        
        infer_text_rec_output_shape = list(infer_text_rec_output.shape)
        inputs = []
        outputs = []
        
        inputs.append(grpcclient.InferInput("post_rec_input", infer_text_rec_output_shape, "FP32"))
        inputs[0].set_data_from_numpy(infer_text_rec_output)

        outputs.append(grpcclient.InferRequestedOutput("post_rec_output"))
        outputs.append(grpcclient.InferRequestedOutput("post_rec_output_score"))

        results = self.triton_client.infer(model_name='post_text_rec', inputs=inputs, outputs=outputs)
        
        texts = results.as_numpy("post_rec_output")
        scores = results.as_numpy("post_rec_output_score")
        
        for text in texts:
            ori_text = self.decode(text)
            print(ori_text)

        return texts, scores




