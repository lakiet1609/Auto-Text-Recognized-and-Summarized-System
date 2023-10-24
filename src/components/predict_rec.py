import os
import sys
from PIL import Image
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import numpy as np
import math
import time
import traceback
import paddle
from copy import deepcopy
from src.common.ppocr.postprocess import build_post_process
import tritonclient.grpc as grpcclient


class TextRecognizer(object):
    def __init__(self, args):
        self.rec_batch_num = 6
        postprocess_params = {
            'name': 'CTCLabelDecode',
            "character_dict_path": args.rec_char_dict_path,
            "use_space_char": args.use_space_char
        }
        self.postprocess_op = build_post_process(postprocess_params)

        self.url = '192.168.1.10:8001'
        self.triton_client = grpcclient.InferenceServerClient(url=self.url, verbose=False)
        self.model_name = 'text_rec'
    

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
        preds = infer_text_rec_output

        rec_result = self.postprocess_op(preds)
        print(type(rec_result[0][0]))
        return rec_result




