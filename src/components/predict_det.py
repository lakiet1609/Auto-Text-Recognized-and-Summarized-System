# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

from copy import deepcopy
import cv2
import numpy as np
import sys
import tritonclient.grpc as grpcclient

import src.common.utility as utility
from src.common.ppocr.utils.logging import get_logger
from src.common.ppocr.data import create_operators, transform
from src.common.ppocr.postprocess import build_post_process

logger = get_logger()

class TextDetector(object):
    def __init__(self):
        postprocess_params = {}
        postprocess_params['name'] = 'DBPostProcess'
        postprocess_params["thresh"] = 0.3
        postprocess_params["box_thresh"] = 0.6
        postprocess_params["max_candidates"] = 1000
        postprocess_params["unclip_ratio"] = 1.5
        postprocess_params["use_dilation"] = False
        postprocess_params["score_mode"] = "fast"
        postprocess_params["box_type"] = 'quad'

        self.postprocess_op = build_post_process(postprocess_params)
        
        self.url = '192.168.1.10:8001'
        self.triton_client = grpcclient.InferenceServerClient(url=self.url, verbose=False)
        self.model_name = 'text_det'

    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(np.array(tmp), axis=1)
        rect[1] = tmp[np.argmin(diff)]
        rect[3] = tmp[np.argmax(diff)]
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if type(box) is list:
                box = np.array(box)
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if type(box) is list:
                box = np.array(box)
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def __call__(self, img):
        ori_img = deepcopy(img)
        
        img_triton = np.expand_dims(img, axis=0)
        img_shape = list(img_triton.shape)
 
        inputs = []
        outputs = []
        inputs.append(grpcclient.InferInput("images", img_shape, "UINT8"))
        inputs[0].set_data_from_numpy(img_triton)

        outputs.append(grpcclient.InferRequestedOutput("text_det_infer_output"))
        outputs.append(grpcclient.InferRequestedOutput("pre_det_shape_list"))

        results = self.triton_client.infer(model_name=self.model_name, inputs=inputs, outputs=outputs)
        
        text_det_infer_output = results.as_numpy("text_det_infer_output")
        pre_det_shape_list = results.as_numpy("pre_det_shape_list")
        print(text_det_infer_output.shape)

        preds = {}
        preds['maps'] = text_det_infer_output
        post_result = self.postprocess_op(preds, pre_det_shape_list)
        dt_boxes = post_result[0]['points']
        dt_boxes = self.filter_tag_det_res(dt_boxes, ori_img.shape)
        print(dt_boxes.shape)

        return dt_boxes


if __name__ == "__main__":
    args = utility.parse_args()
    img = cv2.imread(args.image_dir)
    text_detector = TextDetector()
    res = text_detector(img)
    src_im = utility.draw_text_det_res(res, img)
    cv2.imwrite(f"test_results/result.jpg", src_im)



