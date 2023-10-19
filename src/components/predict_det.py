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
    def __init__(self, args):
        self.args = args
        pre_process_list = [{
            'DetResizeForTest': {
                'limit_side_len': args.det_limit_side_len,
                'limit_type': args.det_limit_type,
            }
        }, {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': ['image', 'shape']
            }
        }]
        
        postprocess_params = {}

        postprocess_params['name'] = 'DBPostProcess'
        postprocess_params["thresh"] = args.det_db_thresh
        postprocess_params["box_thresh"] = args.det_db_box_thresh
        postprocess_params["max_candidates"] = 1000
        postprocess_params["unclip_ratio"] = args.det_db_unclip_ratio
        postprocess_params["use_dilation"] = args.use_dilation
        postprocess_params["score_mode"] = args.det_db_score_mode
        postprocess_params["box_type"] = args.det_box_type

        self.preprocess_op = create_operators(pre_process_list)
        self.postprocess_op = build_post_process(postprocess_params)
        self.predictor, self.input_tensor, self.output_tensors, self.config = utility.create_predictor(args, 'det', logger)
        
        self.url = '192.168.1.10:8001'
        self.triton_client = grpcclient.InferenceServerClient(url=self.url, verbose=False)
        self.model_name = 'infer_text_det'

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

        outputs.append(grpcclient.InferRequestedOutput("pre_det_image"))
        outputs.append(grpcclient.InferRequestedOutput("pre_det_shape_list"))

        results = self.triton_client.infer(model_name='pre_text_det', inputs=inputs, outputs=outputs)
        
        pre_det_image = results.as_numpy("pre_det_image")
        pre_det_shape_list = results.as_numpy("pre_det_shape_list")
        # print(pre_det_shape_list.shape)
        
        # # H, W, C 
        # ori_im = img.copy()
        # data = {'image': img}
        # data = transform(data, self.preprocess_op)
        # img, shape_list = data
        
        # if img is None:
        #     return None, 0
        
        # img = np.expand_dims(img, axis=0) #B, C, H, W
        # shape_list = np.expand_dims(shape_list, axis=0)
        
        # print(img.shape)
        # print(shape_list)
        
        pre_img_shape = list(pre_det_image.shape)
        inputs = []
        outputs = []
        
        inputs.append(grpcclient.InferInput("x", pre_img_shape, "FP32"))
        inputs[0].set_data_from_numpy(pre_det_image)

        outputs.append(grpcclient.InferRequestedOutput("sigmoid_0.tmp_0"))

        results = self.triton_client.infer(model_name=self.model_name, inputs=inputs, outputs=outputs)
        output_data = results.as_numpy("sigmoid_0.tmp_0")

        preds = {}
        preds['maps'] = output_data
        post_result = self.postprocess_op(preds, pre_det_shape_list)
        dt_boxes = post_result[0]['points']
        dt_boxes = self.filter_tag_det_res(dt_boxes, ori_img.shape)

        return dt_boxes


if __name__ == "__main__":
    args = utility.parse_args()
    img = cv2.imread(args.image_dir)
    text_detector = TextDetector(args)
    res = text_detector(img)
    src_im = utility.draw_text_det_res(res, img)
    cv2.imwrite(f"test_results/result.jpg", src_im)



