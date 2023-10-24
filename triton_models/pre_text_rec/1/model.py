import triton_python_backend_utils as pb_utils
import cv2
import math
import numpy as np
import json
from copy import deepcopy
import logging

class TritonPythonModel:
    def initialize(self, args):
        self.logger = logging
        self.model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(self.model_config, "pre_rec_output")
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

        self.rec_image_shape = [3,48,320]

        print('Initialized...')
           
    def get_rotate_crop_image(self, img, points):
        assert len(points) == 4, "shape of points must be 4*2"
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                            [img_crop_width, img_crop_height],
                            [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape
        assert imgC == img.shape[2]
        imgW = int((imgH * max_wh_ratio))
    
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im
    
    def sorted_boxes(self, dt_boxes):
        num_boxes = dt_boxes.shape[0]
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)

        for i in range(num_boxes - 1):
            for j in range(i, -1, -1):
                if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                        (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                    tmp = _boxes[j]
                    _boxes[j] = _boxes[j + 1]
                    _boxes[j + 1] = tmp
                else:
                    break
        return _boxes

    def execute(self, requests):
        responses = []
        for request in requests:
            dt_boxes = pb_utils.get_input_tensor_by_name(request, "dt_boxes").as_numpy()
            if dt_boxes is None:
                return None

            images = pb_utils.get_input_tensor_by_name(request, "images").as_numpy()
            images = np.squeeze(images, axis=0)

            dt_boxes = self.sorted_boxes(dt_boxes)
            dt_boxes = dt_boxes[0]

            img_list = []
            for bno in range(len(dt_boxes)):
                tmp_box = deepcopy(dt_boxes[bno])
                img_crop = self.get_rotate_crop_image(images, tmp_box)
                img_list.append(img_crop)
            
            width_list = []
            for cr_img in img_list:
                width_list.append(cr_img.shape[1] / float(cr_img.shape[0]))

            max_wh_ratio = max(width_list)
            imgC, imgH, imgW = self.rec_image_shape[:3]
            setting_max_wh_ratio = imgW / imgH
            max_wh_ratio = max(max_wh_ratio, setting_max_wh_ratio)

            norm_img_batch = []
            for img in img_list:
                norm_img = self.resize_norm_img(img,max_wh_ratio)
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)

            norm_img_batch = np.concatenate(norm_img_batch)

            norm_img_batch = np.ascontiguousarray(norm_img_batch, dtype=self.output0_dtype)

            out_tensor_0 = pb_utils.Tensor("pre_rec_output", norm_img_batch.astype(self.output0_dtype))

            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0])
            
            responses.append(inference_response)

        return responses

    def finalize(self):
        print('Cleaning up...')
