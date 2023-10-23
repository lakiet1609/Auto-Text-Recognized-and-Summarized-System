import triton_python_backend_utils as pb_utils
from ppocr.postprocess import build_post_process
import numpy as np
import json
import logging

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(self.model_config, "post_text_output")
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

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

        self.logger = logging

        print('Initialized...')
    
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

    def execute(self, requests):
        
        responses = []
        for request in requests:
            post_text_input = pb_utils.get_input_tensor_by_name(request, "post_text_input").as_numpy()
            pre_det_shape_list = pb_utils.get_input_tensor_by_name(request, "pre_det_shape_list").as_numpy()
            
            
            
            batch_size = pre_det_shape_list.shape[0]
            
            preds = {}
            preds['maps'] = post_text_input
            post_result = self.postprocess_op(preds, pre_det_shape_list)

            tmp_res = []
            for i in range(batch_size):
                dt_boxes = post_result[i]['points']
                ori_h, ori_w, ori_c = int(list(pre_det_shape_list[i])[0]), int(list(pre_det_shape_list[i])[1]), 3
                dt_boxes = self.filter_tag_det_res(dt_boxes, (ori_h, ori_w, ori_c))
                tmp_res.append(dt_boxes)

            max_shape = max([item.shape[0] for item in tmp_res])
            results = []
            for res in tmp_res:
                extra = max_shape - res.shape[0]
                if extra != 0:
                    dummy_array = np.ones((extra,4,2), dtype=np.float32) * -1
                    result = np.concatenate((res, dummy_array), axis=0)
                else:
                    result = res
                results.append(result)
            
            results = np.ascontiguousarray(results, dtype=self.output0_dtype)

            out_tensor_0 = pb_utils.Tensor("post_text_output", results.astype(self.output0_dtype))
            
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0])
            
            responses.append(inference_response)
        return responses

    def finalize(self):
        print('Cleaning up...')
