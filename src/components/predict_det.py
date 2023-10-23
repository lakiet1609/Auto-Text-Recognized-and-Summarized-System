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


from src.common.ppocr.postprocess import build_post_process
from src.common import utility

class TextDetector(object):
    def __init__(self):   
        self.url = '192.168.1.10:8001'
        self.triton_client = grpcclient.InferenceServerClient(url=self.url, verbose=False)
        self.model_name = 'text_det'


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

        inputs = []
        outputs = []
        inputs.append(grpcclient.InferInput("post_text_input", list(text_det_infer_output.shape), "FP32"))
        inputs.append(grpcclient.InferInput("pre_det_shape_list", list(pre_det_shape_list.shape), "FP32"))
        inputs[0].set_data_from_numpy(text_det_infer_output)
        inputs[1].set_data_from_numpy(pre_det_shape_list)

        outputs.append(grpcclient.InferRequestedOutput("post_text_output"))

        results = self.triton_client.infer(model_name='post_text_det', inputs=inputs, outputs=outputs)
        
        post_text_output = results.as_numpy("post_text_output")

        return post_text_output


if __name__ == "__main__":
    args = utility.parse_args()
    img = cv2.imread(args.image_dir)
    text_detector = TextDetector()
    res = text_detector(img)
    src_im = utility.draw_text_det_res(res, img)
    cv2.imwrite(f"test_results/result.jpg", src_im)



