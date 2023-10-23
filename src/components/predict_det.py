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


    def __call__(self, imgs):
        inputs = []
        outputs = []
        
        inputs.append(grpcclient.InferInput("images", imgs.shape, "UINT8"))
        inputs[0].set_data_from_numpy(imgs)

        outputs.append(grpcclient.InferRequestedOutput("post_text_output"))

        results = self.triton_client.infer(model_name=self.model_name, inputs=inputs, outputs=outputs)
        
        post_text_output = results.as_numpy("post_text_output")

        return post_text_output


if __name__ == "__main__":
    args = utility.parse_args()
    img = cv2.imread(args.image_dir)
    img = np.expand_dims(img, axis=0)
    text_detector = TextDetector()
    res = text_detector(img)
    src_im = utility.draw_text_det_res(res, img)
    cv2.imwrite(f"test_results/result.jpg", src_im)



