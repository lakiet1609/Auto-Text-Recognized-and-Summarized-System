import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))


import tritonclient.grpc as grpcclient


class TextRecognizer(object):
    def __init__(self):
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
        inputs = []
        outputs = []
        
        inputs.append(grpcclient.InferInput("dt_boxes", list(dt_boxes.shape), "FP32"))
        inputs.append(grpcclient.InferInput("images", list(ori_img.shape), "UINT8"))
        inputs[0].set_data_from_numpy(dt_boxes)
        inputs[1].set_data_from_numpy(ori_img)

        outputs.append(grpcclient.InferRequestedOutput("post_rec_output"))
        outputs.append(grpcclient.InferRequestedOutput("post_rec_output_score"))

        results = self.triton_client.infer(model_name=self.model_name, inputs=inputs, outputs=outputs)
        
        texts = results.as_numpy("post_rec_output")
        scores = results.as_numpy("post_rec_output_score")
        
        for text, score in zip(texts,scores):
            ori_text = self.decode(text)
            print(ori_text, score)

        return texts, scores




