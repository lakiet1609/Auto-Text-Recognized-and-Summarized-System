import triton_python_backend_utils as pb_utils
from ppocr.postprocess import build_post_process
import numpy as np
import json
import os
import logging

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(self.model_config, "post_rec_output")
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

        output1_config = pb_utils.get_output_config_by_name(self.model_config, "post_rec_output_score")
        self.output1_dtype = pb_utils.triton_string_to_numpy(output1_config["data_type"])

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        en_dict = 'en_dict.txt'
        en_dict_file_path = os.path.join(cur_dir, en_dict)
        
        postprocess_params = {
            'name': 'CTCLabelDecode',
            "character_dict_path": en_dict_file_path,
            "use_space_char": True
        }
        self.postprocess_op = build_post_process(postprocess_params)
        self.dictionary = (open(en_dict_file_path, 'r').read().split('\n'))
        self.dictionary_map = {x: i for i, x  in enumerate(self.dictionary)}
        self.max_length = 50

        self.logger = logging

        print('Initialized...')
    
    def encode(self, text):
        encoded_text = [-1] * self.max_length
        for i, char in enumerate(text):
            encoded_text[i] = self.dictionary_map[char]
            if i == self.max_length - 1:
                break
        
        encoded_text = np.array(encoded_text)
        return encoded_text
    
    def execute(self, requests):
        responses = []
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "post_rec_input").as_numpy()
            rec_result = self.postprocess_op(in_0)
            
            texts = [res[0] for res in rec_result]
            scores = [res[1] for res in rec_result]

            text_result = []
            score_result = []
            for text, score in zip(texts, scores):
                encoded_text = self.encode(text)
                text_result.append(encoded_text)
                score_result.append(score)
            
            text_result = np.array(text_result)
            text_result = np.ascontiguousarray(text_result, dtype = self.output0_dtype)

            score_result = np.array(score_result)
            score_result = np.ascontiguousarray(score_result, dtype = self.output1_dtype)
            
            out_tensor_0 = pb_utils.Tensor("post_rec_output", text_result.astype(self.output0_dtype))
            out_tensor_1 = pb_utils.Tensor("post_rec_output_score", score_result.astype(self.output1_dtype))

            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0, out_tensor_1])
            responses.append(inference_response)
        return responses

    def finalize(self):
        print('Cleaning up...')
