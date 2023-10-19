import triton_python_backend_utils as pb_utils
from ppocr.data import create_operators, transform
import numpy as np
import json
import logging

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(self.model_config, "pre_det_image")
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

        output1_config = pb_utils.get_output_config_by_name(self.model_config, "pre_det_shape_list")
        self.output1_dtype = pb_utils.triton_string_to_numpy(output1_config["data_type"])

        self.logger = logging

        pre_process_list = [{
            'DetResizeForTest': {
                'limit_side_len': 960,
                'limit_type': 'max',
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

        self.preprocess_op = create_operators(pre_process_list)

        print('Initialized...')

    def execute(self, requests):
        responses = []
        for request in requests:
            images = pb_utils.get_input_tensor_by_name(request, "images").as_numpy() # B,H,W,C
            image_results = []
            shape_list_results = []
            
            for image in images:
                data = {'image': image}
                data = transform(data, self.preprocess_op)
                image, shape_list = data
                if image is None:
                    return None
                
                image_results.append(image)
                shape_list_results.append(shape_list)
            
            image_results = np.array(image_results)
            image_results = np.ascontiguousarray(image_results, dtype=self.output0_dtype)
            out_tensor_0 = pb_utils.Tensor("pre_det_image", image_results.astype(self.output0_dtype))

            shape_list_results = np.array(shape_list_results)
            shape_list_results = np.ascontiguousarray(shape_list_results, dtype=self.output1_dtype)
            out_tensor_1 = pb_utils.Tensor("pre_det_shape_list", shape_list_results.astype(self.output1_dtype))

            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0, out_tensor_1])
            responses.append(inference_response)
            
        return responses

    def finalize(self):
        print('Cleaning up...')
