import triton_python_backend_utils as pb_utils
from ppocr.postprocess import build_post_process
import numpy as np
import json
import logging

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(self.model_config, "post_rec_output")
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

        output0_config = pb_utils.get_output_config_by_name(self.model_config, "post_rec_output_score")
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

        self.logger = logging

        print('Initialized...')
    

    def execute(self, requests):
        responses = []
        for request in requests:
         pass
        return responses

    def finalize(self):
        print('Cleaning up...')
