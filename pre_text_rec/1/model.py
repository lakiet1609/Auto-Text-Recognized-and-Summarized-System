import triton_python_backend_utils as pb_utils
from ppocr.data import create_operators, transform
import numpy as np
import json
import logging

class TritonPythonModel:
    def initialize(self):
        self.logger = logging

        print('Initialized...')

    def execute(self, requests):
        responses = []
        for request in requests:
            pass
            
        return responses

    def finalize(self):
        print('Cleaning up...')
