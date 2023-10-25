import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

from src.TextSummarization.utils.common import read_yaml, create_directory
from src.TextSummarization.constant import *

content = read_yaml(CONFIG_FILE_PATH)
data = content['data_validation']

print(data['ALL_REQUIRED_FILES'])