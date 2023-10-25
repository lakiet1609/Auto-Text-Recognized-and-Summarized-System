import os
from pathlib import Path    
import yaml

def read_yaml(path: str):
    with open(path) as file:
        content = yaml.safe_load(file)
    return content


def create_directory(path: str):
    os.makedirs(path, exist_ok=True)

