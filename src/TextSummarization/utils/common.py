import os
from ensure import ensure_annotations
from pathlib import Path    
import yaml

@ensure_annotations
def read_yaml(path: str):
    with open(path) as file:
        content = yaml.safe_load(file)
    return content


@ensure_annotations
def create_directory(paths: str):
    for path in paths:
        os.makedirs(path, exist_ok=True)

