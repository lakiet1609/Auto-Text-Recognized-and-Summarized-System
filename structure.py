import os
from pathlib import Path

project1_name = 'OCR'
project2_name = 'TextSummarization'

list_files = [
    f'src/{project1_name}/__init__.py',
    f'src/{project1_name}/components/__init__.py',
    f'src/{project1_name}/models/__init__.py',
    f'src/{project1_name}/pipeline/__init__.py',
    f'src/{project1_name}/common/__init__.py',
    
    f'src/{project2_name}/__init__.py',
    f'src/{project2_name}/components/__init__.py',
    f'src/{project2_name}/constant/__init__.py',
    f'src/{project2_name}/entity/__init__.py',
    f'src/{project2_name}/config/__init__.py',
    f'src/{project2_name}/config/configuration.py',
    f'src/{project2_name}/pipeline/__init__.py',
    f'src/{project2_name}/utils/__init__.py',
    f'src/{project2_name}/utils/common.py',
    f'src/{project2_name}/artifacts/.gitkeep',
    f'src/{project2_name}/files/config.yaml',
    f'src/{project2_name}/files/params.yaml',
    f'src/{project2_name}/main.py',
    f'src/{project2_name}/app.py',
    'requirements.txt',
    'triton_models/.gitkeep',
    
]

for file in list_files:
    file = Path(file)
    file_dir, file_name = os.path.split(file)

    if file_dir != '':
        os.makedirs(file_dir, exist_ok=True)

    if (not os.path.exists(file)) or (os.path.getsize(file) == 0):
        with open(file, 'w') as f:
            pass