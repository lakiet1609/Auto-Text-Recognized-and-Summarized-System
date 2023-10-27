import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

from database.text_crud import TextCRUD
from database.content_crud import ContentCRUD

if __name__ == '__main__':
    text_crud = TextCRUD()
    text_crud.delete_all_text()
    # content_crud = ContentCRUD()
    # content_crud.insert_content(text_id='kiet', image='OCR/test/mckinlay-820.jpg')

