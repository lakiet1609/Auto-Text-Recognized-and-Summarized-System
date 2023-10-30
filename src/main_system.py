import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

from database.text_crud import TextCRUD
from database.content_crud import ContentCRUD
from TextSummarization.pipeline.prediction import Prediction

if __name__ == '__main__':
    text_crud = TextCRUD()
    content_crud = ContentCRUD()
    prediction = Prediction()

