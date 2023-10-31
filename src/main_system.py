import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

from database.text_crud import TextCRUD
from database.content_crud import ContentCRUD
from TextSummarization.pipeline.prediction import Prediction
import cv2

if __name__ == '__main__':
    text_crud = TextCRUD()
    content_crud = ContentCRUD()
    prediction = Prediction()

    image = 'OCR/test/text1.jpg'
    img = cv2.imread(image)
    text_recognition = content_crud.select_all_contents_by_id(text_id='book')
    predicted_summarization = prediction.predict(text_recognition)
    
    print(f'Text Recognition: {text_recognition}')
    print('---------------------------------------------------------------------------------------')
    print(f'Text Summarization: {predicted_summarization}')
    

