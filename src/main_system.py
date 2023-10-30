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
    # texts = text_crud.insert_text(id='book', name='rabbit')
    # content_crud.insert_content(text_id='book', image='OCR/test/text3.jpg')
    # content_crud.delete_all_contents(text_id='book')
    # content_crud.delete_content_by_id(text_id='book', content_id='dc0fd093046947e5a93deb5821a3bf41')
    a = content_crud.select_content_by_id(text_id='book', content_id='6628036270df4803875db34831df1f27')
    print(a)
    # print(a['content'][0][0])

    # result = prediction.predict(a)
    # print(result)
