from fastapi import APIRouter, Response, status, UploadFile, File
from urllib.parse import unquote
import cv2
import numpy as np
from TextSummarization.pipeline.prediction import Prediction
from database.content_crud import ContentCRUD

content_crud = ContentCRUD()
text_summarization = Prediction()

router = APIRouter(prefix='/prediction', tags = ['text'])

@router.post('', status_code=status.HTTP_200_OK)
async def summarize_specific_content(text_id, content_id):
    text = content_crud.select_content_by_id(text_id, content_id)
    text = text[0]
    summarization = text_summarization.predict(text)
    return summarization


@router.post('', status_code=status.HTTP_200_OK)
async def summarize_all_content(text_id):
    text = content_crud.select_all_content_by_id(text_id)
    text = text[0]
    summarization = text_summarization.predict(text)
    return summarization