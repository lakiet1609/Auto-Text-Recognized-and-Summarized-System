from fastapi import APIRouter, Response, status, UploadFile, File
from urllib.parse import unquote

from TextSummarization.pipeline.prediction import Prediction
from database.content_crud import ContentCRUD

content_crud = ContentCRUD()
text_summarization = Prediction()

router = APIRouter(prefix='/summarization', tags = ['outputs'])

@router.post('/{text_id}/outputs/{content_id}', status_code=status.HTTP_200_OK)
async def summarize_specific_content(text_id, content_id):
    text = content_crud.select_content_by_id(text_id, content_id)
    text = text[0]
    summarization = text_summarization.predict(text)
    return summarization


@router.post('/{text_id}/outputs', status_code=status.HTTP_200_OK)
async def summarize_all_contents(text_id):
    text = content_crud.select_all_contents_by_id(text_id)
    text = text[0]
    summarization = text_summarization.predict(text)
    return summarization