from fastapi import APIRouter, Response, status, UploadFile, File
from fastapi.responses import JSONResponse
from urllib.parse import unquote
from database.content_crud import ContentCRUD
from utility.schema import ImageValidation
import cv2 
import numpy as np


content_crud = ContentCRUD()

router = APIRouter(prefix='/text', tags=['content'])

@router.get('/{text_id}/inputs', status_code=status.HTTP_200_OK)
async def get_all_content(text_id, skip:int, limit:int):
    content_docs = content_crud.select_all_content_of_text(text_id, skip, limit)
    return content_docs

@router.get('/{text_id}/inputs/{content_id}', status_code=status.HTTP_200_OK)
async def get_content_by_id(text_id, content_id):
    content_docs = content_crud.select_content_by_id(text_id, content_id)
    return content_docs

@router.post('/{text_id}/inputs', response_model = ImageValidation, status_code=status.HTTP_201_CREATED)
async def insert_content(text_id: str, image: UploadFile = File(...)):
    content = await image.read()
    image_buffer = np.frombuffer(content, np.uint8)
    img_decode = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
    text_id = unquote(text_id)
    result = content_crud.insert_content(text_id, img_decode)
    return result

@router.delete('/{text_id}/inputs/{content_id}', response_class = Response, status_code = status.HTTP_204_NO_CONTENT)
async def delete_content_by_id(text_id, content_id):
    content_crud.delete_content_by_id(text_id, content_id)


@router.delete('/{text_id}/inputs', response_class = Response, status_code = status.HTTP_204_NO_CONTENT)
async def delete_all_content(text_id):
    content_crud.delete_all_contents(text_id)