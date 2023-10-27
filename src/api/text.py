from fastapi import APIRouter, Response, status
from urllib.parse import unquote
from src.database.text_crud import TextCRUD

text_crud = TextCRUD()
router = APIRouter(prefix='/text', tags=['text'])

@router.get('', status_code=status.HTTP_200_OK)
async def get_all_text(skip: int, limit: int):
    text_list = text_crud.select_all_text(skip=skip, limit=limit)
    return text_list

@router.get('/{text_id}', status_code=status.HTTP_200_OK)
async def select_text_by_id(text_id):
    text_id = unquote(text_id)
    text_doc = text_crud.select_text_by_id(text_id)
    return text_doc

@router.post('', status_code=status.HTTP_201_CREATED)
async def insert_text(text_id):
    text_id, name = unquote(text_id), unquote(name)
    text_doc = text_crud.insert_person(text_id, name)
    return text_doc

@router.post('/{text_id}/name', response_class= Response, status_code= status.HTTP_204_NO_CONTENT)
async def update_name(text_id: str, name: str):
    text_id, name = unquote(text_id), unquote(name)
    text_crud.update_text_name(text_id, name)

@router.delete('/{id}', response_class= Response, status_code= status.HTTP_204_NO_CONTENT)
async def delete_text_by_ID(id: str):
    text_crud.delete_text_by_id(id)

@router.delete('', response_class= Response, status_code= status.HTTP_204_NO_CONTENT)
async def delete_all_people():
    text_crud.delete_all_text()