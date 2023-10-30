from fastapi import FastAPI, APIRouter
from api import text
from api import content
from api import recognition

router = APIRouter()
router.include_router(text.router)
router.include_router(content.router)
router.include_router(recognition.router)

app = FastAPI(title='Account', version='1.0.0')
app.include_router(router)