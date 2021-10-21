from fastapi import FastAPI
from fastapi import Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordBearer
from requests.models import Response
from starlette.routing import Host
from starlette.config import Config
import uvicorn
from pydantic import BaseModel
import sys
sys.path.append('./src/')
from src.infer import Process
import time, io
import base64
from PIL import Image

infer_model = Process()
app = FastAPI()

class InputChangeCloth(BaseModel):
    image_human: str
    image_cloth: str

@app.post("/cloth")
async def change_cloth(data: InputChangeCloth):
    try:
        image_human = Image.open(io.BytesIO(base64.b64decode(data.image_human)))
        image_cloth = Image.open(io.BytesIO(base64.b64decode(data.image_cloth)))
    except FileNotFoundError:
        response = {
            "status": False,
            "message": "File is empty!"
        }
        return response
    start = time.time()
    result = infer_model._predict(image_human, image_cloth)
    image = base64.b64decode((infer_model._img_encode(result)).decode())
    img = Image.open(io.BytesIO(image))
    imagePath = ("./result/result.jpeg")
    img.save(imagePath, 'jpeg')
    response = {
        "status": True,
        "input" : {
            "image_human": data.image_human,
            "image_cloth": data.image_cloth
        },
        "result": (infer_model._img_encode(result)).decode(),
        "time_infer": time.time() - start,
        "timestamp": time.time()
    }
    return response

if __name__=="__main__":
    uvicorn.run(app, host='0.0.0.0', port=8888)