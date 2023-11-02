import torch
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi import FastAPI,Request
from contextlib import asynccontextmanager
from loguru import logger
import time
import os
import yaml
from pydantic import BaseModel, Field
# #日志-------------------------------------
# from logger import setup_logger
# log_file = 'server.log' 
# logger = setup_logger(log_file,"server")

app = FastAPI()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# def load_config(file_path):
#     with open(file_path, "r") as file:
#         config = yaml.safe_load(file)
#     return config

# cfg = load_config("config.yaml")

def _gc(forced: bool = False):
    global args
    if args.disable_gc and not forced:
        return

    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@asynccontextmanager
async def lifespan(app: FastAPI):  # collects GPU memory
    yield
    _gc(forced=True)


app = FastAPI(lifespan=lifespan)


app.add_middleware(
   CORSMiddleware, # 中间件类，用于处理CORS
   allow_origins=["*"], # 允许来自任何源的访问
   allow_credentials=True, #表示允许发送凭据
   allow_methods=["*"], #允许所有的 HTTP 方法
   allow_headers=["*"], #允许所有的请求标头
)

class TTsRequest(BaseModel):
    text: str
    language : str = "简体中文"
    speed : float = 1
    speaker:str = "kt"

from kuontts.offline import OfflineTTS
tts =  OfflineTTS()
@app.post("/tts/convert")
async def convert_text(request:TTsRequest):
    logger.info("接收到转化任务：{}".format(request))
    result,audio = tts.run(text=request.text,speaker=request.speaker,language=request.language,speed=request.speed)
    if result == "Success":
        audio_list = audio[1].tolist() 
        logger.info("转化成功：{}".format(result))
        return {"result": result, "rate": audio[0], "audio": audio_list}
    else:
        logger.info("转化失败：{}".format(audio))
        return {"result": result,"message":audio}

# @app.middleware("http")
# async def add_process_time_header(request: Request, call_next):
#     start_time = time.time()
#     response = await call_next(request)
#     process_time =round(time.time() - start_time,2)
#     logger.info("request: {} , process_time: {}".format(request.url,process_time))
#     return response


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=20003, workers=1,log_level="info")
