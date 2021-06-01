import os
import fastapi as fa
from typing import Optional
from pydantic import BaseModel

from fastapi.staticfiles import StaticFiles # static files on the system
from fastapi.templating import Jinja2Templates # jinja templates

from doc_manager import get_all_files, Processor

try:
  from daily import *
except ImportError as e:
  import requests
  x = requests.get("https://gist.githubusercontent.com/yashbonde/62df9d16858a43775c22a6af00a8d707/raw/0764da94f5e243b2bca983a94d5d6a4e4a7eb28a/daily.py").content
  with open("daily.py", "wb") as f:
    f.write(x)
  from daily import *


###### global constants ######
FOLDER = folder(__file__)
app = fa.FastAPI()
app.mount("/static", StaticFiles(directory="assets/static"), name="static")
templates = Jinja2Templates(directory="assets")

proc = Processor(
  hf_backbone = "distilbert-base-uncased",
  np_path = os.path.join(FOLDER, "params.npy"),
  class_to_id = os.path.join(FOLDER, "classes.json")
)


###### Models ######
# https://fastapi.tiangolo.com/tutorial/response-model/

class SimpleReq(BaseModel):
  here = folder(__file__)
  folder: Optional[str] = os.path.join(here, "sample") # this is the default folder to check for
  data: Optional[list] = None # list of objects {"file_name", "tag"}


@app.get("/")
def root(request: fa.Request, folder: str = None):
  print(folder)
  if folder:
    files = get_all_files(folder)
    tags = proc.process(files)
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "data": [
          {"file": f, "tags": t} for f,t in zip(files, tags)
        ]
    })
  else:
    return templates.TemplateResponse("index.html", {
          "request": request, 
          "data": [
            {"file": "asdfasdfasdf.png", "tags": "image, png"},
            {"file": "546345g3j345.png", "tags": "fish, png"},
            {"file": "gd9f8g7783j4.jpg", "tags": "dog, jpg"},
            {"file": "0fg8g7sd88gg.pdf", "tags": "cat, pdf"},
          ]
      })
