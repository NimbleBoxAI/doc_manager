import fastapi as fa
from typing import Optional
from pydantic import BaseModel

from fastapi.staticfiles import StaticFiles # static files on the system
from fastapi.templating import Jinja2Templates # jinja templates

from doc_manager import get_all_files, Processor


###### global constants ######
app = fa.FastAPI()
app.mount("/static", StaticFiles(directory="assets/static"), name="static")
templates = Jinja2Templates(directory="assets")

proc = Processor("distilbert-base-uncased", "./params.npy", "./classes.json")


###### Models ######
# https://fastapi.tiangolo.com/tutorial/response-model/

class SimpleReq(BaseModel):
  folder: Optional[str] = "./sample" # this is the default folder to check for
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
