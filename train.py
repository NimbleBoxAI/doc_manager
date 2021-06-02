"""What does any pytorch based training script look like:

- class ModelConfig
- class Model : nn.Module

- class DataConfig
- class Data : torch.utils.data.Dataset
  - one for training: dstrain
  - one for eval.   : dstest

- class TrainerConfig
  - batch_size
  - lr
  - n_step
  ...

- class Trainer
  - train(): method to train
    
    dl_train = DataLoader(ds_train)
    dl_test = DataLoader(ds_test)
    
    for input, global_step in zip(dl_train, range(n_training_steps)):
      loss = model(input)
      loss.backward()
      optim.step()

      if global_step % test_every_steps == 0:
        test_loss = mean([
          loss = model(test_input)
          for test_input in dl_test
        ])

      if test_loss < previous_lowest_loss:
        save_model()
        no_improvement_evals = 0

      else:
        no_improvement_evals += 1

      if no_improvement_evals == patience:
        break training
"""

import os
import re
import textract
import numpy as np
from glob import glob

import torch
from torch import nn
from torch.nn import functional as F

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

# load all the text
def read_pdf(filename) -> str:
  if os.name == "nt":
    # windows bypass for reading PDF files
    import PyPDF2
    with open(filename, 'rb') as pdf_file:
      read_pdf = PyPDF2.PdfFileReader(pdf_file)
      number_of_pages = read_pdf.getNumPages()
      page = read_pdf.getPage(0)
      page_content = page.extractText()
      text = page_content
  elif os.name == "posix":
      #linux bypass for reading PDF files
    text = textract.process(filename, method='pdfminer')
    text = text.decode("utf-8")
  return text


files = glob("./sample/*.pdf")
all_text = []
for f in files:
  text = read_pdf(f)
  text = re.sub("\s+", " ", text)
  all_text.append(text)


# get logits
with torch.no_grad():
  out = tokenizer(all_text, return_tensors = "pt", padding = "longest")
  output = model(
    **{k:v[:, :model.config.max_position_embeddings] for k,v in out.items()}
  )
  logits = torch.sum(output.last_hidden_state, dim = 1)


# define classifier head
class ClassifierHead(nn.Module):
  def __init__(self, i, c):
    super().__init__()
    self.w = nn.Parameter(data = torch.normal(mean = 0, std = 0.02, size = [i,c]), requires_grad=True)
    self.b = nn.Parameter(data = torch.zeros([c,]))
  def forward(self, x):
    return x@self.w + self.b

c = ClassifierHead(model.config.hidden_dim, 3)

# train the model
optim = torch.optim.AdamOptimizer(c.parameters())
t = torch.Tensor([0, 1, 2, 0, 1, 5, 6]).long() # define target method
for i in range(10):
  out = c(logits)
  loss = F.cross_entropy(out, target = t)
  loss.backward()
  optim.step()

ws = np.vstack([c.w.detach().numpy(), c.b.view(1, c.b.shape[-1]).detach().numpy()])
np.save("./params.npy", ws)
