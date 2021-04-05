import glob
import textract
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

# load all the text
all_text = []
for f in files:
  text = textract.process(f, method='pdfminer')
  text = re.sub("\s+", " ", text.decode("utf-8"))
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

c = ClassifierHead(768, 3)


t = torch.Tensor([0, 1, 2, 0, 1, 5, 6]).long() # define target method
for i in range(10):
  out = c(logits)
  loss = F.cross_entropy(out, target = t)
  loss.backward()
  optim.step()

ws = np.vstack([c.w.detach().numpy(), c.b.view(1, c.b.shape[-1]).detach().numpy()])
np.save("./params.npy", ws)