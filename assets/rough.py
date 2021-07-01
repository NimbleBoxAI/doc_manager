#!/usr/bin/env python3

# actual human readable dataset
ds = {
  "./sample/HAWB1.pdf": "HAWB",
  "./sample/ACM.pdf": "ACM",
  "./sample/INV.pdf": "INV"
}

# step 1: prcoess input and labele
# output of this step:
# {
#   <text_0>: 0,
#   <text_1>: 1,
#   <text_2>: 2,
# }

LABEL_MAPS = {
  "HAWB": 0,
  "ACM": 1,
  "INV": 2
}

def read_pdf(*args):
  return "sadasdfsadfasdfasdfasfd"

INPUT = [read_pdf(path) for path in ds.keys()]
LABEL = [LABEL_MAPS[v] for v in ds.values()]

print(INPUT)

print(LABEL)

# step 2: tokenisation



# ---- how it works on a larger dataset

import torch

class PDFDs(torch.utils.data.Dataset):
  def __init__(self, ds):
    self.ds = ds
    self.files = list(ds.keys())
    self.labels = [LABEL_MAPS[v] for v in ds.values()]

  def __len__(self):
    return len(self.ds)

  def __getitem__(self, i):
    f = self.files[i]
    text = read_pdf(f)
    l = self.labels[i]

    # roughly the output from tokenizer
    out = {
      "input_ids": torch.randint(0, 10, (12,))
    }

    return {
      **out,
      "label": torch.tensor(l).long()
    }
  
print()
d = PDFDs(ds)
print(d[1])

print()

for _ in range(n_epochs):
  dl = torch.utils.data.DataLoader(d, batch_size = 2)
  for x in dl:
    print(x)
    
    loss = model()

    loss.backward()
    optim.step()
