import os
import re
import json
import torch
import textract
import numpy as np
from tqdm import trange
from transformers import AutoTokenizer, AutoModel

def get_all_files(folder):
  files = []
  for rf,_,fs in os.walk(folder):
    for f in fs:
      files.append(os.path.join(rf, f))
  return files


class Processor:
  def __init__(self, hf_backbone, np_path, class_to_id):
    assert "bert" in hf_backbone.lower(), "Supports only BERT Models"

    self.tokenizer = AutoTokenizer.from_pretrained(hf_backbone)
    self.device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    self.model = AutoModel.from_pretrained(hf_backbone).to(self.device)
    self.model.eval()
    self.model_config = self.model.config

    # instead of saving the entire torch model that can be very heavy we merge
    # weights and biases into a single matrix and store as numpy object
    #
    # self.wb.shape = [max_position_embeddings + 1, 3] # 3 classes on which trained
    # w.shape = [max_position_embeddings, 3]
    # b.shape = [1, 3]
    #
    # you can train any model as you want with the code in the accompanying
    # README file.
    self.wb = np.load(np_path)

    # class_to_id is path to a file that has the labels
    with open(class_to_id, "r") as f:
      self.class_to_id = json.load(f)

  def classify(self, all_text):
    maxlen = self.model_config.max_position_embeddings # maximum length supported by BERT model
    encodings = self.tokenizer(all_text, return_tensors = "pt", padding = "longest")
    with torch.no_grad():
      encodings = {k:v[:, :maxlen].to(self.device) for k,v in encodings.items()}
      output = self.model(**encodings).last_hidden_state
      logits = torch.sum(output, dim = 1) # pool the outputs
    logits = logits.cpu().numpy()

    # now to the linear kernel
    # logits = logits @ weights + bias
    # max_classes = logits.argmax(-1)
    classes = (logits @ self.wb[:-1, :] + self.wb[-1]).argmax(-1)
    return classes

  def process(self, files):
    # load the text in the documents
    all_text = []
    pbar = trange(len(files))
    for i in pbar:
      f = files[i]
      pbar.set_description(f"Opening >> {f[:30]}")
      text = textract.process(f, method='pdfminer')
      text = re.sub("\s+", " ", text.decode("utf-8"))
      all_text.append(text)

    # get embeddings from the model
    classes = self.classify(all_text)
    class_labels = [self.class_to_id[x] for x in classes]
    return class_labels
