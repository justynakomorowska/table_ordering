"""
author: jk
date: 11/6/2022
description: model from HuggingFace used as speech recognizer

"""

from pathlib import Path
from tqdm import tqdm

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pyctcdecode import build_ctcdecoder
from wavio import read
import jiwer
import arpa


processor=Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-10k-voxpopuli-ft-pl')
model=Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-10k-voxpopuli-ft-pl')

files={}

path = "."
#path = "./samples"

for f in Path(path).glob('*.wav'): #. bo folder glowny. odwoluje sie do folderu nad soba i szuka plikow .wav
  print(f)  
  data=read(str(f))
  files[f.stem]=data.data.squeeze().astype('float32')


Fs=data.rate
for name,d in files.items():
  print(f'{name}: {d.size/Fs:0.2f}s')

#print(model.summarize())

trans={}
for name,data in tqdm(files.items()):
  print(data)
  feats=processor(data,sampling_rate=Fs,return_tensors='pt',padding=True)
  print(feats)
  print(feats.input_values.shape)
  out=model(input_values=feats.input_values)
  predicted_ids=torch.argmax(out.logits,dim=-1)
  sent=processor.batch_decode(predicted_ids)[0]
  trans[name]=sent

print(trans)


def sr(f):
    files={}
    print(f)  
    data=read(str(f))
    files[f.stem]=data.data.squeeze().astype('float32')


    Fs=data.rate
    for name,d in files.items():
        print(f'{name}: {d.size/Fs:0.2f}s')


    trans={}
    for name,data in tqdm(files.items()):
        print(data)
        feats=processor(data,sampling_rate=Fs,return_tensors='pt',padding=True)
        print(feats)
        print(feats.input_values.shape)
        out=model(input_values=feats.input_values)
        predicted_ids=torch.argmax(out.logits,dim=-1)
        sent=processor.batch_decode(predicted_ids)[0]
        trans[name]=sent
    return trans