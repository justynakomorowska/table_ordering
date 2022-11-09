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

ref = {
"data_rezerwacji" : "podaj datę rezerwacji",
"godzina_rezerwacji" : "na która godzinę zarezerwować stolik",
"liczba_osob" : "dla ilu osob chcesz zarezerwować stolik",
"nr_telefonu" : "zostaw swój numer telefonu",
"pozegnanie": "dziękujemy do zobaczenia",
"powitanie" : "dzień dobry tu restauracja mamma italiana witaj w systemie rezerwacji stolika",
"nagranie1": "raz dwa trzy cztery pięć",
"nagranie2": "raz sześć siedem osiem dziewięć "
}

MODELS = {
    "voxpopuli": "facebook/wav2vec2-base-10k-voxpopuli-ft-pl",
    "fb_large": "facebook/wav2vec2-large-960h-lv60-self",
    "alex_large":"alexcleu/wav2vec2-large-xlsr-polish"
}

#USE = "voxpopuli"
#USE = "fb_large"
USE = "alex_large"

processor=Wav2Vec2Processor.from_pretrained(MODELS[USE])
model=Wav2Vec2ForCTC.from_pretrained(MODELS[USE])


files={}

path = "." #. bo folder glowny. odwoluje sie do folderu nad soba i szuka plikow .wav
#path = "./samples"
"""
for f in Path(path).glob('nagranie*.wav'): 
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

"""
def wer(name):
    """
    compares transcription between name.wav file content and ref[name] - text file content
    """
    trans = sr(f"{name}.wav")
    h = []
    r = []
    h.append(trans[name])
    r.append(ref[name])
    print(f"trans: {trans[name]}")
    print(f"ref: {ref[name]}")

    return jiwer.compute_measures(r,h)

def sr(f):
    files={}
    f=Path(f"./{f}") # changing file name to path object
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

if __name__ == "__main__":

    print(wer("powitanie"))