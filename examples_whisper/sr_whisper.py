

from pathlib import Path
from tqdm import tqdm

import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from pyctcdecode import build_ctcdecoder
from wavio import read
import jiwer
import arpa

processor = AutoProcessor.from_pretrained("openai/whisper-base")
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-base")

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
  #feats=processor(data,sampling_rate=Fs,return_tensors='pt',padding=True)
  #print(feats)
  #print(feats.input_features.shape)
  #out=model(feats.input_features)
  #predicted_ids=torch.argmax(out.logits,dim=-1)
  #sent=processor.batch_decode(predicted_ids)[0]
  #trans[name]=sent

  model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language = "pl", task = "transcribe")
  input_features = processor(data, return_tensors="pt").input_features 
  predicted_ids = model.generate(input_features)
  trans[name] = processor.batch_decode(predicted_ids)

print(trans)