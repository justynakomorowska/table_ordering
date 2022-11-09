import pyaudio
import wave

from pathlib import Path
from tqdm import tqdm

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pyctcdecode import build_ctcdecoder
from wavio import read
import jiwer
import arpa


MODELS = {
    "voxpopuli": "facebook/wav2vec2-base-10k-voxpopuli-ft-pl",
    "fb_large": "facebook/wav2vec2-large-960h-lv60-self", #pomyłka, to jest model nie dla języka polskiego
    "alex_large":"alexcleu/wav2vec2-large-xlsr-polish"
}

#USE = "voxpopuli"
#USE = "fb_large"
USE = "alex_large"

processor=Wav2Vec2Processor.from_pretrained(MODELS[USE])
model=Wav2Vec2ForCTC.from_pretrained(MODELS[USE])


path = "."

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
#RATE = 44100
RATE = 16000
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"

def record(fileout=WAVE_OUTPUT_FILENAME, seconds=RECORD_SECONDS):
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    play("ping.wav")
    print("* recording")

    frames = []
    for i in range(0, int(RATE / CHUNK * seconds)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(fileout, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


def play(fil):
    # Set chunk size of 1024 samples per data frame
    #print(f"playing: {fil}")
    chunk = 1024  
    
    # Open the sound file 
    wf = wave.open(fil, 'rb')
    
    # Create an interface to PortAudio
    p = pyaudio.PyAudio()
    
    # Open a .Stream object to write the WAV file to
    # 'output = True' indicates that the sound will be played rather than recorded
    stream = p.open(format = p.get_format_from_width(wf.getsampwidth()),
                    channels = wf.getnchannels(),
                    rate = wf.getframerate(),
                    output = True)
    
    # Read data in chunks
    data = wf.readframes(chunk)
    
    # Play the sound by writing the audio data to the stream
    while data != b'':
        stream.write(data)
        data = wf.readframes(chunk)
    
    # Close and terminate the stream
    #print("play closing stream")
    stream.close()
    #print("play terminating")
    p.terminate()
    return 0

def sr(fil):
    files={}
    #for f in Path(fil): #.glob(f"{fil}.wav"):
    #print(f)  
    f = Path(fil)
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