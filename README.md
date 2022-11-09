# table_ordering
Speech recognition app for table ordering use case.
For Polish speakers.

# references

1. [RealPython recording example](https://realpython.com/python-speech-recognition/#working-with-microphones)
2. [RealPython .wav files playing example](https://realpython.com/playing-and-recording-sound-python/#pyaudio)
3. [Polish Speeach Recognition Models on Hugging Face](https://huggingface.co/models?language=pl&pipeline_tag=automatic-speech-recognition&sort=downloads). 

## bash commands

All commands below to be rum in terminal

```bash
pip install https://github.com/huggingface/transformers/archive/refs/heads/master.zip
pip install https://github.com/kensho-technologies/pyctcdecode/archive/refs/heads/main.zip
pip install https://github.com/kpu/kenlm/archive/master.zip # ignore
pip install wavio
pip install jiwer
pip install arpa

```
### **record.py** records what you're saying.
### **sr.py** stands for speech recognition. Blackbox model downloaded from HF transcripts the speech.

#### Examples:
1. run examples_pyaudio/record.py - output.wav will be generated 
2. run examples_zum2sr.py - recorded text will be recognized. Best attempt.

## Three models have been compared and on basis of WER value the best one has been chosen.
1. [voxpopuli pl](https://huggingface.co/facebook/wav2vec2-base-10k-voxpopuli-ft-pl)
2. [whisper base](https://huggingface.co/openai/whisper-base)
3. [alexcleu large polish](https://huggingface.co/alexcleu/wav2vec2-large-xlsr-polish)

The last one *alexcleu large polish* has the best performance. It is going to be used for table ordering transcription.
