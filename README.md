# table_ordering
Speech recognition app for table ordering use case.
For Polish speakers.

# references

1. [RealPython example](https://realpython.com/python-speech-recognition/#working-with-microphones)
2. [Polish Speeach Recognition Models on Hugging Face](https://huggingface.co/models?language=pl&pipeline_tag=automatic-speech-recognition&sort=downloads). 

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

Examples:
1. run examples_pyaudio/record.py - output.wav will be generated 
2. run examples_zum2sr.py - recorded text will be recognized. Best attempt.