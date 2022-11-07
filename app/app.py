from utils import record, play, sr

odp = {}

if __name__ == "__main__":
    play("powitanie.wav")
    task="liczba_osob"
    print(task)
    play(f"{task}.wav")
    odp_fil = f"odp_{task}.wav"
    record(odp_fil)
    odp[task] = sr(odp_fil)
    
    task="data_rezerwacji"
    print(task)
    play(f"{task}.wav")
    odp_fil = f"odp_{task}.wav"
    record(odp_fil)
    odp[task] = sr(odp_fil)
    
    task="godzina_rezerwacji"
    print(task)
    play(f"{task}.wav")
    odp_fil = f"odp_{task}.wav"
    record(odp_fil)
    odp[task] = sr(odp_fil)
    
    task="nr_telefonu"
    print(task)
    play(f"{task}.wav")
    odp_fil = f"odp_{task}.wav"
    record(odp_fil)
    odp[task] = sr(odp_fil)

    play("pozegnanie.wav")

    print(odp)