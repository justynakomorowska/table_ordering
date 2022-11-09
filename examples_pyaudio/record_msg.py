"""
desc: helping script for recording messages played to users
"""

from record import record
from time import sleep
if __name__ =="__main__":
    messages=["powitanie", "liczba_osob", "data_rezerwacji", "godzina_rezerwacji", "nr_telefonu","pozegnanie"]

    for msg in messages:
        print(f"recording {msg}")
        record(f"{msg}.wav", 7)
        sleep(3)