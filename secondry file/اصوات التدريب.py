import sounddevice as sd
from scipy.io.wavfile import write
import os


def record_command(command_name, n_samples=25, duration=2, fs=16000):
    folder = f"secondry file/commands/{command_name}"
    os.makedirs(folder, exist_ok=True)

    for i in range(0, n_samples):
        print(f"Recording {command_name} sample {i}/{n_samples+200} ...")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        write(f"{folder}/{command_name}_{i}.wav", fs, recording)

#record_command("off")
#record_command("on")
#record_command("unlock")
#record_command("count")
record_command("number")