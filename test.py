import os
from IPython.display import Audio, display

file_path = "uploaded_audio.wav"

if os.path.exists(file_path):
    print("File found! Trying to play...")
    display(Audio(file_path, autoplay=True))
else:
    print("Error: File not found!")
