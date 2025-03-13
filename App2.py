import numpy as np
import pyaudio
import torch
from transformers import Wav2Vec2ProcessorWithLM, Wav2Vec2ForCTC
import torchaudio

# Audio Configuration
FORMAT = pyaudio.paInt16  
CHANNELS = 1
RATE = 16000  # Required sample rate for wav2vec2
CHUNK = 1024
RECORD_SECONDS = 3  # Adjustable

# Load Wav2Vec2 Model
processor = Wav2Vec2ProcessorWithLM.from_pretrained("./wav2vec2-large-xlsr-53-english")
model = Wav2Vec2ForCTC.from_pretrained("./wav2vec2-large-xlsr-53-english")

# Start Recording
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print("Listening... Speak now.")

def record_audio():
    frames = []
    for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):  # Capture for a few seconds
        data = stream.read(CHUNK)
        frames.append(np.frombuffer(data, dtype=np.int16))

    audio_data = np.concatenate(frames, axis=0)
    print("Recording Done.")
    return audio_data.astype(np.float32) / 32768.0  # Normalize to [-1,1]


def audio_to_text(audio_data):
    input_values = processor(audio_data, return_tensors="pt", sampling_rate=RATE).input_values

    print("Input Shape:", input_values.shape)

    with torch.no_grad():
        logits = model(input_values).logits

    print("Logits Shape:", logits.shape)

    # Remove batch dimension and convert to numpy array
    logits_np = logits[0].numpy()

    # Decode using the decoder from the processor
    decoded_beams = processor.decoder.decode_beams(logits_np)
    
    # Extract the top hypothesis (text)
    text = decoded_beams[0][0]

    return text


# Record & Process
audio_data = record_audio()
print("Captured Audio Data:", audio_data[:10])  # Print first 10 samples
extracted_text = audio_to_text(audio_data)

print("Extracted Text:", extracted_text)

# Reference Pronunciation (Example)
reference_text = "pat"  # Example expected word
similarity_score = sum(1 for a, b in zip(extracted_text, reference_text) if a == b) / len(reference_text)

print(f"Pronunciation Accuracy: {similarity_score * 100:.2f}%")
