from flask import Flask, request, jsonify
import numpy as np
import torch
from transformers import Wav2Vec2ProcessorWithLM, Wav2Vec2ForCTC
import io
import soundfile as sf
import torchaudio
import os
import uuid
import librosa
from flask_cors import CORS  # Import CORS for cross-origin support

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load Wav2Vec2 Model (Load once at startup)
processor = Wav2Vec2ProcessorWithLM.from_pretrained("./wav2vec2-large-xlsr-53-english")
model = Wav2Vec2ForCTC.from_pretrained("./wav2vec2-large-xlsr-53-english")

# Target sample rate for the model
TARGET_SAMPLE_RATE = 16000

def audio_to_text(audio_data, sample_rate):
    # Ensure sample rate matches what model expects
    if sample_rate != TARGET_SAMPLE_RATE:
        print(f"Resampling from {sample_rate}Hz to {TARGET_SAMPLE_RATE}Hz")
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)
    
    # Ensure audio is floating point and properly scaled
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    
    # Normalize audio if needed
    if np.max(np.abs(audio_data)) > 1.0:
        audio_data = audio_data / np.max(np.abs(audio_data))
    
    # Process with model
    input_values = processor(audio_data, return_tensors="pt", sampling_rate=TARGET_SAMPLE_RATE).input_values
    
    with torch.no_grad():
        logits = model(input_values).logits
    
    logits_np = logits[0].numpy()
    decoded_beams = processor.decoder.decode_beams(logits_np)
    text = decoded_beams[0][0]
    
    return text

def calculate_similarity(extracted_text, reference_text):
    if not reference_text:
        return 0.0
        
    min_length = min(len(extracted_text), len(reference_text))
    if min_length == 0:
        return 0.0
        
    similarity_score = sum(1 for a, b in zip(extracted_text[:min_length], reference_text[:min_length]) if a == b) / len(reference_text)
    return similarity_score * 100

@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'audio' not in request.files or 'referenceText' not in request.form:
        return jsonify({'error': 'Audio file and reference text are required'}), 400
    
    audio_file = request.files['audio']
    reference_text = request.form['referenceText']
    
    # Get original sample rate if provided
    original_sample_rate = request.form.get('originalSampleRate', '0')
    try:
        original_sample_rate = int(original_sample_rate)
    except ValueError:
        original_sample_rate = 0
    
    print(f"Client reported original sample rate: {original_sample_rate}Hz")
    
    try:
        # Generate unique filenames
        temp_filename = f"{uuid.uuid4().hex}.webm"
        temp_filepath = os.path.join("audio_storage", temp_filename)
        
        # Ensure directory exists
        os.makedirs("audio_storage", exist_ok=True)
        
        # Save uploaded file
        audio_file.save(temp_filepath)
        print(f"Saved uploaded file to {temp_filepath}, size: {os.path.getsize(temp_filepath)} bytes")
        
        # Load the audio and get its properties
        try:
            # First attempt with soundfile
            print("Attempting to load with soundfile...")
            audio_data, sample_rate = sf.read(temp_filepath)
            print(f"Loaded with soundfile: {sample_rate}Hz, shape: {audio_data.shape}, dtype: {audio_data.dtype}")
            
        except Exception as sf_error:
            print(f"Soundfile error: {str(sf_error)}")
            
            try:
                # Try with librosa instead
                print("Attempting to load with librosa...")
                audio_data, sample_rate = librosa.load(temp_filepath, sr=None)
                print(f"Loaded with librosa: {sample_rate}Hz, shape: {audio_data.shape}, dtype: {audio_data.dtype}")
                
            except Exception as librosa_error:
                print(f"Librosa error: {str(librosa_error)}")
                
                try:
                    # Last resort: torchaudio
                    print("Attempting to load with torchaudio...")
                    waveform, sample_rate = torchaudio.load(temp_filepath)
                    audio_data = waveform.numpy()[0]  # Take first channel if stereo
                    print(f"Loaded with torchaudio: {sample_rate}Hz, shape: {audio_data.shape}, dtype: {audio_data.dtype}")
                    
                except Exception as torch_error:
                    return jsonify({
                        'error': f"Failed to load audio file. Try a different format.",
                        'details': f"SF: {sf_error}, Librosa: {librosa_error}, Torch: {torch_error}"
                    }), 500
        
        # Check for obviously wrong sample rates (caused by format issues)
        if sample_rate < 8000 or sample_rate > 96000:
            if original_sample_rate > 8000 and original_sample_rate < 96000:
                print(f"Detected likely incorrect sample rate {sample_rate}Hz, using client-reported {original_sample_rate}Hz")
                sample_rate = original_sample_rate
            else:
                print(f"Detected likely incorrect sample rate {sample_rate}Hz, defaulting to {TARGET_SAMPLE_RATE}Hz")
                sample_rate = TARGET_SAMPLE_RATE
        
        # Handle multi-channel audio
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            print("Converting multi-channel to mono")
            audio_data = np.mean(audio_data, axis=1)
            
        # Convert to float32 if needed
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
            
        # Normalize audio amplitude
        audio_data = audio_data / np.max(np.abs(audio_data))
            
        # Save a properly formatted WAV for verification (and possible download)
        wav_filename = f"{uuid.uuid4().hex}.wav"
        wav_filepath = os.path.join("audio_storage", wav_filename)
        sf.write(wav_filepath, audio_data, TARGET_SAMPLE_RATE, subtype='FLOAT')
        print(f"Saved normalized audio to {wav_filepath}")
        
        # Process audio
        extracted_text = audio_to_text(audio_data, sample_rate)
        similarity = calculate_similarity(extracted_text, reference_text)
        
        # Clean up temporary files
        try:
            os.remove(temp_filepath)
            # Keep the WAV file for debugging if needed
            # os.remove(wav_filepath)
        except Exception as e:
            print(f"Warning: Failed to clean up temp files: {e}")
        
        return jsonify({
            'extractedText': extracted_text, 
            'similarityScore': similarity,
            'details': {
                'originalSampleRate': sample_rate,
                'processingRate': TARGET_SAMPLE_RATE,
                'audioLength': len(audio_data) / TARGET_SAMPLE_RATE
            }
        })
        
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        print(f"Error processing audio: {str(e)}")
        print(traceback_str)
        return jsonify({'error except': str(e), 'traceback': traceback_str}), 500

if __name__ == '__main__':
    app.run(debug=True)