import librosa
import soundfile as sf
import numpy as np
import subprocess
import os

def load_audio(file_path, sr=22050, mono=True, convert=True):
    """
    Load audio file.
    
    Args:
        file_path (str): Path to audio file
        sr (int): Target sample rate
        mono (bool): Convert to mono if True
        convert (bool): Convert to float32 if True
        
    Returns:
        tuple: (audio_data, sample_rate)
    """
    try:
        # Load audio file
        audio_data, file_sr = librosa.load(file_path, sr=sr, mono=mono)
        
        # Convert to float32 if requested
        if convert:
            audio_data = audio_data.astype(np.float32)
        
        return audio_data, sr
        
    except Exception as e:
        print(f"Error loading audio file: {str(e)}")
        raise

def save_audio(audio_data, file_path, sr=22050, mp3_bitrate='8k'):
    """
    Save audio file.
    
    Args:
        audio_data (numpy.ndarray): Audio signal to save
        file_path (str): Output file path
        sr (int): Sample rate
        mp3_bitrate (str): MP3 bitrate (e.g., '8k', '16k', '32k', '64k', '96k', '128k', '192k', '256k', '320k')
    """
    try:
        # Ensure audio data is float32
        audio_data = audio_data.astype(np.float32)
        
        # Normalize if needed
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-10)
        
        # For MP3 files, we'll use ffmpeg
        if file_path.lower().endswith('.mp3'):
            # First save as WAV
            temp_wav = file_path + '.temp.wav'
            sf.write(temp_wav, audio_data, sr, format='WAV', subtype='PCM_16')
            
            try:
                # Convert WAV to MP3 using ffmpeg
                bitrate = mp3_bitrate.lower()
                subprocess.run([
                    'ffmpeg',
                    '-y',  # Overwrite output file if it exists
                    '-i', temp_wav,
                    '-acodec', 'libmp3lame',
                    '-ab', bitrate,
                    '-ar', str(sr),
                    file_path
                ], check=True, capture_output=True)
            finally:
                # Clean up temporary WAV file
                if os.path.exists(temp_wav):
                    os.remove(temp_wav)
        else:
            # For non-MP3 files, use soundfile directly
            sf.write(file_path, audio_data, sr)
            
    except Exception as e:
        print(f"Error saving audio file: {str(e)}")
        raise