import numpy as np
import librosa

class SilenceDetector:
    def __init__(self, threshold_db=-45, min_silence_duration=0.2, window_size=2048, hop_length=512):
        """
        Initialize Silence Detector.
        
        Args:
            threshold_db (float): Silence threshold in dB
            min_silence_duration (float): Minimum silence duration in seconds
            window_size (int): Window size for energy calculation
            hop_length (int): Number of samples between successive frames
        """
        self.threshold_db = threshold_db
        self.min_silence_duration = min_silence_duration
        self.window_size = window_size
        self.hop_length = hop_length
    
    def _get_energy(self, audio_data):
        """
        Calculate energy of audio signal.
        
        Args:
            audio_data (numpy.ndarray): Input audio signal
            
        Returns:
            numpy.ndarray: Energy values for each frame
        """
        # Compute STFT
        stft_matrix = librosa.stft(
            audio_data,
            n_fft=self.window_size,
            hop_length=self.hop_length,
            window='hann',
            center=True
        )
        
        # Calculate energy
        energy = np.sum(np.abs(stft_matrix)**2, axis=0)
        
        # Convert to dB
        energy_db = librosa.power_to_db(energy, ref=np.max)
        
        return energy_db
    
    def _find_silence_regions(self, energy_db, sr):
        """
        Find silence regions in audio.
        
        Args:
            energy_db (numpy.ndarray): Energy values in dB
            sr (int): Sample rate
            
        Returns:
            list: List of (start, end) tuples indicating silence regions
        """
        # Find frames below threshold
        is_silence = energy_db < self.threshold_db
        
        # Convert frame indices to sample indices
        silence_starts = []
        silence_ends = []
        
        if is_silence[0]:
            silence_starts.append(0)
        
        # Find silence start/end points
        for i in range(1, len(is_silence)):
            if not is_silence[i-1] and is_silence[i]:
                silence_starts.append(i)
            elif is_silence[i-1] and not is_silence[i]:
                silence_ends.append(i)
        
        if is_silence[-1]:
            silence_ends.append(len(is_silence))
        
        # Convert frame indices to sample indices
        silence_regions = []
        min_frames = int(self.min_silence_duration * sr / self.hop_length)
        
        for start, end in zip(silence_starts, silence_ends):
            if end - start >= min_frames:
                start_sample = start * self.hop_length
                end_sample = min(end * self.hop_length, len(energy_db) * self.hop_length)
                silence_regions.append((start_sample, end_sample))
        
        return silence_regions
    
    def remove_silence(self, audio_data, sr):
        """
        Remove silence regions from audio.
        
        Args:
            audio_data (numpy.ndarray): Input audio signal
            sr (int): Sample rate
            
        Returns:
            numpy.ndarray: Audio signal with silence removed
        """
        # Calculate energy
        energy_db = self._get_energy(audio_data)
        
        # Find silence regions
        silence_regions = self._find_silence_regions(energy_db, sr)
        
        if not silence_regions:
            return audio_data
        
        # Create mask for non-silence regions
        mask = np.ones(len(audio_data), dtype=bool)
        for start, end in silence_regions:
            mask[start:end] = False
        
        # Return audio without silence
        return audio_data[mask]
