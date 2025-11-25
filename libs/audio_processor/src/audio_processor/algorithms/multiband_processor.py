import numpy as np
from scipy import signal
from ..utils.signal_processing import stft, istft

class MultiBandProcessor:
    def __init__(self, n_bands=8, n_fft=2048, hop_length=512):
        """
        Initialize Multi-Band Processor.
        
        Args:
            n_bands (int): Number of frequency bands (increased for better resolution)
            n_fft (int): FFT window size
            hop_length (int): Number of samples between successive frames
        """
        self.n_bands = n_bands
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Create frequency band boundaries using mel scale
        self.band_boundaries = self._create_band_boundaries()
    
    def _create_band_boundaries(self):
        """
        Create frequency band boundaries using mel scale.
        
        Returns:
            list: List of frequency band boundary indices
        """
        # Use mel scale for band division
        mel_max = 2595 * np.log10(1 + (self.n_fft/2) / 700.0)
        mel_points = np.linspace(0, mel_max, self.n_bands + 1)
        
        # Convert back to Hz
        freq_points = 700 * (10**(mel_points / 2595.0) - 1)
        
        # Convert to FFT bin indices
        return np.round(freq_points * (self.n_fft / 2) / (self.n_fft/2)).astype(int)
    
    def _reduce_static_noise(self, stft_band):
        """
        Reduce static noise in a frequency band.
        
        Args:
            stft_band (numpy.ndarray): STFT coefficients for a frequency band
            
        Returns:
            numpy.ndarray: Enhanced STFT coefficients
        """
        # Get magnitude and phase
        magnitude = np.abs(stft_band)
        phase = np.angle(stft_band)
        
        # Estimate noise floor
        noise_floor = np.percentile(magnitude, 15, axis=1, keepdims=True)
        
        # Compute time-varying threshold
        threshold = noise_floor * 1.5
        
        # Create soft mask
        gain = np.maximum(1 - threshold / (magnitude + 1e-10), 0)
        gain = gain ** 2  # Squared for smoother transition
        
        # Apply temporal smoothing to gain
        window_size = 5
        smoothing_kernel = np.hanning(window_size)[:, np.newaxis] / np.sum(np.hanning(window_size))
        gain = signal.convolve2d(gain, smoothing_kernel, mode='same', boundary='symm')
        
        # Apply gain and preserve phase
        return magnitude * gain * np.exp(1j * phase)
    
    def process(self, audio_data):
        """
        Apply multi-band processing with static noise reduction.
        
        Args:
            audio_data (numpy.ndarray): Input audio signal
            
        Returns:
            numpy.ndarray: Enhanced audio signal
        """
        # Compute STFT
        stft_matrix = stft(audio_data, n_fft=self.n_fft, hop_length=self.hop_length)
        
        # Process each band
        processed_stft = np.zeros_like(stft_matrix)
        
        for i in range(self.n_bands):
            start_bin = self.band_boundaries[i]
            end_bin = self.band_boundaries[i + 1]
            
            # Extract band
            band = stft_matrix[start_bin:end_bin, :]
            
            # Process band
            processed_band = self._reduce_static_noise(band)
            
            # Store processed band
            processed_stft[start_bin:end_bin, :] = processed_band
        
        # Apply additional denoising for high frequencies (where static noise is often present)
        high_freq_start = self.band_boundaries[int(self.n_bands * 0.7)]  # Top 30% of bands
        processed_stft[high_freq_start:, :] *= 0.7  # Reduce high frequency content
        
        # Inverse STFT
        enhanced_signal = istft(processed_stft, hop_length=self.hop_length)
        
        return enhanced_signal
