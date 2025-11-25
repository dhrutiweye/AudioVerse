import numpy as np
from ..utils.signal_processing import stft, istft

class WienerFilter:
    def __init__(self, n_fft=512, hop_length=128, smoothing_factor=0.1):
        """
        Initialize Wiener Filter with minimum statistics noise estimation.
        
        Args:
            n_fft (int): FFT window size (reduced for better time resolution)
            hop_length (int): Number of samples between successive frames
            smoothing_factor (float): Smoothing factor for noise estimation
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.smoothing_factor = smoothing_factor
    
    def estimate_noise(self, power_spectrum, window_size=30):
        """
        Estimate noise using minimum statistics method.
        
        Args:
            power_spectrum (numpy.ndarray): Power spectrum matrix
            window_size (int): Window size for minimum search
            
        Returns:
            numpy.ndarray: Estimated noise power spectrum
        """
        n_freqs, n_frames = power_spectrum.shape
        
        # Initialize with mean of first few frames
        noise_estimate = np.mean(power_spectrum[:, :min(window_size, n_frames)], axis=1, keepdims=True)
        
        # Sliding window minimum search with temporal smoothing
        for i in range(n_frames - window_size):
            window = power_spectrum[:, i:i+window_size]
            min_in_window = np.min(window, axis=1, keepdims=True)
            noise_estimate = (1 - self.smoothing_factor) * noise_estimate + self.smoothing_factor * min_in_window
        
        return noise_estimate
    
    def process(self, audio_data):
        """
        Apply Wiener filtering with minimum statistics noise estimation.
        
        Args:
            audio_data (numpy.ndarray): Input audio signal
            
        Returns:
            numpy.ndarray: Enhanced audio signal
        """
        # Ensure input is numpy array
        audio_data = np.asarray(audio_data, dtype=np.float32)
        
        # Compute STFT
        stft_matrix = stft(audio_data, n_fft=self.n_fft, hop_length=self.hop_length)
        
        # Get magnitude and phase
        magnitude = np.abs(stft_matrix)
        phase = np.angle(stft_matrix)
        
        # Compute power spectrum
        power_spectrum = magnitude ** 2
        
        # Estimate noise using minimum statistics
        noise_power = self.estimate_noise(power_spectrum)
        
        # Compute a priori SNR with flooring
        prior_snr = np.maximum(power_spectrum / (noise_power + 1e-10) - 1, 0.1)
        
        # Compute Wiener filter gain (more conservative)
        gain = (prior_snr / (prior_snr + 1)) ** 0.5
        
        # Apply temporal smoothing to gain
        smoothed_gain = np.zeros_like(gain)
        alpha = 0.9  # Smoothing factor
        smoothed_gain[:, 0] = gain[:, 0]
        for i in range(1, gain.shape[1]):
            smoothed_gain[:, i] = alpha * smoothed_gain[:, i-1] + (1 - alpha) * gain[:, i]
        
        # Blend original and processed magnitudes
        blend_factor = 0.8  # 80% original, 20% processed
        final_magnitude = blend_factor * magnitude + (1 - blend_factor) * (magnitude * smoothed_gain)
        
        # Reconstruct with original phase
        enhanced_stft = final_magnitude * np.exp(1j * phase)
        
        # Inverse STFT
        enhanced_signal = istft(enhanced_stft, hop_length=self.hop_length)
        
        # Ensure output length matches input
        if len(enhanced_signal) > len(audio_data):
            enhanced_signal = enhanced_signal[:len(audio_data)]
        elif len(enhanced_signal) < len(audio_data):
            enhanced_signal = np.pad(enhanced_signal, (0, len(audio_data) - len(enhanced_signal)))
        
        return enhanced_signal
