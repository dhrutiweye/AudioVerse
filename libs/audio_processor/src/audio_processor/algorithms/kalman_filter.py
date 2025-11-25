import numpy as np
from ..utils.signal_processing import stft, istft

class KalmanFilter:
    def __init__(self, state_dim=2, n_fft=512, hop_length=128):
        """
        Initialize Kalman Filter for audio enhancement.
        
        Args:
            state_dim (int): Dimension of state vector
            n_fft (int): FFT window size (reduced for better time resolution)
            hop_length (int): Number of samples between successive frames
        """
        self.state_dim = state_dim
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Initialize state transition matrix (more conservative)
        self.F = np.array([[0.98, 0.0],
                          [0.0, 0.98]])
        
        # Initialize measurement matrix
        self.H = np.array([[1.0, 0.0]])
        
        # Initialize process noise covariance (reduced for less aggressive filtering)
        self.Q = np.array([[0.0001, 0.0],
                          [0.0, 0.0001]])
        
        # Initialize measurement noise covariance (increased trust in measurements)
        self.R = np.array([[1.0]])
    
    def _init_state(self):
        """
        Initialize state vector and covariance.
        
        Returns:
            tuple: (Initial state vector, Initial state covariance)
        """
        x = np.zeros((self.state_dim, 1))
        P = np.array([[0.01, 0.0],
                     [0.0, 0.01]])
        return x, P
    
    def process(self, audio_data):
        """
        Apply Kalman filtering for audio enhancement.
        
        Args:
            audio_data (numpy.ndarray): Input audio signal
            
        Returns:
            numpy.ndarray: Enhanced audio signal
        """
        # Ensure input is numpy array with correct shape
        audio_data = np.asarray(audio_data)
        if audio_data.ndim == 0:
            audio_data = np.zeros(1, dtype=np.float32)
        elif audio_data.ndim == 1:
            audio_data = audio_data.astype(np.float32)
        else:
            audio_data = audio_data.flatten().astype(np.float32)
        
        # Compute STFT
        stft_matrix = stft(audio_data, n_fft=self.n_fft, hop_length=self.hop_length)
        
        # Initialize output STFT matrix
        enhanced_stft = np.zeros_like(stft_matrix)
        
        # Process each frequency bin
        for freq_bin in range(stft_matrix.shape[0]):
            # Initialize state for this frequency bin
            x, P = self._init_state()
            
            # Get magnitude and phase
            magnitude = np.abs(stft_matrix[freq_bin, :])
            phase = np.angle(stft_matrix[freq_bin, :])
            
            # Process each time frame
            enhanced_magnitude = np.zeros_like(magnitude)
            for frame in range(stft_matrix.shape[1]):
                # Prediction step
                x_pred = self.F @ x
                P_pred = self.F @ P @ self.F.T + self.Q
                
                # Update step
                measurement = magnitude[frame].reshape(1, 1)
                S = self.H @ P_pred @ self.H.T + self.R
                K = P_pred @ self.H.T @ np.linalg.inv(S)
                x = x_pred + K @ (measurement - self.H @ x_pred)
                P = (np.eye(self.state_dim) - K @ self.H) @ P_pred
                
                # Store enhanced magnitude
                enhanced_magnitude[frame] = x[0, 0]
            
            # Blend original and enhanced magnitudes (less aggressive)
            blend_factor = 0.7  # 70% original, 30% enhanced
            final_magnitude = blend_factor * magnitude + (1 - blend_factor) * enhanced_magnitude
            
            # Preserve original phase
            enhanced_stft[freq_bin, :] = final_magnitude * np.exp(1j * phase)
        
        # Inverse STFT with overlap-add
        enhanced_signal = istft(enhanced_stft, hop_length=self.hop_length)
        
        # Ensure output length matches input
        if len(enhanced_signal) > len(audio_data):
            enhanced_signal = enhanced_signal[:len(audio_data)]
        elif len(enhanced_signal) < len(audio_data):
            enhanced_signal = np.pad(enhanced_signal, (0, len(audio_data) - len(enhanced_signal)))
        
        return enhanced_signal.astype(np.float32)
