import numpy as np
from sklearn.decomposition import FastICA
import librosa

class ICAProcessor:
    def __init__(self, n_components=2, n_fft=2048, hop_length=512, window='hann'):
        """
        Initialize ICA Processor for speech separation.
        
        Args:
            n_components (int): Number of independent components to extract
            n_fft (int): FFT window size
            hop_length (int): Number of samples between successive frames
            window (str): Window type for STFT
        """
        self.n_components = n_components
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = window
        self.ica = FastICA(
            n_components=n_components,
            random_state=42,
            max_iter=1000,
            tol=0.0001,
            whiten='unit-variance'
        )
    
    def process(self, audio_data):
        """
        Apply ICA-based speech separation.
        
        Args:
            audio_data (numpy.ndarray): Input audio signal
            
        Returns:
            numpy.ndarray: Enhanced audio signal with separated voices
        """
        # Compute STFT
        stft_matrix = librosa.stft(
            audio_data,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True
        )
        
        # Get magnitude and phase
        magnitude = np.abs(stft_matrix)
        phase = np.angle(stft_matrix)
        
        # Prepare features matrix (time_frames x features)
        X = magnitude.T
        
        # Normalize features
        X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-10)
        
        try:
            # Apply ICA
            separated = self.ica.fit_transform(X)
            
            # Reshape back to STFT-like structure
            separated = separated.T.reshape(self.n_components, magnitude.shape[0], -1)
            
            # Process each separated component
            enhanced_components = []
            for i in range(self.n_components):
                # Get current component
                component = separated[i]
                
                # Apply soft mask
                mask = np.abs(component) / (np.sum(np.abs(separated), axis=0) + 1e-10)
                enhanced_magnitude = magnitude * mask
                
                # Simple noise reduction
                threshold = np.mean(enhanced_magnitude) * 0.5
                noise_mask = enhanced_magnitude > threshold
                enhanced_magnitude = enhanced_magnitude * noise_mask
                
                # Reconstruct with original phase
                enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
                
                # Inverse STFT
                enhanced_signal = librosa.istft(
                    enhanced_stft,
                    hop_length=self.hop_length,
                    window=self.window,
                    center=True
                )
                
                enhanced_components.append(enhanced_signal)
            
            # Combine enhanced components
            combined = np.sum(enhanced_components, axis=0)
            
            # Normalize
            combined = combined / (np.max(np.abs(combined)) + 1e-10)
            
            return combined
            
        except Exception as e:
            # If ICA fails, fall back to simple enhancement
            print("Warning: ICA separation failed, falling back to simple enhancement")
            
            # Simple spectral subtraction
            threshold = np.mean(magnitude) * 0.5
            mask = magnitude > threshold
            enhanced_magnitude = magnitude * mask
            
            # Reconstruct
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_signal = librosa.istft(
                enhanced_stft,
                hop_length=self.hop_length,
                window=self.window,
                center=True
            )
            
            return enhanced_signal
