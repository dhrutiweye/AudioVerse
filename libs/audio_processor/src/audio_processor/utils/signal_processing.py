import numpy as np
import librosa

def stft(audio_data, n_fft=2048, hop_length=512, win_length=None, window='hann'):
    """
    Compute the Short-Time Fourier Transform.
    
    Args:
        audio_data (numpy.ndarray): Input audio signal
        n_fft (int): FFT window size
        hop_length (int): Number of samples between successive frames
        win_length (int, optional): Window length. Defaults to n_fft
        window (str): Window type
        
    Returns:
        numpy.ndarray: Complex-valued matrix of STFT coefficients
    """
    return librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length,
                       win_length=win_length, window=window)

def istft(stft_matrix, hop_length=512, win_length=None, window='hann'):
    """
    Compute the Inverse Short-Time Fourier Transform.
    
    Args:
        stft_matrix (numpy.ndarray): STFT coefficient matrix
        hop_length (int): Number of samples between successive frames
        win_length (int, optional): Window length. Defaults to n_fft
        window (str): Window type
        
    Returns:
        numpy.ndarray: Time-domain signal
    """
    return librosa.istft(stft_matrix, hop_length=hop_length,
                        win_length=win_length, window=window)

def get_power_spectrum(stft_matrix):
    """
    Compute the power spectrum from STFT coefficients.
    
    Args:
        stft_matrix (numpy.ndarray): STFT coefficient matrix
        
    Returns:
        numpy.ndarray: Power spectrum matrix
    """
    return np.abs(stft_matrix) ** 2

def get_magnitude_spectrum(stft_matrix):
    """
    Compute the magnitude spectrum from STFT coefficients.
    
    Args:
        stft_matrix (numpy.ndarray): STFT coefficient matrix
        
    Returns:
        numpy.ndarray: Magnitude spectrum matrix
    """
    return np.abs(stft_matrix)

def get_phase_spectrum(stft_matrix):
    """
    Compute the phase spectrum from STFT coefficients.
    
    Args:
        stft_matrix (numpy.ndarray): STFT coefficient matrix
        
    Returns:
        numpy.ndarray: Phase spectrum matrix
    """
    return np.angle(stft_matrix)

def apply_gain(spectrum, gain):
    """
    Apply gain to spectrum.
    
    Args:
        spectrum (numpy.ndarray): Input spectrum
        gain (numpy.ndarray): Gain to apply
        
    Returns:
        numpy.ndarray: Modified spectrum
    """
    return spectrum * gain
