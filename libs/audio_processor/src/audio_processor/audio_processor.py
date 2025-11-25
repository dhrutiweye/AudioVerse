import os
import tempfile
import numpy as np
import librosa
from typing import Optional, Tuple

from .utils.audio_io import load_audio, save_audio
from .utils.silence_detector import SilenceDetector
from helper import get_logger

logger = get_logger(os.path.basename(__file__))


class AudioProcessor:
    def __init__(self):
        # Initialize silence detectors with different thresholds
        self.initial_detector = SilenceDetector(
            threshold_db=-55,  # Very aggressive
            min_silence_duration=0.02,  # Very short for static bursts
            window_size=1024,
            hop_length=256
        )
        self.aggressive_detector = SilenceDetector(
            threshold_db=-45,  # Moderately aggressive
            min_silence_duration=0.1,
            window_size=2048,
            hop_length=512
        )
        self.standard_detector = SilenceDetector(
            threshold_db=-40,  # Standard threshold
            min_silence_duration=1.0,
            window_size=2048,
            hop_length=512
        )

    def detect_static_noise(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        """
        Detect frames containing static noise based on spectral characteristics.
        """
        # Compute STFT with consistent parameters
        n_fft = 2048
        hop_length = 512

        stft_matrix = librosa.stft(
            audio_data,
            n_fft=n_fft,
            hop_length=hop_length,
            window='hann',
            center=True
        )

        magnitude = np.abs(stft_matrix)

        # Compute spectral features
        spectral_flatness = librosa.feature.spectral_flatness(S=magnitude)[0]
        spectral_centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0]
        rms = librosa.feature.rms(S=magnitude)[0]

        # Normalize features
        spectral_flatness = (spectral_flatness - np.mean(spectral_flatness)) / (np.std(spectral_flatness) + 1e-10)
        spectral_centroid = (spectral_centroid - np.mean(spectral_centroid)) / (np.std(spectral_centroid) + 1e-10)
        rms = (rms - np.mean(rms)) / (np.std(rms) + 1e-10)

        # Compute frame-level static detection
        static_frames = np.zeros_like(spectral_flatness, dtype=bool)

        # Static noise typically has:
        # - High spectral flatness
        # - Stable spectral centroid
        # - Relatively constant RMS
        for i in range(1, len(spectral_flatness)):
            is_static = (
                    spectral_flatness[i] > 0.5 and  # High flatness
                    abs(spectral_centroid[i] - spectral_centroid[i - 1]) < 0.3 and  # Stable centroid
                    abs(rms[i] - rms[i - 1]) < 0.3  # Stable RMS
            )
            static_frames[i] = is_static

        # Convert frame-wise detection to samples
        static_mask = np.repeat(static_frames, hop_length)

        # Ensure mask length matches audio length
        if len(static_mask) > len(audio_data):
            static_mask = static_mask[:len(audio_data)]
        else:
            static_mask = np.pad(static_mask, (0, len(audio_data) - len(static_mask)))

        return static_mask

    def remove_static_segments(self, audio_data: np.ndarray, sr: int, min_duration: float = 0.05) -> np.ndarray:
        """
        Remove segments containing static noise.
        """
        # Detect static noise
        static_mask = self.detect_static_noise(audio_data, sr)

        # Find static regions
        static_regions = []
        region_start = None
        min_samples = int(min_duration * sr)

        for i in range(len(static_mask)):
            if static_mask[i] and region_start is None:
                region_start = i
            elif not static_mask[i] and region_start is not None:
                region_end = i
                if (region_end - region_start) >= min_samples:
                    static_regions.append((region_start, region_end))
                region_start = None

        if region_start is not None and (len(static_mask) - region_start) >= min_samples:
            static_regions.append((region_start, len(static_mask)))

        # Remove static regions with crossfading
        clean_audio = np.copy(audio_data)
        fade_samples = int(0.01 * sr)  # 10ms crossfade

        for start, end in static_regions:
            # Apply fade out before static
            if start >= fade_samples:
                fade_out = np.linspace(1, 0, fade_samples)
                clean_audio[start - fade_samples:start] *= fade_out

            # Apply fade in after static
            if end + fade_samples <= len(clean_audio):
                fade_in = np.linspace(0, 1, fade_samples)
                clean_audio[end:end + fade_samples] *= fade_in

            # Zero out static region
            clean_audio[start:end] = 0

        # Remove zero regions
        non_zero = clean_audio != 0
        clean_audio = clean_audio[non_zero]

        return clean_audio

    def enhance_speech(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        """
        Minimal speech enhancement that preserves original quality.
        """
        n_fft = 2048
        hop_length = 512

        stft_matrix = librosa.stft(
            audio_data,
            n_fft=n_fft,
            hop_length=hop_length,
            window='hann',
            center=True
        )

        magnitude = np.abs(stft_matrix)
        phase = np.angle(stft_matrix)

        # Noise floor estimation
        noise_floor = np.percentile(magnitude, 15, axis=1, keepdims=True)

        # Create soft mask
        gain = np.maximum(1 - noise_floor / (magnitude + 1e-10), 0)
        gain = gain ** 0.7

        # Temporal smoothing
        window_size = 3
        smoothing_kernel = np.hanning(window_size)[:, np.newaxis] / np.sum(np.hanning(window_size))
        gain = np.apply_along_axis(
            lambda x: np.convolve(x, smoothing_kernel.flatten(), mode='same'),
            1,
            gain
        )

        # Preserve speech frequencies (300-3400 Hz)
        speech_range = (int(300 * n_fft / sr), int(3400 * n_fft / sr))
        gain[speech_range[0]:speech_range[1], :] = np.maximum(
            gain[speech_range[0]:speech_range[1], :],
            0.8
        )

        # Apply gain and reconstruct
        enhanced_stft = magnitude * gain * np.exp(1j * phase)

        return librosa.istft(
            enhanced_stft,
            hop_length=hop_length,
            window='hann',
            center=True
        )

    async def process_audio(
            self,
            input_path: str,
            request_id: str,
            sample_rate: int = 22050,
            mp3_bitrate: str = '8k'
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Process audio file using enhanced algorithms
        
        Args:
            input_path: Path to input audio file
            request_id: Unique request ID for the call
            sample_rate: Target sample rate in Hz
            mp3_bitrate: Output MP3 bitrate
            
        Returns:
            Tuple of (success, processed_s3_key, error_message)
        """
        try:
            logger.info(f"Processing audio file: {input_path}")
            logger.info(f"Using sample rate: {sample_rate} Hz")

            # Load and convert audio
            audio_data, sr = load_audio(input_path, sr=sample_rate, mono=True, convert=True)

            # Validate audio data
            if audio_data.size == 0:
                return False, None, "Input audio file is empty"
            if not np.isfinite(audio_data).all():
                return False, None, "Input audio contains invalid values (inf or nan)"

            # Remove static noise (first pass)
            logger.info("Removing static noise...")
            audio_data = self.remove_static_segments(audio_data, sr)

            # Multi-pass silence removal
            logger.info("Removing silence (initial pass)...")
            audio_data = self.initial_detector.remove_silence(audio_data, sr)

            logger.info("Removing silence (aggressive pass)...")
            audio_data = self.aggressive_detector.remove_silence(audio_data, sr)

            logger.info("Removing silence (standard pass)...")
            audio_data = self.standard_detector.remove_silence(audio_data, sr)

            # Apply speech enhancement
            logger.info("Applying speech enhancement...")
            audio_data = self.enhance_speech(audio_data, sr)

            # Final cleanup
            logger.info("Final cleanup...")
            audio_data = self.remove_static_segments(audio_data, sr)
            audio_data = self.standard_detector.remove_silence(audio_data, sr)

            # Normalize audio
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / (max_val + 1e-10)
                audio_data *= 0.95

            # Create temporary file for processed audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp_path = tmp.name
                save_audio(audio_data, tmp_path, sr, mp3_bitrate=mp3_bitrate)
            return True, tmp_path, None
        except Exception as e:
            error_msg = f"Error processing audio: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg
