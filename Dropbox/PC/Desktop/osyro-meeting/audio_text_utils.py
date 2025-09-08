import wave
import audioop
import numpy as np
import contextlib
import librosa
import subprocess
import os
import tempfile
import soundfile as sf

def normalize_audio(audio, sample_rate):
    """
    Normalize audio to -1.0 to 1.0 range with better handling
    """
    if len(audio) == 0:
        return audio
        
    max_amp = np.max(np.abs(audio))
    if max_amp == 0:
        return audio
        
    # Normalize to prevent clipping
    audio = audio / max_amp
    
    # Apply soft limiting to prevent harsh clipping
    audio = np.tanh(audio * 0.9)  # Soft limiter
    
    # Boost quiet audio more intelligently
    rms_level = np.sqrt(np.mean(audio**2))
    target_rms = 0.3  # Target RMS level
    
    if rms_level < target_rms and rms_level > 0:
        gain = min(target_rms / rms_level, 4.0)  # Limit max gain
        print(f"[normalize_audio] Applying gain: {gain:.2f}x (RMS: {rms_level:.4f} -> {target_rms:.4f})")
        audio = audio * gain
        
    # Final safety clipping
    audio = np.clip(audio, -0.99, 0.99)
    
    return audio

def remove_silence(audio, sample_rate, threshold_db=-30):
    """
    Remove silence from beginning and end of audio using energy-based detection
    """
    if len(audio) == 0:
        return audio
        
    try:
        # Convert to dB scale
        audio_db = librosa.amplitude_to_db(np.abs(audio), ref=np.max)
        
        # Find non-silent regions
        non_silent = audio_db > threshold_db
        
        if not np.any(non_silent):
            print(f"[remove_silence] All audio below threshold {threshold_db} dB")
            return audio  # Return original if all is considered silence
            
        # Find start and end of non-silent audio
        non_silent_indices = np.where(non_silent)[0]
        start_idx = max(0, non_silent_indices[0] - int(0.1 * sample_rate))  # Add 0.1s padding
        end_idx = min(len(audio), non_silent_indices[-1] + int(0.1 * sample_rate))  # Add 0.1s padding
        
        trimmed = audio[start_idx:end_idx]
        
        print(f"[remove_silence] Trimmed from {len(audio)} to {len(trimmed)} samples")
        print(f"[remove_silence] Removed {start_idx} samples from start, {len(audio) - end_idx} from end")
        
        return trimmed
        
    except Exception as e:
        print(f"[remove_silence] Error: {e}, returning original audio")
        return audio

def enhance_audio_for_speech(audio, sample_rate):
    """
    Enhance audio specifically for speech recognition
    """
    if len(audio) == 0:
        return audio
        
    try:
        # Apply pre-emphasis filter (common in speech processing)
        pre_emphasis = 0.97
        audio_emphasized = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
        
        # Apply gentle low-pass filter to remove high-frequency noise
        from scipy import signal
        # Butterworth filter: cutoff at 8000 Hz (above typical speech range)
        nyquist = sample_rate / 2
        if nyquist > 8000:
            cutoff = 8000 / nyquist
            b, a = signal.butter(4, cutoff, btype='low')
            audio_filtered = signal.filtfilt(b, a, audio_emphasized)
        else:
            audio_filtered = audio_emphasized
            
        # Gentle noise gate
        rms = np.sqrt(np.mean(audio_filtered**2))
        noise_gate_threshold = rms * 0.05  # 5% of RMS level
        
        audio_gated = np.where(np.abs(audio_filtered) > noise_gate_threshold, 
                              audio_filtered, 
                              audio_filtered * 0.1)  # Don't completely silence, just reduce
        
        return audio_gated
        
    except ImportError:
        print("[enhance_audio_for_speech] scipy not available, skipping filtering")
        return audio
    except Exception as e:
        print(f"[enhance_audio_for_speech] Error: {e}, returning original audio")
        return audio

def preprocess_audio_for_ml(audio_path, target_sr=16000):
    """
    Comprehensive audio preprocessing for machine learning models
    """
    try:
        print(f"[preprocess_audio_for_ml] Processing: {audio_path}")
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        original_length = len(audio)
        
        print(f"[preprocess_audio_for_ml] Loaded: {original_length} samples at {sr} Hz")
        
        if original_length == 0:
            print("[preprocess_audio_for_ml] Empty audio file")
            return None
            
        # Check minimum length (at least 0.5 seconds)
        min_length = int(0.5 * sr)
        if original_length < min_length:
            print(f"[preprocess_audio_for_ml] Audio too short: {original_length} < {min_length}")
            # Pad with silence if too short
            padding_needed = min_length - original_length
            audio = np.pad(audio, (0, padding_needed), mode='constant')
            print(f"[preprocess_audio_for_ml] Padded to {len(audio)} samples")
        
        # Remove silence from ends
        audio = remove_silence(audio, sr)
        
        # Check if still long enough after silence removal
        if len(audio) < min_length // 2:  # At least 0.25 seconds after trimming
            print("[preprocess_audio_for_ml] Audio too short after silence removal")
            # Use original audio if trimming made it too short
            audio, _ = librosa.load(audio_path, sr=target_sr, mono=True)
            
        # Normalize audio
        audio = normalize_audio(audio, sr)
        
        # Enhance for speech
        audio = enhance_audio_for_speech(audio, sr)
        
        # Final length check
        final_length = len(audio)
        print(f"[preprocess_audio_for_ml] Final: {final_length} samples ({final_length/sr:.2f}s)")
        
        return audio, sr
        
    except Exception as e:
        print(f"[preprocess_audio_for_ml] Error processing {audio_path}: {e}")
        return None

def save_processed_audio(audio, sample_rate, output_path):
    """
    Save processed audio to file
    """
    try:
        # Ensure audio is in the right format
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
            
        sf.write(output_path, audio, sample_rate, subtype='PCM_16')
        print(f"[save_processed_audio] Saved to: {output_path}")
        return True
    except Exception as e:
        print(f"[save_processed_audio] Error saving to {output_path}: {e}")
        return False

def clean_transcript(text):
    """
    Enhanced transcript cleaning
    """
    if not text or isinstance(text, str) and text.startswith('['):
        return text
        
    # Basic cleaning
    text = text.strip()
    
    # Remove multiple spaces
    text = ' '.join(text.split())
    
    # Fix common transcription artifacts
    replacements = {
        ' .': '.',
        ' ,': ',',
        ' ?': '?',
        ' !': '!',
        '  ': ' ',
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Capitalize first letter if needed
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
        
    # Ensure sentence ends with punctuation
    if text and not text[-1] in '.!?':
        text += '.'
    
    return text

def convert_webm_to_wav(webm_path, wav_path):
    """
    Convert .webm audio file to .wav using ffmpeg with better error handling
    """
    if not os.path.exists(webm_path):
        print(f"[convert_webm_to_wav] Input file not found: {webm_path}")
        return False
        
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(wav_path), exist_ok=True)
        
        command = [
            'ffmpeg', '-y', '-i', webm_path,
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',      # Mono
            '-f', 'wav',     # WAV format
            '-acodec', 'pcm_s16le',  # 16-bit PCM
            wav_path
        ]
        
        print(f"[convert_webm_to_wav] Converting: {webm_path} -> {wav_path}")
        
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True,
            timeout=30  # 30 second timeout
        )
        
        if result.returncode == 0 and os.path.exists(wav_path):
            print(f"[convert_webm_to_wav] Conversion successful")
            print(f"[convert_webm_to_wav] Output size: {os.path.getsize(wav_path)} bytes")
            return True
        else:
            print(f"[convert_webm_to_wav] FFmpeg failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("[convert_webm_to_wav] FFmpeg timeout")
        return False
    except FileNotFoundError:
        print("[convert_webm_to_wav] FFmpeg not found")
        return False
    except Exception as e:
        print(f"[convert_webm_to_wav] Error: {e}")
        return False

def get_audio_info(audio_path):
    """
    Get information about an audio file
    """
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        duration = len(audio) / sr
        
        return {
            "duration": duration,
            "sample_rate": sr,
            "samples": len(audio),
            "channels": 1,  # librosa loads as mono by default
            "max_amplitude": float(np.max(np.abs(audio))),
            "rms_level": float(np.sqrt(np.mean(audio**2))),
            "file_size": os.path.getsize(audio_path)
        }
    except Exception as e:
        print(f"[get_audio_info] Error analyzing {audio_path}: {e}")
        return None