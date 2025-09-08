import whisper
import os
import numpy as np
import librosa
import soundfile as sf
import tempfile

class WhisperTranscriber:
    def __init__(self, model_size='base'):
        """Initialize with base model for better accuracy"""
        try:
            print(f"[Whisper] Loading {model_size} model...")
            self.model = whisper.load_model(model_size)
            print(f"[Whisper] Model loaded successfully")
        except Exception as e:
            print(f"[Whisper] Error loading model: {e}")
            # Fallback to tiny model
            try:
                print("[Whisper] Falling back to tiny model...")
                self.model = whisper.load_model('tiny')
                print("[Whisper] Tiny model loaded")
            except Exception as e2:
                print(f"[Whisper] Failed to load any model: {e2}")
                self.model = None

    def transcribe(self, audio_path, language=None):
        """Transcribe audio with preprocessing for better accuracy"""
        if self.model is None:
            print("[Whisper] Model not loaded")
            return "[Whisper model not available]"
            
        try:
            # Normalize path
            audio_path = os.path.normpath(str(audio_path))
            
            print(f"[Whisper] Transcribing: {audio_path}")
            print(f"[Whisper] File exists: {os.path.exists(audio_path)}")
            
            if not os.path.exists(audio_path):
                return "[Audio file not found]"
                
            # Check file size
            file_size = os.path.getsize(audio_path)
            print(f"[Whisper] File size: {file_size} bytes")
            
            if file_size < 1000:  # Less than 1KB
                return "[Audio file too small]"
                
            # Preprocess audio for better transcription
            processed_path = self._preprocess_audio(audio_path)
            
            try:
                # Transcription options for better accuracy
                options = {
                    'task': 'transcribe',
                    'fp16': False,  # Use fp32 for better accuracy
                    'temperature': 0.0,  # Deterministic output
                    'best_of': 5,  # Try multiple decodings
                    'beam_size': 5,  # Use beam search
                    'patience': 1.0,
                    'length_penalty': 1.0,
                    'suppress_tokens': "-1",  # Don't suppress any tokens
                    'initial_prompt': None,
                    'condition_on_previous_text': True,
                    'compression_ratio_threshold': 2.4,
                    'logprob_threshold': -1.0,
                    'no_speech_threshold': 0.6
                }
                
                # Set language if specified
                if language:
                    allowed_languages = ['en', 'hi', 'ur', 'ar', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh']
                    if language in allowed_languages:
                        options['language'] = language
                    else:
                        print(f"[Whisper] Language {language} not in allowed list, using auto-detection")
                
                result = self.model.transcribe(processed_path, **options)
                
                # Extract text and clean it
                text = result.get('text', '').strip()
                
                # Log detected language
                detected_lang = result.get('language', 'unknown')
                print(f"[Whisper] Detected language: {detected_lang}")
                
                # Log confidence if available
                if 'segments' in result:
                    avg_confidence = np.mean([seg.get('avg_logprob', 0) for seg in result['segments']]) if result['segments'] else 0
                    print(f"[Whisper] Average confidence: {avg_confidence:.4f}")
                
                if not text:
                    return "[No speech detected]"
                    
                print(f"[Whisper] Transcription result: '{text}'")
                return text
                
            finally:
                # Clean up processed file if it's different from original
                if processed_path != audio_path and os.path.exists(processed_path):
                    try:
                        os.remove(processed_path)
                        print(f"[Whisper] Cleaned up processed file: {processed_path}")
                    except Exception as e:
                        print(f"[Whisper] Could not clean up processed file: {e}")
                        
        except Exception as e:
            print(f"[Whisper] Transcription error: {e}")
            import traceback
            traceback.print_exc()
            return f"[Transcription error: {str(e)}]"

    def _preprocess_audio(self, audio_path):
        """Preprocess audio for better transcription accuracy"""
        try:
            print(f"[Whisper] Preprocessing audio: {audio_path}")
            
            # Load audio with librosa
            y, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            print(f"[Whisper] Original audio: {len(y)} samples at {sr} Hz")
            
            # Check if audio is too short
            if len(y) < sr * 0.5:  # Less than 0.5 seconds
                print("[Whisper] Audio too short for reliable transcription")
                return audio_path  # Return original path
                
            # Normalize volume
            if np.max(np.abs(y)) > 0:
                y = y / np.max(np.abs(y)) * 0.8  # Normalize to 80% max volume
                
            # Remove silence at beginning and end
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)
            
            # Check if trimmed audio is not too short
            if len(y_trimmed) < sr * 0.3:  # Less than 0.3 seconds after trimming
                y_trimmed = y  # Use original if trimming made it too short
                
            print(f"[Whisper] Processed audio: {len(y_trimmed)} samples")
            
            # Apply noise reduction if the audio is very noisy
            # Simple spectral gating
            stft = librosa.stft(y_trimmed)
            magnitude = np.abs(stft)
            
            # Estimate noise floor (bottom 10% of magnitude values)
            noise_floor = np.percentile(magnitude, 10)
            
            # Create a soft mask
            mask = magnitude / (magnitude + noise_floor * 2)
            
            # Apply mask
            stft_clean = stft * mask
            y_clean = librosa.istft(stft_clean)
            
            # Ensure the cleaned audio isn't too quiet
            if np.max(np.abs(y_clean)) < 0.01:
                y_clean = y_trimmed  # Fall back to trimmed version
                
            # Save processed audio to temporary file
            temp_dir = tempfile.gettempdir()
            temp_filename = f"whisper_processed_{os.getpid()}_{hash(audio_path) % 10000}.wav"
            temp_path = os.path.join(temp_dir, temp_filename)
            
            sf.write(temp_path, y_clean, sr)
            print(f"[Whisper] Saved processed audio to: {temp_path}")
            
            return temp_path
            
        except Exception as e:
            print(f"[Whisper] Error preprocessing audio: {e}")
            # Return original path if preprocessing fails
            return audio_path