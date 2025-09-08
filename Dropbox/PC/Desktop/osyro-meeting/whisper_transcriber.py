import whisper
import torch
import os
import numpy as np
import librosa
import soundfile as sf
import tempfile

class WhisperTranscriber:
    def __init__(self, model_size='small'):
        """Initialize Whisper model with GPU support and enhanced processing"""
        self.model_size = model_size
        self.model = None
        self.device = self._get_best_device()
        self.load_model()
        
    def _get_best_device(self):
        """Determine the best device (GPU if available)"""
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🚀 CUDA GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
            return device
        else:
            print("💻 Using CPU (CUDA not available)")
            return "cpu"
        
    def load_model(self):
        """Load the Whisper model with GPU support"""
        try:
            print(f"[Whisper] Loading '{self.model_size}' model on {self.device}...")
            
            # Load model with device specification
            self.model = whisper.load_model(self.model_size, device=self.device)
            
            # Verify model is on correct device
            if hasattr(self.model, 'encoder'):
                actual_device = next(self.model.encoder.parameters()).device
                print(f"✅ Whisper '{self.model_size}' model loaded on {actual_device}")
            else:
                print(f"✅ Whisper '{self.model_size}' model loaded successfully")
                
        except Exception as e:
            print(f"❌ Failed to load Whisper '{self.model_size}' model: {e}")
            # Re-raise the exception so the app.py can try fallback models
            raise e

    def detect_language_and_speech(self, audio_path):
        """Detect language and check for actual speech content"""
        try:
            # Quick language detection using a small portion
            audio_for_detection = whisper.load_audio(audio_path)
            audio_for_detection = whisper.pad_or_trim(audio_for_detection)
            
            # Make log-Mel spectrogram with error handling
            try:
                mel = whisper.log_mel_spectrogram(audio_for_detection).to(self.model.device)
                
                # Detect language with proper error handling
                _, probs = self.model.detect_language(mel)
                detected_language = max(probs, key=probs.get)
                confidence = probs[detected_language]
                
                # Check for speech vs non-speech based on confidence
                has_speech = confidence > 0.1 and detected_language != 'unknown'
                
                print(f"[Whisper] Language detection: {detected_language} (confidence: {confidence:.3f})")
                
                return detected_language, confidence, has_speech
                
            except Exception as mel_error:
                print(f"[Whisper] Mel spectrogram error: {mel_error}")
                # Fallback to English for stability
                return "en", 0.5, True
            
        except Exception as e:
            print(f"[Whisper] Language detection error: {e}")
            return "en", 0.5, True  # Default fallback

    def filter_repetitive_patterns(self, text):
        """Subtly filter extreme repetitive patterns while preserving natural speech"""
        if not text or len(text.strip()) < 2:
            return "[No speech detected]"
        
        text = text.strip()
        
        # Only filter very extreme repetitive patterns
        words = text.split()
        if len(words) > 10:  # Only check longer texts
            # Count word frequency
            word_counts = {}
            for word in words:
                clean_word = word.lower().strip('.,!?')
                if len(clean_word) > 1:  # Ignore single characters and short words
                    word_counts[clean_word] = word_counts.get(clean_word, 0) + 1
            
            if word_counts:
                most_frequent_word = max(word_counts, key=word_counts.get)
                most_frequent_count = word_counts[most_frequent_word]
                
                # Only filter if one word appears more than 85% of the time (very extreme)
                if most_frequent_count / len(words) > 0.85:
                    return "[Repetitive audio pattern detected]"
        
        # Check for obvious character repetition patterns
        if len(text) > 20:
            # Look for same character repeated 8+ times consecutively
            for i in range(len(text) - 7):
                char = text[i]
                if char != ' ' and all(text[j] == char for j in range(i, min(i + 8, len(text)))):
                    return "[Repetitive speech pattern]"
        
        return text

    def transcribe(self, audio_path, language=None):
        """Enhanced transcription with robust error handling and CPU fallback"""
        if self.model is None:
            print("[Whisper] Model not loaded")
            return "[Whisper model not available]"
            
        try:
            # Normalize path
            audio_path = os.path.normpath(str(audio_path))
            
            if not os.path.exists(audio_path):
                return "[Audio file not found]"
                
            # Check file size
            file_size = os.path.getsize(audio_path)
            if file_size < 1000:  # Less than 1KB
                return "[Audio file too small]"
            
            # Enhanced preprocessing with speech detection
            audio, status = self.preprocess_audio_with_speech_detection(audio_path)
            if audio is None:
                return f"[{status}]"
            
            # Language and speech detection with error handling
            try:
                detected_lang, lang_confidence, has_speech = self.detect_language_and_speech(audio_path)
                
                if not has_speech:
                    return "[No speech detected]"
            except Exception as lang_error:
                print(f"[Whisper] Language detection failed: {lang_error}")
                detected_lang, lang_confidence, has_speech = "en", 0.5, True
            
            # Create temp file for transcription
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
                sf.write(temp_path, audio, 16000)
            
            try:
                # Conservative transcription options for stability
                transcribe_options = {
                    'task': 'transcribe',
                    'fp16': False,                  # Disable FP16 to avoid tensor issues
                    'temperature': 0.0,             # Deterministic output
                    'beam_size': 1,                 # Single beam for stability
                    'best_of': 1,                   # Single attempt
                    'patience': 1.0,
                    'no_speech_threshold': 0.6,     # Speech detection threshold
                    'logprob_threshold': -1.0,      # Quality threshold
                    'compression_ratio_threshold': 2.4,  # Repetition detection
                    'condition_on_previous_text': False,  # Fresh context
                    'verbose': False
                }
                
                # Language-specific handling (KEEP INTACT)
                if language:
                    transcribe_options['language'] = language
                elif detected_lang in ['en', 'ur', 'ar']:
                    # Use detected language for native scripts
                    transcribe_options['language'] = detected_lang
                    print(f"[Whisper] Using native language: {detected_lang}")
                else:
                    # Force English for Roman transliteration
                    transcribe_options['language'] = 'en'
                    print(f"[Whisper] Using English for Roman transliteration of {detected_lang}")
                
                # Perform transcription with GPU error handling
                print(f"[Whisper] Transcribing on {self.device} with {self.model_size} model...")
                
                try:
                    result = self.model.transcribe(temp_path, **transcribe_options)
                except Exception as gpu_error:
                    # GPU fallback to CPU if tensor errors occur
                    if self.device == "cuda" and ("tensor" in str(gpu_error).lower() or "dimension" in str(gpu_error).lower()):
                        print(f"[Whisper] GPU tensor error, trying CPU fallback...")
                        # Clear GPU memory
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        # Load CPU model for this transcription
                        cpu_model = whisper.load_model(self.model_size, device="cpu")
                        result = cpu_model.transcribe(temp_path, **transcribe_options)
                        print(f"[Whisper] ✅ CPU fallback successful")
                    else:
                        raise gpu_error
                
                # Extract and validate text
                text = result.get('text', '').strip()
                if not text:
                    return "[No speech detected]"
                
                # Apply subtle filtering for extreme repetitive patterns
                filtered_text = self.filter_repetitive_patterns(text)
                
                # Log successful transcription
                if not filtered_text.startswith('['):
                    print(f"[Whisper] ✅ Transcribed ({detected_lang}): '{filtered_text[:50]}{'...' if len(filtered_text) > 50 else ''}'")
                
                return filtered_text
                
            finally:
                try:
                    os.unlink(temp_path)
                except:
                    pass
                    
        except Exception as e:
            error_msg = str(e)
            print(f"[Whisper] ❌ Transcription error: {error_msg}")
            
            # Specific error handling with more details
            if "key.size" in error_msg or "tensor" in error_msg.lower():
                return "[GPU tensor mismatch - audio format issue]"
            elif "reshape" in error_msg.lower():
                return "[Audio processing error - invalid dimensions]"
            elif "CUDA" in error_msg or "GPU" in error_msg:
                return "[GPU memory issue]"
            elif "dimension" in error_msg.lower():
                return "[Audio format incompatible]"
            else:
                return f"[Transcription error: {error_msg[:50]}]"

    def preprocess_audio_with_speech_detection(self, audio_path):
        """Enhanced audio preprocessing with speech detection"""
        try:
            # Load audio using librosa for better compatibility
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            if len(audio) == 0:
                return None, "Empty audio file"
            
            # Basic energy-based silence detection
            energy = np.sqrt(np.mean(audio**2))
            if energy < 0.001:  # Very quiet threshold
                return None, "Silent audio detected"
            
            # Spectral analysis for speech-like content
            # Check if there's reasonable spectral diversity (not pure noise or tones)
            try:
                stft = librosa.stft(audio, n_fft=512, hop_length=256)
                spectral_centroids = librosa.feature.spectral_centroid(S=np.abs(stft))[0]
                
                # Speech typically has varied spectral centroid
                if np.std(spectral_centroids) < 100:  # Too monotonic
                    return None, "Non-speech audio pattern"
                    
            except:
                pass  # If spectral analysis fails, continue anyway
            
            # Smart length handling
            min_length = 16000  # 1 second minimum for good recognition
            max_length = 16000 * 30  # 30 seconds maximum
            
            if len(audio) < min_length:
                # Pad with silence instead of rejecting
                padding = min_length - len(audio)
                audio = np.pad(audio, (0, padding), mode='constant', constant_values=0)
                print(f"[Whisper] Padded audio to {min_length/16000:.1f}s")
            elif len(audio) > max_length:
                # Trim but preserve speech
                audio = audio[:max_length]
                print(f"[Whisper] Trimmed audio to {max_length/16000:.1f}s")
            
            return audio, "OK"
            
        except Exception as e:
            return None, f"Preprocessing error: {e}"

    def transcribe_audio(self, audio_path):
        """Alias for backward compatibility"""
        return self.transcribe(audio_path)