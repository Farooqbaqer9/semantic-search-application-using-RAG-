import torch
from speechbrain.inference import EncoderClassifier
import numpy as np
import os
import tempfile
import shutil
import librosa
import soundfile as sf

class SpeechBrainSpeakerRecognizer:
    def __init__(self, model_source="speechbrain/spkrec-ecapa-voxceleb"):
        try:
            self.classifier = EncoderClassifier.from_hparams(source=model_source)
            print("[SpeechBrain] Model loaded successfully")
        except Exception as e:
            print(f"[SpeechBrain] Error loading model: {e}")
            self.classifier = None

    def enroll_speaker(self, speaker_id, audio_path):
        """Enroll a speaker with proper path handling"""
        if self.classifier is None:
            print("[SpeechBrain] Model not loaded")
            return None
            
        # Convert to string and normalize path
        audio_path = os.path.normpath(str(audio_path))
        
        print(f"[SpeechBrain] Enrolling speaker {speaker_id} with audio: {audio_path}")
        print(f"[SpeechBrain] File exists: {os.path.exists(audio_path)}")
        print(f"[SpeechBrain] File size: {os.path.getsize(audio_path) if os.path.exists(audio_path) else 'N/A'} bytes")
        
        if not os.path.exists(audio_path):
            print(f"[SpeechBrain] Audio file not found: {audio_path}")
            return None
            
        # Check file size
        if os.path.getsize(audio_path) < 1000:  # Less than 1KB
            print(f"[SpeechBrain] Audio file too small: {os.path.getsize(audio_path)} bytes")
            return None
            
        embedding = self._extract_embedding_safe(audio_path)
        if embedding is not None:
            print(f"[SpeechBrain] ✅ Successfully extracted embedding for {speaker_id}")
            print(f"[SpeechBrain] Embedding shape: {embedding.shape}")
            print(f"[SpeechBrain] Embedding dtype: {embedding.dtype}")
            print(f"[SpeechBrain] Embedding min/max: {embedding.min():.4f}/{embedding.max():.4f}")
            print(f"[SpeechBrain] Embedding mean: {embedding.mean():.4f}")
            return embedding
        else:
            print(f"[SpeechBrain] ❌ Failed to extract embedding for {speaker_id}")
            return None

    def identify_speaker(self, audio_path, external_embeddings):
        """Identify speaker with proper error handling"""
        if self.classifier is None:
            print("[SpeechBrain] Model not loaded")
            return None, 0.0
            
        # Convert to string and normalize path
        audio_path = os.path.normpath(str(audio_path))
        
        print(f"[SpeechBrain] Identifying speaker from: {audio_path}")
        print(f"[SpeechBrain] File exists: {os.path.exists(audio_path)}")
        
        if not os.path.exists(audio_path):
            print(f"[SpeechBrain] Audio file not found: {audio_path}")
            return None, 0.0
            
        if not external_embeddings:
            print("[SpeechBrain] No enrolled speakers available")
            return None, 0.0
            
        test_emb = self._extract_embedding_safe(audio_path)
        if test_emb is None:
            print("[SpeechBrain] Failed to extract test embedding")
            return None, 0.0
            
        best_id = None
        best_score = -1.0
        
        print(f"[SpeechBrain] Comparing against {len(external_embeddings)} enrolled speakers")
        
        for speaker_id, emb in external_embeddings.items():
            try:
                emb = np.array(emb)
                score = self._cosine_similarity(test_emb, emb)
                print(f"[SpeechBrain] Speaker {speaker_id}: similarity = {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_id = speaker_id
            except Exception as e:
                print(f"[SpeechBrain] Error comparing with speaker {speaker_id}: {e}")
                continue
                
        print(f"[SpeechBrain] Best match: {best_id} with score {best_score:.4f}")
        return best_id, best_score

    def _extract_embedding_safe(self, audio_path):
        """Extract embedding with simplified, robust approach"""
        try:
            # Normalize and validate the path
            audio_path = os.path.normpath(str(audio_path))
            
            if not os.path.exists(audio_path):
                print(f"[SpeechBrain] Audio file not found: {audio_path}")
                return None
                
            print(f"[SpeechBrain] Processing audio file: {audio_path}")
            print(f"[SpeechBrain] File size: {os.path.getsize(audio_path)} bytes")
            
            # Use librosa to load and preprocess audio properly
            print(f"[SpeechBrain] Loading audio with librosa...")
            audio_data, sr = librosa.load(audio_path, sr=16000, mono=True)
            print(f"[SpeechBrain] Librosa loaded: shape={audio_data.shape}, sr={sr}")
            
            # Ensure minimum length (3 seconds for good speaker recognition)
            min_length = 16000 * 3  # 3 seconds at 16kHz
            if len(audio_data) < min_length:
                print(f"[SpeechBrain] Audio too short ({len(audio_data)/16000:.2f}s), padding to 3s")
                audio_data = np.pad(audio_data, (0, min_length - len(audio_data)), mode='constant')
            
            # Limit maximum length (30 seconds)
            max_length = 16000 * 30
            if len(audio_data) > max_length:
                print(f"[SpeechBrain] Audio too long ({len(audio_data)/16000:.2f}s), truncating to 30s")
                audio_data = audio_data[:max_length]
            
            # Normalize audio
            audio_data = librosa.util.normalize(audio_data)
            
            # Convert to torch tensor directly
            print(f"[SpeechBrain] Converting to torch tensor...")
            signal = torch.tensor(audio_data, dtype=torch.float32)
            
            # Ensure proper shape for SpeechBrain (batch_size=1, samples)
            if signal.dim() == 1:
                signal = signal.unsqueeze(0)  # Add batch dimension
            
            print(f"[SpeechBrain] Signal tensor shape: {signal.shape}")
            print(f"[SpeechBrain] Signal dtype: {signal.dtype}")
            
            # Extract embedding directly without file I/O
            print(f"[SpeechBrain] Encoding signal to embedding...")
            with torch.no_grad():  # Disable gradient computation for inference
                emb = self.classifier.encode_batch(signal)
                print(f"[SpeechBrain] Raw embedding tensor shape: {emb.shape}")
                
                # Convert to numpy
                emb_np = emb.squeeze().cpu().numpy()
                print(f"[SpeechBrain] Final embedding numpy shape: {emb_np.shape}")
                print(f"[SpeechBrain] Embedding dtype: {emb_np.dtype}")
                print(f"[SpeechBrain] Embedding stats - min: {emb_np.min():.4f}, max: {emb_np.max():.4f}, mean: {emb_np.mean():.4f}")
                
                # Validate embedding
                if emb_np.size == 0:
                    print(f"[SpeechBrain] ❌ Empty embedding generated")
                    return None
                if np.any(np.isnan(emb_np)):
                    print(f"[SpeechBrain] ❌ NaN values in embedding")
                    return None
                if np.any(np.isinf(emb_np)):
                    print(f"[SpeechBrain] ❌ Infinite values in embedding")
                    return None
                
                print(f"[SpeechBrain] ✅ Valid embedding extracted, shape: {emb_np.shape}")
                return emb_np
                
        except Exception as e:
            print(f"[SpeechBrain] Error in embedding extraction: {e}")
            print(f"[SpeechBrain] Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return None

    def _cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors"""
        try:
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
                
            return np.dot(a, b) / (norm_a * norm_b)
        except Exception as e:
            print(f"[SpeechBrain] Error calculating cosine similarity: {e}")
            return 0.0