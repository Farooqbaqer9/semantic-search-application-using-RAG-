import torch
from speechbrain.inference import EncoderClassifier
import numpy as np
import os
import tempfile
import shutil

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
            print(f"[SpeechBrain] Successfully extracted embedding for {speaker_id}, shape: {embedding.shape}")
            return embedding
        else:
            print(f"[SpeechBrain] Failed to extract embedding for {speaker_id}")
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
        """Extract embedding with comprehensive error handling and path fixes"""
        try:
            # Normalize and validate the path
            audio_path = os.path.normpath(str(audio_path))
            
            if not os.path.exists(audio_path):
                print(f"[SpeechBrain] Audio file not found: {audio_path}")
                return None
                
            # Check if file is readable
            try:
                with open(audio_path, 'rb') as f:
                    f.read(1)
            except Exception as e:
                print(f"[SpeechBrain] Cannot read audio file: {e}")
                return None
            
            # Create a temporary copy in a simple location to avoid path issues
            temp_dir = tempfile.gettempdir()
            temp_filename = f"speechbrain_temp_{os.getpid()}_{hash(audio_path) % 10000}.wav"
            temp_path = os.path.join(temp_dir, temp_filename)
            
            try:
                # Copy the file to temp location
                shutil.copy2(audio_path, temp_path)
                print(f"[SpeechBrain] Created temp copy: {temp_path}")
                
                # Extract embedding from temp file
                signal = self.classifier.load_audio(temp_path)
                emb = self.classifier.encode_batch(signal)
                emb_np = emb.squeeze().cpu().numpy()
                
                print(f"[SpeechBrain] Successfully extracted embedding, shape: {emb_np.shape}")
                return emb_np
                
            except Exception as e:
                print(f"[SpeechBrain] Error during embedding extraction: {e}")
                import traceback
                traceback.print_exc()
                return None
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                        print(f"[SpeechBrain] Cleaned up temp file: {temp_path}")
                    except Exception as cleanup_e:
                        print(f"[SpeechBrain] Could not clean up temp file: {cleanup_e}")
                        
        except Exception as e:
            print(f"[SpeechBrain] Unexpected error in embedding extraction: {e}")
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