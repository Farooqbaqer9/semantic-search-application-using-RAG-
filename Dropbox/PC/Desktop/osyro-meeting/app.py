import os
import uuid
import json
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from supabase import create_client, Client
import shutil
import tempfile
import time


# Load secrets from environment variables (set in .env or cloud dashboard)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

from flask_cors import CORS
from dotenv import load_dotenv
from pathlib import Path

# Import our custom modules
from speechbrain_speaker import SpeechBrainSpeakerRecognizer
 #from transcription_service import TranscriptionService
from audio_text_utils import clean_transcript, convert_webm_to_wav
from whisper_transcriber import WhisperTranscriber

# Load environment variables if any
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configure CORS properly to allow connections from the frontend
FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "https://meeting_roomAI.com")
CORS(app, resources={r"/*": {"origins": FRONTEND_ORIGIN, "methods": ["GET", "POST", "OPTIONS"], 
                            "allow_headers": ["Content-Type", "Authorization"]}})

# Add a global response header middleware
@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

# Initialize services
speechbrain_speaker = SpeechBrainSpeakerRecognizer()
#transcription_service = TranscriptionService()

# Initialize Whisper model with better hierarchy and GPU support
whisper_transcriber = None
try:
    # Try small model first (best for GTX 1650)
    print("🚀 Attempting to load Whisper 'small' model with GPU support...")
    whisper_transcriber = WhisperTranscriber(model_size='small')
    print("✅ Whisper 'small' model loaded successfully")
except Exception as e:
    print(f"❌ Could not load Whisper 'small' model: {e}")
    try:
        # Fallback to base model
        print("🔄 Attempting to load Whisper 'base' model...")
        whisper_transcriber = WhisperTranscriber(model_size='base')
        print("✅ Whisper 'base' model loaded as fallback")
    except Exception as e2:
        print(f"❌ Could not load Whisper 'base' model: {e2}")
        try:
            # Final fallback to tiny model
            print("🔄 Attempting to load Whisper 'tiny' model as final fallback...")
            whisper_transcriber = WhisperTranscriber(model_size='tiny')
            print("✅ Whisper 'tiny' model loaded as final fallback")
        except Exception as e3:
            print(f"❌ Could not load any Whisper model: {e3}")
            whisper_transcriber = None

def create_temp_file(audio_bytes, file_extension='.wav'):
    """Create a temporary file with proper cleanup"""
    try:
        # Use system temp directory
        temp_dir = tempfile.gettempdir()
        
        # Create unique filename
        temp_filename = f"audio_{uuid.uuid4().hex}{file_extension}"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        # Write audio data to temp file
        with open(temp_path, 'wb') as temp_file:
            temp_file.write(audio_bytes)
        
        print(f"[TempFile] Created: {temp_path} ({len(audio_bytes)} bytes)")
        return temp_path
    except Exception as e:
        print(f"[TempFile] Error creating temp file: {e}")
        return None

def cleanup_temp_file(temp_path):
    """Clean up temporary file"""
    if temp_path and os.path.exists(temp_path):
        try:
            os.remove(temp_path)
            print(f"[TempFile] Cleaned up: {temp_path}")
        except Exception as e:
            print(f"[TempFile] Could not clean up {temp_path}: {e}")

@app.route('/')
def index():
    return jsonify({
        "status": "running",
        "message": "Oysro Meeting Room API is running",
        "endpoints": [
            "/api/enroll", 
            "/api/process_meeting",
            "/api/speakers",
            "/api/meetings"
        ]
    })

@app.route('/api/enroll', methods=['POST'])
def enroll_speaker():
    """Enroll a new speaker with voice samples"""
    temp_audio_path = None
    
    try:
        if 'audio' not in request.files or 'name' not in request.form:
            return jsonify({"error": "Missing audio file or name"}), 400

        audio_file = request.files['audio']
        speaker_name = request.form['name']
        speaker_id = request.form.get('speaker_id', str(uuid.uuid4()))

        print(f"[Enrollment] Starting enrollment for speaker: {speaker_name} (ID: {speaker_id})")

        # Read audio data
        audio_bytes = audio_file.read()
        
        if len(audio_bytes) < 1000:  # Less than 1KB
            return jsonify({"error": "Audio file too small or empty"}), 400

        print(f"[Enrollment] Audio file size: {len(audio_bytes)} bytes")

        # Create temporary file for processing
        file_extension = os.path.splitext(audio_file.filename)[1] or '.wav'
        temp_audio_path = create_temp_file(audio_bytes, file_extension)
        
        if not temp_audio_path:
            return jsonify({"error": "Failed to create temporary file"}), 500

        # Upload to Supabase Storage first
        sample_filename = f"enrollment_{speaker_id}_{uuid.uuid4()}.wav"
        
        try:
            storage_upload = supabase.storage.from_('audio').upload(
                sample_filename, 
                audio_bytes, 
                {'content-type': 'audio/wav'}
            )
            print(f"[Enrollment] Uploaded to Supabase Storage: {sample_filename}")
        except Exception as upload_error:
            print(f"[Enrollment] Error uploading to Supabase Storage: {upload_error}")
            return jsonify({"error": f"Failed to upload audio: {upload_error}"}), 500

        # Process with SpeechBrain
        print(f"[Enrollment] Processing with SpeechBrain: {temp_audio_path}")
        embedding = speechbrain_speaker.enroll_speaker(speaker_id, temp_audio_path)
        
        embedding_success = embedding is not None
        print(f"[Enrollment] SpeechBrain embedding success: {embedding_success}")

        # Store speaker and sample metadata in Supabase
        try:
            # Get or create speaker in Supabase
            speaker_result = supabase.table("speakers").select("id").eq("user_id", speaker_id).execute()
            
            if speaker_result.data:
                supabase_speaker_id = speaker_result.data[0]["id"]
                print(f"[Enrollment] Found existing speaker: {supabase_speaker_id}")
            else:
                speaker_data = {
                    "user_id": speaker_id, 
                    "name": speaker_name, 
                    "meta": {"status": "active"}
                }
                result = supabase.table("speakers").insert(speaker_data).execute()
                supabase_speaker_id = result.data[0]["id"] if result.data else None
                print(f"[Enrollment] Created new speaker: {supabase_speaker_id}")

            # Store sample metadata
            if supabase_speaker_id:
                sample_data = {
                    "speaker_id": supabase_speaker_id,
                    "storage_path": sample_filename,
                    "file_size": len(audio_bytes),
                    "mime_type": "audio/wav"
                }
                
                sample_result = supabase.table("speaker_samples").insert(sample_data).execute()
                print(f"[Enrollment] Speaker sample stored: {sample_result}")

                # Store embedding if available
                if embedding is not None:
                    print(f"[DEBUG] Embedding received - Type: {type(embedding)}")
                    print(f"[DEBUG] Embedding shape: {embedding.shape}")
                    print(f"[DEBUG] Embedding dtype: {embedding.dtype}")
                    print(f"[DEBUG] Embedding sample values: {embedding[:5]}")
                    
                    try:
                        # Convert to list
                        print(f"[DEBUG] Converting embedding to list...")
                        embedding_list = embedding.tolist()
                        print(f"[DEBUG] Conversion successful - List length: {len(embedding_list)}")
                        print(f"[DEBUG] First 5 values: {embedding_list[:5]}")
                        
                        embedding_data = {
                            "speaker_id": supabase_speaker_id,
                            "embedding_vector": embedding_list,
                            "feature_type": "speechbrain",
                            "vector_dimension": len(embedding_list)
                        }
                        
                        print(f"[DEBUG] Embedding data prepared: speaker_id={supabase_speaker_id}, dimension={len(embedding_list)}")
                        print(f"[DEBUG] Attempting to insert into speaker_embeddings table...")
                        
                        emb_result = supabase.table("speaker_embeddings").insert(embedding_data).execute()
                        
                        print(f"[DEBUG] Database insert result: {emb_result}")
                        if emb_result.data:
                            print(f"[Enrollment] ✅ Speaker embedding stored successfully: ID {emb_result.data[0]['id']}")
                        else:
                            print(f"[Enrollment] ❌ No data returned from embedding insert")
                            
                    except Exception as embedding_error:
                        print(f"[Enrollment] ❌ Embedding storage error: {embedding_error}")
                        print(f"[DEBUG] Error type: {type(embedding_error)}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"[Enrollment] ❌ No embedding received from SpeechBrain")

        except Exception as e:
            print(f"[Enrollment] Database error: {e}")
            return jsonify({"error": f"Failed to store speaker data: {e}"}), 500

        return jsonify({
            "success": embedding_success,
            "speaker_id": speaker_id,
            "name": speaker_name,
            "embedding_success": embedding_success,
            "message": "Speaker enrolled successfully" if embedding_success else "Speaker stored but embedding failed"
        })

    except Exception as e:
        print(f"[Enrollment] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Enrollment failed: {str(e)}"}), 500
    
    finally:
        cleanup_temp_file(temp_audio_path)

@app.route('/api/process_meeting', methods=['POST'])
def process_meeting():
    """Process meeting audio for speaker recognition and transcription"""
    temp_audio_path = None
    
    try:
        if 'audio' not in request.files:
            return jsonify({
                "error": "Missing audio file", 
                "transcript": "[No audio file provided]",
                "success": False
            }), 400

        audio_file = request.files['audio']
        meeting_id = request.form.get('meeting_id', str(uuid.uuid4()))
        filename = audio_file.filename or 'audio.wav'
        mimetype = audio_file.mimetype or 'audio/wav'
        
        print(f"[Meeting] Processing audio: {filename}, MIME: {mimetype}, Meeting: {meeting_id}")

        # Read audio data
        audio_bytes = audio_file.read()
        
        if len(audio_bytes) < 1000:  # Less than 1KB
            return jsonify({
                "error": "Audio file too small",
                "transcript": "[Audio file too small]",
                "success": False
            })

        print(f"[Meeting] Audio size: {len(audio_bytes)} bytes")

        # Create temporary file for processing
        file_ext = os.path.splitext(filename)[1] or '.wav'
        temp_audio_path = create_temp_file(audio_bytes, file_ext)
        
        if not temp_audio_path:
            return jsonify({
                "error": "Failed to create temporary file",
                "transcript": "[Processing error]",
                "success": False
            }), 500

        # Upload to Supabase Storage
        chunk_filename = f"meeting_{meeting_id}_{uuid.uuid4()}{file_ext}"
        
        try:
            storage_upload = supabase.storage.from_('audio').upload(
                chunk_filename, 
                audio_bytes, 
                {'content-type': mimetype}
            )
            print(f"[Meeting] Uploaded to storage: {chunk_filename}")
        except Exception as upload_error:
            print(f"[Meeting] Storage upload error: {upload_error}")

        # Speaker Recognition
        speaker_id = None
        speaker_user_id = None
        speaker_name = "Unknown"
        speaker_confidence = 0.0
        
        try:
            print("[Meeting] Starting speaker recognition...")
            
            # Load all embeddings from Supabase
            embeddings_result = supabase.table("speaker_embeddings").select(
                "speaker_id, embedding_vector"
            ).execute()
            
            if embeddings_result.data:
                embeddings = {
                    row["speaker_id"]: row["embedding_vector"] 
                    for row in embeddings_result.data
                }
                print(f"[Meeting] Loaded {len(embeddings)} speaker embeddings")
                
                sb_speaker_id, sb_confidence = speechbrain_speaker.identify_speaker(
                    temp_audio_path, 
                    external_embeddings=embeddings
                )
                
                print(f"[Meeting] Speaker recognition result: {sb_speaker_id}, confidence: {sb_confidence}")
                
                # Lower threshold for recognition (0.4 instead of 0.45)
                if sb_speaker_id and sb_confidence > 0.4:
                    # Get speaker name from database
                    speaker_result = supabase.table("speakers").select(
                        "id, user_id, name"
                    ).eq("id", sb_speaker_id).execute()
                    
                    if speaker_result.data:
                        speaker_data = speaker_result.data[0]
                        speaker_id = speaker_data["id"]  # Use UUID for database FK
                        speaker_user_id = speaker_data["user_id"]  # Keep user_id for display
                        speaker_name = speaker_data["name"]
                        speaker_confidence = float(sb_confidence)
                        print(f"[Meeting] Identified speaker: {speaker_name} ({speaker_user_id})")
                    
            else:
                print("[Meeting] No speaker embeddings found in database")
                
        except Exception as e:
            print(f"[Meeting] Speaker recognition error: {e}")

        # Transcription
        transcript = "[Transcription failed]"
        
        if whisper_transcriber:
            try:
                print("[Meeting] Starting transcription...")
                transcript = whisper_transcriber.transcribe(temp_audio_path)
                print(f"[Meeting] Transcription result: '{transcript}'")
                
                if not transcript or transcript.strip() == "":
                    transcript = "[No speech detected]"
                    
            except Exception as e:
                print(f"[Meeting] Transcription error: {e}")
                transcript = f"[Transcription error: {str(e)}]"
        else:
            print("[Meeting] Whisper transcriber not available")
            transcript = "[Whisper transcriber not available]"

        # Clean transcript
        transcript = clean_transcript(transcript)

        # Store meeting data in Supabase
        try:
            # Ensure meeting exists
            existing_meeting = supabase.table("meetings").select("*").eq("id", meeting_id).execute()
            
            if not existing_meeting.data:
                meeting_record = {
                    "id": meeting_id,
                    "title": f"Meeting {meeting_id[:8]}",
                    "meta": {"status": "active"}
                }
                supabase.table("meetings").insert(meeting_record).execute()
                print(f"[Meeting] Created meeting record: {meeting_id}")

            # Store meeting chunk
            meeting_chunk_data = {
                "meeting_id": meeting_id,
                "chunk_index": 0,
                "timestamp": time.time(),
                "duration": 0.0,
                "storage_path": chunk_filename,
                "file_size": len(audio_bytes),
                "mime_type": mimetype,
                "speaker_id": speaker_id if speaker_id else None
            }
            
            chunk_insert_result = supabase.table("meeting_chunks").insert(meeting_chunk_data).execute()
            meeting_chunk_id = None
            
            if chunk_insert_result.data:
                meeting_chunk_id = chunk_insert_result.data[0].get('id')
                print(f"[Meeting] Stored meeting chunk: {meeting_chunk_id}")

            # Store transcript
            if meeting_chunk_id and transcript and not transcript.startswith("["):
                transcript_data = {
                    "meeting_chunk_id": meeting_chunk_id,
                    "speaker_id": speaker_id,
                    "transcript_text": transcript,
                    "text": transcript,
                    "confidence": speaker_confidence,
                    "timestamp": time.time(),
                    "chunk_index": 0,
                    "meeting_id": meeting_id
                }
                
                transcript_result = supabase.table("transcripts").insert(transcript_data).execute()
                print(f"[Meeting] Stored transcript: {transcript_result}")

        except Exception as db_error:
            print(f"[Meeting] Database storage error: {db_error}")

        # Determine success
        success = transcript and not transcript.startswith("[")
        
        return jsonify({
            "success": success,
            "meeting_id": meeting_id,
            "speaker_id": speaker_user_id,  # Return user_id for frontend
            "speaker_name": speaker_name,
            "speaker_confidence": speaker_confidence,
            "transcript": transcript
        })

    except Exception as e:
        print(f"[Meeting] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": f"Processing failed: {str(e)}",
            "transcript": f"[Error: {str(e)}]",
            "success": False
        }), 500
    
    finally:
        cleanup_temp_file(temp_audio_path)

@app.route('/api/speakers', methods=['GET'])
def get_speakers():
    """Get list of enrolled speakers"""
    try:
        # Get speakers from Supabase
        result = supabase.table("speakers").select("*").execute()
        speakers = []
        
        for speaker in result.data:
            # Count samples for this speaker
            samples_result = supabase.table("speaker_samples").select("id").eq("speaker_id", speaker["id"]).execute()
            sample_count = len(samples_result.data) if samples_result.data else 0
            
            # Count embeddings for this speaker
            embeddings_result = supabase.table("speaker_embeddings").select("id").eq("speaker_id", speaker["id"]).execute()
            embedding_count = len(embeddings_result.data) if embeddings_result.data else 0
            
            speakers.append({
                "id": speaker["user_id"],
                "name": speaker["name"],
                "sample_count": sample_count,
                "embedding_count": embedding_count,
                "status": "enrolled" if embedding_count > 0 else "pending"
            })
        
        print(f"Retrieved {len(speakers)} speakers from Supabase")
        return jsonify({"speakers": speakers})
        
    except Exception as e:
        print(f"Error getting speakers from Supabase: {e}")
        return jsonify({"error": "Failed to fetch speakers"}), 500

@app.route('/api/meetings', methods=['GET'])
def get_meetings():
    """Get list of meetings"""
    try:
        result = supabase.table("meetings").select("*").order("created_at", desc=True).execute()
        meetings = []
        
        for meeting in result.data:
            # Get transcript count for this meeting
            transcript_result = supabase.table("transcripts").select("id").eq("meeting_id", meeting["id"]).execute()
            transcript_count = len(transcript_result.data) if transcript_result.data else 0
            
            # Get chunk count for this meeting
            chunk_result = supabase.table("meeting_chunks").select("id").eq("meeting_id", meeting["id"]).execute()
            chunk_count = len(chunk_result.data) if chunk_result.data else 0
            
            meeting_data = meeting.copy()
            meeting_data["transcript_count"] = transcript_count
            meeting_data["chunk_count"] = chunk_count
            meetings.append(meeting_data)
            
        return jsonify({"meetings": meetings})
    except Exception as e:
        print(f"Error getting meetings: {e}")
        return jsonify({"error": "Failed to fetch meetings"}), 500

@app.route('/api/meeting/<meeting_id>', methods=['GET'])
def get_meeting_details(meeting_id):
    """Get detailed transcript for a meeting"""
    try:
        meeting_result = supabase.table("meetings").select("*").eq("id", meeting_id).execute()
        if not meeting_result.data:
            return jsonify({"error": "Meeting not found"}), 404
            
        meeting_data = meeting_result.data[0]
        
        # Get transcripts with speaker information
        transcripts_query = """
        transcripts.*, 
        speakers.name as speaker_name,
        speakers.user_id as speaker_user_id
        """
        
        transcripts_result = supabase.table("transcripts").select(transcripts_query).eq("meeting_id", meeting_id).order("timestamp").execute()
        
        meeting_data["transcripts"] = transcripts_result.data if transcripts_result.data else []
        
        return jsonify(meeting_data)
    except Exception as e:
        print(f"Error getting meeting details: {e}")
        return jsonify({"error": "Failed to fetch meeting details"}), 500

@app.route('/api/audio/<path:filename>')
def get_audio(filename):
    """Serve audio files from Supabase Storage"""
    try:
        from flask import Response
        
        file_response = supabase.storage.from_('audio').download(filename)
        
        # Determine content type based on file extension
        if filename.endswith('.webm'):
            content_type = 'audio/webm'
        elif filename.endswith('.mp3'):
            content_type = 'audio/mpeg'
        else:
            content_type = 'audio/wav'
            
        return Response(file_response, mimetype=content_type)
    except Exception as e:
        print(f"Error serving audio: {e}")
        return jsonify({"error": "Audio file not found"}), 404

@app.route('/api/end_meeting', methods=['POST'])
def end_meeting():
    """End a meeting"""
    try:
        meeting_id = request.form.get('meeting_id')
        if not meeting_id:
            return jsonify({"error": "Missing meeting_id"}), 400
            
        from datetime import datetime
        ended_at = datetime.utcnow().isoformat()
        
        result = supabase.table("meetings").update({"ended_at": ended_at}).eq("id", meeting_id).execute()
        
        print(f"Meeting {meeting_id} ended at {ended_at}")
        return jsonify({
            "success": True, 
            "meeting_id": meeting_id, 
            "ended_at": ended_at
        })
    except Exception as e:
        print(f"Error ending meeting: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/test_audio', methods=['POST'])
def test_audio():
    """Test endpoint for debugging audio processing"""
    temp_audio_path = None
    
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file"}), 400
            
        audio_file = request.files['audio']
        audio_bytes = audio_file.read()
        
        # Create temp file
        file_ext = os.path.splitext(audio_file.filename)[1] or '.wav'
        temp_audio_path = create_temp_file(audio_bytes, file_ext)
        
        if not temp_audio_path:
            return jsonify({"error": "Failed to create temp file"}), 500
            
        # Test file properties
        file_size = os.path.getsize(temp_audio_path)
        
        # Test SpeechBrain
        sb_result = "N/A"
        try:
            test_embedding = speechbrain_speaker._extract_embedding_safe(temp_audio_path)
            sb_result = f"Success: {test_embedding.shape}" if test_embedding is not None else "Failed"
        except Exception as e:
            sb_result = f"Error: {str(e)}"
            
        # Test Whisper
        whisper_result = "N/A"
        if whisper_transcriber:
            try:
                transcript = whisper_transcriber.transcribe(temp_audio_path)
                whisper_result = f"Success: '{transcript}'"
            except Exception as e:
                whisper_result = f"Error: {str(e)}"
                
        return jsonify({
            "file_size": file_size,
            "temp_path": temp_audio_path,
            "speechbrain_test": sb_result,
            "whisper_test": whisper_result
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        cleanup_temp_file(temp_audio_path)

if __name__ == '__main__':
    print("="*50)
    print("Starting Oysro Meeting Room API Server...")
    print("API available at: http://localhost:5000/api")
    print("Make sure the frontend is also running on http://localhost:8000")
    print("="*50)
    
    # Test Supabase connection
    try:
        result = supabase.table("speakers").select("*").execute()
        print(f"✅ Supabase connected successfully. Found {len(result.data)} speakers in database.")
        
        # Test audio bucket
        try:
            buckets = supabase.storage.list_buckets()
            bucket_names = [bucket.name for bucket in buckets]
            if 'audio' in bucket_names:
                print("✅ Audio storage bucket is available.")
            else:
                print("⚠️ Audio storage bucket not found. Creating...")
                try:
                    supabase.storage.create_bucket('audio', {'public': True})
                    print("✅ Audio storage bucket created.")
                except Exception as bucket_e:
                    print(f"❌ Could not create audio bucket: {bucket_e}")
        except Exception as storage_e:
            print(f"⚠️ Storage test failed: {storage_e}")
            
    except Exception as e:
        print(f"❌ Supabase connection failed: {e}")
        print("Check your Supabase URL and API key.")
    
    # Test models
    print("\n" + "="*30 + " MODEL STATUS " + "="*30)
    
    if speechbrain_speaker.classifier is not None:
        print("✅ SpeechBrain speaker recognition model loaded")
    else:
        print("❌ SpeechBrain speaker recognition model failed to load")
        
    if whisper_transcriber and whisper_transcriber.model is not None:
        print("✅ Whisper transcription model loaded")
    else:
        print("❌ Whisper transcription model failed to load")
    
    print("="*80)
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)