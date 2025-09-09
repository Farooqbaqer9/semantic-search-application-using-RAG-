# Environment Variables for Osyro Meeting Room Deployment

Copy these into your deployment platform's environment variables section:

```
SUPABASE_URL=your-supabase-url
SUPABASE_KEY=your-supabase-key
FLASK_ENV=development
FLASK_DEBUG=0
DEFAULT_SAMPLE_RATE=16000
RECORDING_CHUNK_DURATION=5
SPEAKER_CONFIDENCE_THRESHOLD=0.4
WHISPER_MODEL_SIZE=small
SPEECHBRAIN_MODEL=speechbrain/spkrec-ecapa-voxceleb
FRONTEND_ORIGIN=https://meeting_roomAI.com
# Add any other variables you use
```

- Replace `your-supabase-url` and `your-supabase-key` with your real Supabase credentials.
- You can add more variables as needed (e.g., for model size, etc.).
- Example additional variables you might use:
    - `SECRET_KEY=your-flask-secret-key`  # Flask session security
- Never commit your real `.env` file to git.
