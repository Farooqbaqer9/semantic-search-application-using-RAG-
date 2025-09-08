#!/usr/bin/env python3
"""
Setup script for Oysro Meeting Room
This script helps set up the environment and check dependencies
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path

def run_command(command, description=""):
    """Run a command and return success status"""
    print(f"\n🔄 {description}")
    print(f"Running: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"✅ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: {command[0]}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is too old. Python 3.8+ required.")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_ffmpeg():
    """Check if ffmpeg is available"""
    print("\n🎵 Checking FFmpeg...")
    if run_command(["ffmpeg", "-version"], "Checking FFmpeg installation"):
        return True
    else:
        print("⚠️  FFmpeg not found. Audio conversion may not work properly.")
        print("📥 Please install FFmpeg:")
        print("   Windows: https://ffmpeg.org/download.html#build-windows")
        print("   macOS: brew install ffmpeg")
        print("   Ubuntu: sudo apt install ffmpeg")
        return False

def install_pytorch():
    """Install PyTorch with appropriate configuration"""
    print("\n🔥 Installing PyTorch...")
    
    # Check if CUDA is available
    try:
        import torch
        if torch.cuda.is_available():
            print("✅ CUDA is available, PyTorch already installed")
            return True
    except ImportError:
        pass
    
    # Install CPU version of PyTorch (more reliable for most systems)
    commands = [
        [sys.executable, "-m", "pip", "install", "torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"]
    ]
    
    for command in commands:
        if not run_command(command, "Installing PyTorch CPU version"):
            print("⚠️  PyTorch installation failed, trying alternative...")
            # Fallback to regular pip install
            if not run_command([sys.executable, "-m", "pip", "install", "torch", "torchaudio"], "Installing PyTorch (fallback)"):
                return False
        else:
            return True
    
    return False

def install_requirements():
    """Install Python dependencies"""
    print("\n📦 Installing Python dependencies...")
    
    # First, ensure pip is up to date
    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], "Upgrading pip")
    
    # Install core dependencies first
    core_deps = [
        "numpy>=1.20.0",
        "librosa>=0.8.0",
        "soundfile>=0.10.0",
        "flask>=2.0.0",
        "flask-cors>=3.0.0",
        "python-dotenv>=0.19.0",
        "supabase>=1.0.0",
        "scipy>=1.7.0"
    ]
    
    for dep in core_deps:
        if not run_command([sys.executable, "-m", "pip", "install", dep], f"Installing {dep}"):
            print(f"⚠️  Failed to install {dep}")
    
    # Install PyTorch separately
    if not install_pytorch():
        print("❌ PyTorch installation failed")
        return False
    
    # Install SpeechBrain
    if not run_command([sys.executable, "-m", "pip", "install", "speechbrain"], "Installing SpeechBrain"):
        print("⚠️  SpeechBrain installation failed")
    
    # Install Whisper
    if not run_command([sys.executable, "-m", "pip", "install", "openai-whisper"], "Installing OpenAI Whisper"):
        print("⚠️  Whisper installation failed")
    
    return True

def test_imports():
    """Test if critical modules can be imported"""
    print("\n🧪 Testing module imports...")
    
    modules_to_test = [
        ("flask", "Flask web framework"),
        ("librosa", "Audio processing"),
        ("numpy", "Numerical computing"),
        ("soundfile", "Audio file I/O"),
        ("supabase", "Supabase client"),
        ("torch", "PyTorch"),
        ("speechbrain", "SpeechBrain"),
        ("whisper", "OpenAI Whisper")
    ]
    
    success_count = 0
    
    for module_name, description in modules_to_test:
        try:
            __import__(module_name)
            print(f"✅ {description} ({module_name})")
            success_count += 1
        except ImportError as e:
            print(f"❌ {description} ({module_name}): {e}")
    
    print(f"\n📊 Import test results: {success_count}/{len(modules_to_test)} modules working")
    
    if success_count < len(modules_to_test):
        print("⚠️  Some modules failed to import. The application may not work correctly.")
        return False
    
    return True

def create_env_file():
    """Create a .env file template"""
    print("\n📝 Creating .env file template...")
    
    env_content = """# Oysro Meeting Room Environment Variables

# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-api-key

# Application Settings
FLASK_ENV=development
FLASK_DEBUG=1

# Audio Settings
DEFAULT_SAMPLE_RATE=16000
RECORDING_CHUNK_DURATION=5
SPEAKER_CONFIDENCE_THRESHOLD=0.4

# Model Settings
WHISPER_MODEL_SIZE=base
SPEECHBRAIN_MODEL=speechbrain/spkrec-ecapa-voxceleb
"""
    
    env_file = Path(".env")
    if not env_file.exists():
        with open(env_file, "w") as f:
            f.write(env_content)
        print("✅ Created .env file template")
        print("📝 Please edit .env file with your Supabase credentials")
    else:
        print("ℹ️  .env file already exists")
    
    return True

def create_startup_script():
    """Create a startup script"""
    print("\n🚀 Creating startup script...")
    
    startup_content = """#!/bin/bash
# Oysro Meeting Room Startup Script

echo "Starting Oysro Meeting Room..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# For Windows: venv\\Scripts\\activate.bat

# Start the Flask application
echo "Starting Flask server..."
python app.py

echo "Server stopped."
"""
    
    startup_file = Path("start.sh")
    with open(startup_file, "w") as f:
        f.write(startup_content)
    
    # Make executable on Unix systems
    if os.name != 'nt':
        os.chmod(startup_file, 0o755)
    
    print("✅ Created start.sh script")
    return True

def test_audio_processing():
    """Test basic audio processing functionality"""
    print("\n🎤 Testing audio processing...")
    
    try:
        import numpy as np
        import soundfile as sf
        import tempfile
        
        # Create a test audio signal
        sample_rate = 16000
        duration = 1.0  # 1 second
        frequency = 440  # A4 note
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_signal = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        # Test saving and loading
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
        
        sf.write(temp_path, audio_signal, sample_rate)
        loaded_audio, loaded_sr = sf.read(temp_path)
        
        os.unlink(temp_path)
        
        if len(loaded_audio) > 0 and loaded_sr == sample_rate:
            print("✅ Audio processing test passed")
            return True
        else:
            print("❌ Audio processing test failed")
            return False
            
    except Exception as e:
        print(f"❌ Audio processing test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("=" * 60)
    print("🎤 OYSRO MEETING ROOM SETUP")
    print("=" * 60)
    
    success = True
    
    # Check Python version
    if not check_python_version():
        success = False
    
    # Check FFmpeg
    check_ffmpeg()  # Not critical, just warn
    
    # Install dependencies
    if not install_requirements():
        success = False
    
    # Test imports
    if not test_imports():
        success = False
    
    # Test audio processing
    if not test_audio_processing():
        success = False
    
    # Create configuration files
    create_env_file()
    create_startup_script()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 SETUP COMPLETED SUCCESSFULLY!")
        print("\nNext steps:")
        print("1. Edit the .env file with your Supabase credentials")
        print("2. Create the required tables in your Supabase database:")
        print("   - speakers")
        print("   - speaker_samples") 
        print("   - speaker_embeddings")
        print("   - meetings")
        print("   - meeting_chunks")
        print("   - transcripts")
        print("3. Create an 'audio' storage bucket in Supabase")
        print("4. Run: python app.py")
        print("5. Open the frontend HTML file in your browser")
        print("\n📚 Check the README for database schema details")
    else:
        print("❌ SETUP COMPLETED WITH ISSUES")
        print("\nSome components failed to install or configure properly.")
        print("Please review the error messages above and:")
        print("1. Install missing dependencies manually")
        print("2. Check your Python environment")
        print("3. Ensure you have proper internet connectivity")
        print("4. Consider using a virtual environment")
    
    print("=" * 60)
    
    return success

if __name__ == "__main__":
    main()