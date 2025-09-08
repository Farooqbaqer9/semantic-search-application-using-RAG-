#!/bin/bash
# Oysro Meeting Room Startup Script

echo "Starting Oysro Meeting Room..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# For Windows: venv\Scripts\activate.bat

# Start the Flask application
echo "Starting Flask server..."
python app.py

echo "Server stopped."
