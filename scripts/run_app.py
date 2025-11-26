import os
import subprocess

# Set environment variables for Flask
os.environ.setdefault('FLASK_APP', 'app')
os.environ.setdefault('FLASK_ENV', 'development')

# Change working directory to project root (where app package resides)
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)

# Launch Flask development server
subprocess.run(['flask', 'run'])
