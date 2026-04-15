import sys
import os

# Add the backend directory to sys.path so test files can import backend modules directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
