import sys
import os

# Always add the project root to Python path regardless of how pytest is invoked
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.abspath(__file__))))