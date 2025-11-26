import sys
import os
sys.path.append(os.path.abspath("Task1"))
from Helper import Helper

try:
    h = Helper()
    print("Helper initialized successfully")
except Exception as e:
    import traceback
    traceback.print_exc()
