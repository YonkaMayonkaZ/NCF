from src.data.preprocessing import LeaveOneOutPreprocessor

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


if __name__ == "__main__":
    preprocessor = LeaveOneOutPreprocessor()
    preprocessor.run()
    