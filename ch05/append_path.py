import sys, os

current_file_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_dir, os.pardir))
sys.path.append(parent_dir)
