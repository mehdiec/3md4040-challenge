from pathlib import Path


new_file = "data/train/train_csv/train.csv"


my_file = Path(new_file)
if not my_file.is_file():
    print("I hate u")
