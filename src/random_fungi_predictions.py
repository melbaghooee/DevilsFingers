import os
from pathlib import Path
import random
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent

def do_random_fungi_predictions():
    n_classes = 183
    method_name = 'random_fungi_predictions'

    metadata_file = ROOT_DIR / "data" / "metadata" / "metadata.csv"
    predictions_out = ROOT_DIR / "output" / "predictions_6.csv"

    predictions_out.parent.mkdir(parents=True, exist_ok=True)
    print(predictions_out)

    f_out = open(predictions_out, 'w')
    f_out.write(f"{method_name}\n")

    # Read the metadata from the csv file using pandas
    # check the filename in the first column. If it contains "test" a random prediction is made
    # in the range 0 <= prediction < n_classes
    # create a new csv file with the same filename as the metadata file and the predictions in the second column
    df = pd.read_csv(metadata_file, header=None)
    for index, row in df.iterrows():
        filename = row[0]
        if 'test' in filename:
            prediction = random.randint(0, n_classes - 1)
            f_out.write(f"{filename},{prediction}\n")


if __name__ == '__main__':
    do_random_fungi_predictions()

