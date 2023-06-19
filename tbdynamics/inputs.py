import pathlib
import pandas as pd

BASE_PATH = pathlib.Path(__file__).parent.parent.resolve()
DATA_PATH = BASE_PATH / "data"

def load_pop_data():
    csv_path = pathlib.Path(DATA_PATH /'camau.csv')
    pop_df = pd.read_csv(csv_path)
    pop_df = pop_df.set_index(['year'])
    return pop_df