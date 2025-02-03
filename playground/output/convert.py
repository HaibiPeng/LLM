import pandas as pd
import jsonlines
from sklearn.model_selection import train_test_split

def create_jsonl_files(df, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
    assert train_ratio + valid_ratio + test_ratio == 1, "Ratios should sum to 1"

    train_df, temp_df = train_test_split(df, test_size=1 - train_ratio, random_state=42)

    valid_df, test_df = train_test_split(temp_df,
                                         test_size=test_ratio / (test_ratio + valid_ratio),
                                         random_state=42)

    datasets = {
        'train': train_df,
        'valid': valid_df,
        'test': test_df
    }

    for dataset_name, dataset_df in datasets.items():
        file_path = f'./{dataset_name}.jsonl'
        with jsonlines.open(file_path, 'w') as writer:
            for _, row in dataset_df.iterrows():
                data = {
                    "text": f"Prompt: {row['Question']}, output: {row['Answer']}"
                }
                writer.write(data)
        print(f'{dataset_name} dataset has been saved as {file_path}')

df = pd.read_csv('./dataset.csv')

create_jsonl_files(df)