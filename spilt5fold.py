import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def generate_kfold_csv(
    all_csv_path,
    out_dir,
    n_splits=5,
    seed=42
):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(all_csv_path)
    filenames = df.iloc[:, 0].values
    labels    = df.iloc[:, 1].values

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed
    )

    for fold, (train_idx, test_idx) in enumerate(skf.split(filenames, labels), start=1):

        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df  = df.iloc[test_idx].reset_index(drop=True)

        train_path = os.path.join(out_dir, f"train_list_{fold}.csv")
        test_path  = os.path.join(out_dir, f"test_list_{fold}.csv")

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(f"[Fold {fold}] "
              f"Train: {len(train_df)} | Test: {len(test_df)}")

    print("✅ 5-fold CSV files generated successfully.")

generate_kfold_csv(
    all_csv_path="E:/Awork/data/back_ship3500/back.csv",
    out_dir="E:/Awork/data/back_ship3500",
    n_splits=5
)

