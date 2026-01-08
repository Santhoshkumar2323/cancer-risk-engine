from sklearn.model_selection import train_test_split

def split_holdout(df, n_holdout=5):
    train_df, holdout_df = train_test_split(
        df, test_size=n_holdout, random_state=42
    )
    return train_df, holdout_df
