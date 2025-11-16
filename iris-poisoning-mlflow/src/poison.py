import numpy as np
import pandas as pd

def poison_data(X, y, fraction, random_state=42):
    rng = np.random.default_rng(random_state)

    n = len(X)
    k = int(fraction * n)

    if k == 0:
        return X.copy(), y.copy()

    idx = rng.choice(n, k, replace=False)

    Xp = X.copy()
    yp = y.copy()

    # randomize features
    for col in Xp.columns:
        col_min, col_max = Xp[col].min(), Xp[col].max()
        Xp.loc[idx, col] = rng.uniform(col_min, col_max, size=k)

    # random labels
    unique_labels = np.unique(y)
    yp.iloc[idx] = rng.choice(unique_labels, size=k)

    return Xp, yp
