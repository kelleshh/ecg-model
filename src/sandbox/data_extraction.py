import pandas as pd
import numpy as np
import wfdb
import ast
from pathlib import Path

def load_train_test_ptbxl_data(path: Path,
                               sampling_rate: int = 100,
                               test_fold: int = 10) -> tuple[np.ndarray, pd.Series, np.ndarray, pd.Series]:
    if sampling_rate not in (100, 500):
        raise ValueError('sampling_rate must be 100 or 500')

    def _load_raw_data(df, sampling_rate, path):
        if sampling_rate == 100:
            data = [wfdb.rdsamp(path/f) for f in df.filename_lr]
        else:
            data = [wfdb.rdsamp(path/f) for f in df.filename_hr]
        data = np.array([signal for signal, meta in data])
        return data

    # load and convert annotation data
    Y = pd.read_csv(path / 'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load raw signal data
    X = _load_raw_data(Y, sampling_rate, path)

    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(path / 'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def _aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    # Apply diagnostic superclass
    Y['diagnostic_superclass'] = Y.scp_codes.apply(_aggregate_diagnostic)

    # Train and test mask
    train_mask = Y['strat_fold'] != test_fold
    test_mask = Y['strat_fold'] == test_fold
    # Train
    X_train = X[train_mask]
    y_train = Y.loc[train_mask, 'diagnostic_superclass']
    # Test
    X_test = X[test_mask]
    y_test = Y.loc[test_mask, 'diagnostic_superclass']

    return X_train, y_train, X_test, y_test
