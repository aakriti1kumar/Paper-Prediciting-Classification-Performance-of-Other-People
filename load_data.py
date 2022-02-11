import numpy as np
import pandas as pd


# Load Data
def load_data(dataframe) -> dict:
    I = len(dataframe.uid.unique())  # Number of participants
    J = 16  # Number of problem sets
    K = 12  # Number of items in a problem set

    # True performance
    df_true = dataframe[['uid', 'actCorrectSelf', 'idSet']]
    df_true = df_true.sort_values(['uid', 'idSet'], ascending=[True, True], kind="mergesort")
    df_true = df_true[['uid', 'actCorrectSelf']]
    data_self_true = df_true.actCorrectSelf.values.reshape(I, J)

    # Estimated Self Performance
    df_self = dataframe[['uid', 'estCorrectSelf', 'idSet']]
    df_self = df_self.sort_values(['uid'], ascending=[True], kind="mergesort")
    df_self = df_self[['uid', 'estCorrectSelf', 'idSet']]
    data_self_est = df_self.estCorrectSelf.values.reshape(I, J)
    data_self_idset = df_self.idSet.values.reshape(I, J)

    # Estimated Other Performance
    df_other = dataframe[['uid', 'estCorrectOther', 'actCorrectOther', 'idSet']]
    df_other = df_other.sort_values(['uid'], ascending=[True], kind="mergesort")
    df_other = df_other[['uid', 'estCorrectOther', 'actCorrectOther', 'idSet']]
    df_other.estCorrectOther = np.nan_to_num(df_other.estCorrectOther, copy=True, nan=4, posinf=None, neginf=None)
    df_other.estCorrectOther = df_other.estCorrectOther.astype(int)
    df_other.actCorrectOther = np.nan_to_num(df_other.actCorrectOther, copy=True, nan=4, posinf=None, neginf=None)
    df_other.actCorrectOther = df_other.actCorrectOther.astype(int)

    data_other_est = df_other.estCorrectOther.values.reshape(I, J)
    data_other_true = df_other.actCorrectOther.values.reshape(I, J)

    M = 12
    v = (1 + np.arange(M)) / (M + 1)
    idx = dataframe.uid.unique() - 1
    data_dict = {'I': I,
                 'J': J,
                 'K': K,
                 'df_true': df_true,
                 'df_self': df_self,
                 'df_other': df_other,
                 'data_other_est': data_other_est,
                 'data_other_true': data_other_true,
                 'data_self_est': data_self_est,
                 'data_self_true': data_self_true,
                 'data_self_idset': data_self_idset,
                 'M': M,
                 'v': v,
                 'idx': idx}
    return data_dict


# if __name__ == '__main__':
#     df = pd.read_csv("Exp2_Estimation.csv")
#     df_feedback = df[df.conditionshowFeedback == 1]
#     df_nofeedback = df[df.conditionshowFeedback == 0]
#
#     data_full = load_data(df)
#     data_feedback = load_data(df_feedback)
#     data_nofeedback = load_data(df_nofeedback)
