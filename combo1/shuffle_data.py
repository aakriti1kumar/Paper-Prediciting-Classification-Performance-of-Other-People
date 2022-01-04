import numpy as np

def shuffle(other_id, data_other_true, I, J):
    data_other_true_shuffled = np.zeros((I,J))
    for i in range(I):
        id_other = int(other_id[i])
        data_other_true_shuffled[i] = data_other_true[id_other]
    return data_other_true_shuffled.astype(int)

def shuffle_idset(data_self_idset, data_other_true_shuffled, I, J):
    data_other_true_shuffled_idset = np.zeros((I,J))
    for i in range(I):
        for j in range(J):
            data_other_true_shuffled_idset[i,j] = data_other_true_shuffled[i, (data_self_idset[i,j]-1)]
    return data_other_true_shuffled_idset.astype(int)
