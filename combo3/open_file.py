import pickle
def open_file(filename):
    with open(f"{filename}", "rb") as handle:
        return pickle.load(handle)
