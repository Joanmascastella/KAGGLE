import pandas as pd


# have function that cleans data and normalizes it by removing null values and doing mean operations
# for those missing values

def load_and_clean_data(train_data, test_data):
    train_data = pd.read_csv(train_data)
    test_data = pd.read_csv(test_data)






    return train_data, test_data


# then create two data loader objects and return them