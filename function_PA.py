def get_dataset():
    '''
    Loads dataset from .csv file and return as train set and test set.
    '''
    raw_df = pd.read_csv('data/data_raw.csv')
    raw_df = raw_df.dropna()
    raw_df['full_text'] = raw_df.apply(lambda x: ' '.join([x['title'],x['text']]),axis=1)
    labels=raw_df.label
    x_train,x_test,y_train,y_test=train_test_split(raw_df['full_text'], labels, test_size=0.2, random_state=7)
    return x_train, x_test, y_train, y_test
