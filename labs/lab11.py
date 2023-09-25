'''
Lab 11
CSCI 4553
Jorge Carranza Pena
20563986
'''
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def run(which = ''):
    print("Lab 11: Normalization")

    url = 'https://raw.githubusercontent.com/dkims/CSCI4341/main/auto-mpg.data'
    df = pd.read_csv(url, delim_whitespace=True, header=None, na_values=['?'])


    numeric_columns =df.iloc[:, 1:-1]

    numeric_columns.fillna(numeric_columns.median(), inplace=True)

    scaler = MinMaxScaler()
    scaler.fit(numeric_columns.dropna())
    normalized_data = scaler.transform(numeric_columns)
    normalized_df = pd.DataFrame(normalized_data, columns=[
                                                        'cyl',
                                                        'dis',
                                                        'hp',
                                                        'wei',
                                                        'acc',
                                                        'year',
                                                        'ori',
                                                          ])
    normalized_df.insert(0, 'MPG', df.iloc[:, 0])
    normalized_df['Car Name'] = df.iloc[:, -1]
    if which == 'full':
        with pd.option_context('display.max_rows', None,):
            print(normalized_df)
    else:
        print(normalized_df)

run()
