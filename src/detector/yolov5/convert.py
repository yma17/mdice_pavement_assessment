import pandas as pd
from collections import Counter, OrderedDict

def main():

    # load results
    df = pd.read_csv('results.csv', header=None, names=['id', 'damage'])

    # remove images with no damage detected
    df = df[df.damage.isna() == False]

    # initialize dataframe for counting # of each damage type in each image
    count = pd.DataFrame(columns=['id', 'D00', 'D10', 'D20', 'D40'])

    for img in range(df.shape[0]):
        damage = Counter(df.iloc[img].damage.split(' ')[:-1:5])
        for i in range(4):
            if str(i+1) not in damage.keys():
                damage[str(i+1)] = 0
        count.loc[count.shape[0]] = [df.iloc[img].id] + list(OrderedDict(damage).values())

    # save damage count
    count.to_csv('damage.csv', index=False)


if __name__=="__main__":
    main()