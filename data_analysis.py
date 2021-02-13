import pandas as pd

"""
to read train data and convert it into a dataframe
"""
df = pd.read_csv("Data/train.csv", encoding="ISO-8859-1")
print("Train dataframe:")
print(df.head)

"""
to extract labels from train data
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
"""
labels = [label for label in df.columns][2:]
print("The labels are %s" % ", ".join(labels))

"""
to calculate frequency of multilabels
"""
combinations = df.groupby(labels).size().reset_index().rename(columns={0: 'count'})
print("Multilables:")
print(combinations.head)
# combinations.to_excel("combinations.xlsx")  # write into a file