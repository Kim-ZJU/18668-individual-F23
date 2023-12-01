import re
import pandas as pd
import numpy as np
import contractions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizer, DistilBertModel
import torch


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')


def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
    return embeddings


def load_file(file_path, num_of_records):
    df = pd.read_csv(file_path, sep=';', header=None, nrows=550000, on_bad_lines='skip', low_memory=False)
    selected_columns = [6, 7, 8, 9, 10, 11, 21, 22, 25]  # Column indices for features
    label_column = [17]
    new_df = df[selected_columns + label_column]
    new_df.columns = ['Department', 'Product', 'Component', 'Version', 'Platform', 'OS', 'Reporter', 'Assignee',
                      'Description', 'Severity']
    # print(new_df['Severity'].value_counts())

    p1 = new_df[new_df['Severity'] == 'P1'].sample(n=2 * num_of_records, random_state=42)
    p2 = new_df[new_df['Severity'] == 'P2'].sample(n=3 * num_of_records, random_state=42)
    p3 = new_df[new_df['Severity'] == 'P3'].sample(n=4 * num_of_records, random_state=42)
    p4 = new_df[new_df['Severity'] == 'P4'].sample(n=num_of_records, random_state=42)
    p5 = new_df[new_df['Severity'] == 'P5'].sample(n=num_of_records, random_state=42)

    combined_df = pd.concat([p1, p2, p3, p4, p5])
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(combined_df['Severity'].value_counts())
    return combined_df


def process_descrip(descrip_col):
    descrip_col = descrip_col.str.replace("\r", " ").replace("\n", " ").replace('"', '').replace('\'', '')
    descrip_col = descrip_col.str.lower()
    descrip_col = descrip_col.astype(str)
    descrip_col = descrip_col.apply(contractions.fix)
    # remove punctuation
    punctuation_signs = ["\(", "\)", "\[", "\]", "/", "-"]
    for punct_sign in punctuation_signs:
        descrip_col = descrip_col.str.replace(punct_sign, ' ', regex=True)
    # remove extra spaces
    for index, row in descrip_col.items():
        row = re.sub(r'\s+', ' ', row).strip()
        descrip_col.at[index] = row

    descrip_col = descrip_col.str.replace("'s", "", regex=True)

    # Using TF-IDF Vectorizer for the 'Description' column
    # Parameter election
    ngram_range = (1, 2)
    min_df = 0.01  # ignore terms that appear in less than 1% of the documents
    max_df = 0.8  # ignore terms that appear in more than 80% of the documents
    max_features = 100  # bug description is not as long as app reviews

    tfidf = TfidfVectorizer(ngram_range=ngram_range,
                            stop_words=None,
                            max_df=max_df,
                            min_df=min_df,
                            max_features=max_features,
                            norm='l2',
                            sublinear_tf=True)
    tfidf_features = tfidf.fit_transform(descrip_col).toarray()

    # Creating a new DataFrame for the TF-IDF features
    tfidf_df = pd.DataFrame(tfidf_features, columns=[f'desc_{i}' for i in range(tfidf_features.shape[1])])
    return tfidf_df

    # descrip_col = descrip_col.apply(lambda x: get_bert_embeddings(x))
    # bert_embed = pd.DataFrame(descrip_col.tolist(), columns=[f'bert_{i}' for i in range(descrip_col.iloc[0].shape[0])])
    #
    # return bert_embed


def process_data(df):
    description = process_descrip(df['Description'])

    # Label Encoding for categorical features
    label_encoders = {}
    for column in ['Department', 'Product', 'Component', 'Version', 'Platform', 'OS', 'Reporter', 'Assignee',
                   'Severity']:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le  # Storing label encoders if needed later for inverse transform

    # Rearrange the severity column to the last column
    severity = df['Severity']
    df.drop('Severity', axis=1, inplace=True)
    df = pd.concat([df, description, severity], axis=1)
    df.drop('Description', axis=1, inplace=True)
    print(df.head())
    return df


df = load_file('eclipse_bug_records_22-07-2022-csv.txt', 5000)
df = process_data(df)
# df.to_pickle('tfidf_processed_dataframe.pkl')
df.to_pickle('tfidf_balanced_dataframe.pkl')