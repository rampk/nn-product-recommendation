# Import required libraries
import pandas as pd
import contractions
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import re
from sklearn.metrics.pairwise import cosine_similarity

product_df = pd.read_csv('Data/Transformed data/cleaned_product_data.csv')

## Creating seperate cosine scores for textual and categorical data
# textual data
text_data = pd.DataFrame(columns=['link', 'name_description'])
text_data['link'] = product_df['link']
text_data['name_description'] = product_df['product_name'] + product_df['description']

# numerical and categorical data
num_cat_data = product_df.copy()
num_cat_data = num_cat_data.drop(['product_name','description','img_src','date','time'],axis=1)


# Converting non-ascii
def transform_nonascii(text):
    return ''.join(char for char in text if ord(char) < 128)


# Removing numbers
def remove_numbers(text):
    return re.sub(r'[0-9]+', '', text)


# Removing slashes
def remove_slash(text):
    return re.sub(r'[\n,\b,\t]', '', text)


# Removing contractions
def remove_contractions(text):
    return contractions.fix(text)


# Removing non-alphanumerics:
def remove_nonalpha(text):
    text = re.sub(r'[^\w]', ' ', text)
    text = re.sub(r'_', '', text)
    return text


# Unwanted spaces:
def remove_space(text):
    return re.sub(r' +', ' ', text)


# Cleaning the text
def clean_data(df):
    df['name_description'] = df['name_description'].apply(transform_nonascii)
    df['name_description'] = df['name_description'].apply(remove_numbers)
    df['name_description'] = df['name_description'].apply(remove_slash)
    df['name_description'] = df['name_description'].apply(remove_contractions)
    df['name_description'] = df['name_description'].apply(remove_nonalpha)
    df['name_description'] = df['name_description'].apply(remove_space)
    df['name_description'] = df['name_description'].str.lower()

    return df

# Stop words removal
def remove_stop_words(text):

    removed_list = []
    stop_words = stopwords.words('english')
    for token in text.split():

        if token not in stop_words:
            removed_list.append(token)

    return " ".join(removed_list)


# Stemming
def stemming(text):
    ps = PorterStemmer()
    return " ".join([ps.stem(word) for word in word_tokenize(text)])


text_data = clean_data(text_data)
text_data['name_description'] = text_data['name_description'].apply(remove_stop_words)
text_data['name_description'] = text_data['name_description'].apply(stemming)
vectorizer = CountVectorizer()
text_X = vectorizer.fit_transform(text_data['name_description'])

# Fill null values by 0
num_cat_data = num_cat_data.fillna('0')

# Calculating the cosine similarity
text_similarity = cosine_similarity(text_X,text_X)
num_similarity = cosine_similarity(num_cat_data.drop('link',axis=1),num_cat_data.drop('link',axis=1))

# Giving weightage
normalized_similarity = ((num_similarity * 2) + text_similarity)/3

cosine_similarity = pd.DataFrame(normalized_similarity, columns=product_df['link'],index=product_df['link'])
# Save the Calculated cosine scores
cosine_similarity.to_csv('Data/Bin files/cosine_similarity.csv')
