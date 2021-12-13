# Importing required libraries
import pickle
import cv2
from skimage.metrics import structural_similarity
import sqlite3
import pandas as pd


import tensorflow as tf
from tensorflow import keras

# Import required packages for pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.compose import ColumnTransformer

# Loading the pickled binfiles
with open('Data/Bin files/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

model = tf.keras.models.load_model('Data/Bin files/Custom model 2.h5')


# Constructing a custom transformer for resizing images
class ResizeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cv2 = __import__('cv2')
        self.dim = (72, 72)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = self.cv2.cvtColor(X, self.cv2.COLOR_BGR2GRAY)
        X = self.cv2.resize(X, self.dim, interpolation=self.cv2.INTER_AREA)
        X = X.reshape(1, -1)
        return X


# Constructing a custom transformer for resizing images
class CategoryPredictor(BaseEstimator, TransformerMixin):
    def __init__(self,model):
        self.model = model
        self.labels = ['Hoodie/Sweatshirt', 'Jeans', 'Leather Jacket', 'Pants', 'Raincoat', 'Shorts', 'Sleepwear',
                       'Socks',
                       'Suits/Blazers', 'Sweater']

    def fit(self, X, y=None):
        return self.model

    def predict(self, X, y=None):
        X = X.reshape(-1, 72, 72, 1)
        prediction = self.model.predict_classes(X)
        return self.labels[prediction[0]]


resize_trans = ResizeTransformer()
predictor = CategoryPredictor(model)
# Model pipeline
model_pipeline = Pipeline([('resize_trans', resize_trans), ('scaler', scaler),
                            ('predictor', predictor)])


def shortlist_images(category):
    # DB connection for loading data
    db_connection = sqlite3.connect('Data/Database/job_database.db')
    query = f"SELECT img_src FROM product_info WHERE type = '{category}'"
    product_df = pd.read_sql_query(query, db_connection)
    db_connection.close()
    return product_df


def similar_images(img_src, compare_image):
    # Dict to store similarity
    image_similarity = {}
    # Iterate through images
    dim = (72, 72)
    compare_image = cv2.cvtColor(compare_image, cv2.COLOR_BGR2RGB)
    compare_image = cv2.resize(compare_image, dim, interpolation=cv2.INTER_AREA)
    compare_image = compare_image.reshape(-1)
    for img in img_src['img_src']:
        file_name = img.split('/')[-1]
        path = f'Data/Images/{file_name}'
        image = cv2.imread(path)
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            continue

        # Resize the image
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        image = image.reshape(-1)
        image_similarity[img] = structural_similarity(compare_image, image)

    return image_similarity


def find_similar_products(product_type, image):
    # Write as df
    similarity_score = similar_images(shortlist_images(product_type), image)
    similarity_df = pd.DataFrame(similarity_score.items(), columns=['img_src', 'sim_score'])
    # Read all the data
    # DB connection for loading data
    db_connection = sqlite3.connect('Data/Database/job_database.db')
    query = f"SELECT link,img_src FROM product_info WHERE type = '{product_type}'"
    product_df = pd.read_sql_query(query, db_connection)
    db_connection.close()

    similarity_df = similarity_df.sort_values(by=['sim_score'], ascending=False).head(5)
    return similarity_df.merge(product_df, on=['img_src'])


def cosine_similarity(similar_images_df):
    cosine_df = pd.read_csv('Data/Bin files/cosine_similarity.csv', index_col=0)
    similar_index = set(similar_images_df['link'])
    for index, row in similar_images_df.iterrows():

        if len(similar_index) >= 10:
            break
        index_sim_prod = cosine_df[cosine_df[row['link']] > 0.8][row['link']].index

        for idx in index_sim_prod:
            if len(similar_index) != 10:
                similar_index.add(idx)

    return similar_index


def recommend_products(image, model_pipeline):
    # Predict the type of product
    product_type = model_pipeline.predict(image)
    similar_images_df = find_similar_products(product_type, image)
    similar_links = cosine_similarity(similar_images_df)
    return similar_links


image = cv2.imread('upload/temp.jpg')

recommended_products = recommend_products(image, model_pipeline)

# calculate
with open('upload/product_link.txt', 'wb') as file:
     for link in recommended_products:
            link = link + '\n'
            file.write(link.encode())
