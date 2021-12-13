# Importing required libraries
import sqlite3
import pandas as pd
import numpy as np
import cv2

#### Data Cleaning for cosine similarity

# DB connection for loading data
db_connection = sqlite3.connect('Data/Database/job_database.db')
product_df = pd.read_sql_query('select * from product_info', db_connection)

# Extract numerical ratings
product_df['rating'] = product_df['rating'].apply(lambda x: x.split('out')[0])
product_df['rating'] = product_df['rating'].astype('float64')

# Extract numerical values - no_of_rating
product_df['no_of_rating'] = product_df['no_of_rating'].apply(lambda x: x.split(' global')[0])
product_df['no_of_rating'] = product_df['no_of_rating'].apply(lambda x: x.replace(',',''))
product_df['no_of_rating'] = product_df['no_of_rating'].mask(product_df['no_of_rating'] == 'Total price:')
product_df['no_of_rating'] = product_df['no_of_rating'].mask(product_df['no_of_rating'] == 'Size :')
product_df['no_of_rating'] = product_df['no_of_rating'].astype('float64')

# Extract numerical values - price
product_df['price'] = product_df['price'].apply(lambda x: x.split('$')[-1])
product_df['price'] = product_df['price'].mask(product_df['price'] == '')
product_df['price'] = product_df['price'].replace('[A-Za-z]', np.nan, regex=True)
product_df['price'] = product_df['price'].astype('float64')

# One-hot encode the colors
all_colors = set()  # to extract distinct colors

for colors in product_df['color_available']:
    product_colors = colors.split(', ')[:-1]
    for color in product_colors:
        all_colors.add(color)


for colors in all_colors:
    color_name = 'color_' + colors
    product_df[color_name] = 0  # Create columns for each color

for index, row in product_df.iterrows():
    product_colors = row['color_available'].split(', ')[:-1]
    for color in product_colors:
        color_name = 'color_' + color
        product_df.loc[index, color_name] = 1  # Change value to 1 if color presents

product_df.drop('color_available', axis=1, inplace=True)  # Drop the original row

# One-hot encode the size
all_sizes = set()  # to extract distinct size

for sizes in product_df['size_available']:
    product_sizes = sizes.split(', ')[:-1]
    for size in product_sizes:
        all_sizes.add(size)

for sizes in all_sizes:
    size_name = 'size_' + sizes
    product_df[size_name] = 0  # Create columns for each size

for index, row in product_df.iterrows():
    product_sizes = row['size_available'].split(', ')[:-1]
    for sizes in product_sizes:
        size_name = 'size_' + sizes
        product_df.loc[index, size_name] = 1

product_df.drop('size_available', axis=1, inplace=True)  # Drop the original row

# Clean and one-hot encode brand
product_df['brand'] = product_df['brand'].replace('Visit the ', '', regex=True)
product_df['brand'] = product_df['brand'].replace('Brand: ', '', regex=True)
product_df = pd.get_dummies(product_df, columns=['brand'])

# One-hot encode type
product_df = pd.get_dummies(product_df, columns=['type'])

# Store the cleaned data for cosine similarity
product_df.to_csv('Data/Transformed data/cleaned_product_data.csv', index=False)

### Data Transformation for Image classification

# Extract again to get the unmodified data
# DB connection for loading data
db_connection = sqlite3.connect('Data/Database/job_database.db')
product_df = pd.read_sql_query('select * from product_info', db_connection)


## Removing the real duplicate images
# Images in more than one record
duplicated_links = product_df[product_df.duplicated(subset=['img_src'])]['img_src']
# List to store images that has different labels
real_duplicates = []

for link in duplicated_links:
    # List of distinct labels
    temp_duplicates = set(product_df[product_df['img_src'] == link]['type'])
    # If all duplicates have same label, move to next image
    if len(temp_duplicates) != 1:
        real_duplicates.append(link)

# Indexes of duplicate pants
pant_index = product_df[(product_df['img_src'].isin(real_duplicates)) & (product_df['type'] == 'Pants')].index
# Remove the pant duplicates
product_df = product_df.drop(pant_index)
# Remove all other duplicates
product_df.drop_duplicates(subset=['img_src'], inplace=True)


## Image to numerical conversion
# Arrays to store vectors and labels
image_label_arr = np.empty((0, 1))
image_arr = np.empty((0, 5184))
dim = (72, 72)

# Iterate through the available records
for row in product_df.iterrows():
    # Load the image
    img_src = row[1]['img_src']
    name = img_src.split('/')[-1]
    path = f'Data/Images/{name}'
    image = cv2.imread(path)
    try:
        # Converting the image to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except:
        continue

    # Resize the image
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    image = image.reshape(1, -1)
    # Append to the Numpy array
    image_arr = np.vstack([image_arr, image])
    image_label_arr = np.vstack([image_label_arr, row[1]['type']])

# Save the numpy array as csv
np.savetxt("Data/Transformed data/image_array.csv", image_arr, delimiter=",")
np.savetxt("Data/Transformed data/image_label_array.csv", image_label_arr, delimiter=",", fmt='%s')
