from flask import Flask, render_template, request
import os
import subprocess
import pandas as pd
import sqlite3


app = Flask(__name__)
UPLOAD_FOLDER = '../upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/',  methods=['POST', 'GET'])
@app.route('/index',  methods=['POST', 'GET'])
def home_page():
    if request.method == 'POST':
        if 'filename' not in request.files:
            return 'there is no file1 in form!'
        file1 = request.files['filename']
        path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp.jpg')
        file1.save(path)
        subprocess.call("python model_pipeline.py", shell=True, cwd='../')

        with open('../upload/product_link.txt', 'r') as file:
            links = file.readlines()
            links = [line.rstrip() for line in links]

        links_df = pd.DataFrame(links, columns=['link'])
        db_connection = sqlite3.connect('../Data/Database/job_database.db')
        query = f"SELECT * FROM product_info"
        product_df = pd.read_sql_query(query, db_connection)
        db_connection.close()

        selected_products = links_df.merge(product_df)
        context = dict()
        for idx, values in selected_products.iterrows():
            context[f'name{idx + 1}'] = values['product_name']
            context[f'img_src{idx + 1}'] = values['img_src'].split('/')[-1]
            context[f'link{idx + 1}'] = values['link']
            context[f'description{idx + 1}'] = values['description']
            context[f'price{idx + 1}'] = values['price']
            context[f'color_available{idx + 1}'] = values['color_available']
            context[f'size_available{idx + 1}'] = values['size_available']

        return render_template('recommended.html',**context)

    return render_template('index.html')


@app.route('/recommended',  methods=['POST', 'GET'])
def recommended():
    return render_template('recommended.html')


if __name__ == "__main__":
    app.run()
