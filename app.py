from flask import Flask, render_template, request, url_for
import pandas as pd
import csv
from werkzeug.utils import secure_filename
import uuid #Random Short Id
import os

UPLOAD_FOLDER = 'static/uploads' #Location is saving uploaded
ALLOWED_EXTENSIONS = {'csv'} #Kind of file

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')




#------------------------------------------------------------------------------------------
#start svm

@app.route('/SVM', methods=['GET', 'POST'])
def SVM_index():
    return render_template('SVM/index.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/SVM/data', methods=['GET', 'POST'])
def SVM_data():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            errors = 'No file part! Please choose 1 file excel !'
            return render_template('SVM/data.html', errors=errors)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            errors = 'No selected file'
            return render_template('SVM/data.html', errors=errors)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'SVM_' + str(uuid.uuid4())[:8] + '_' + filename)
            file.save(file_path)

            data = pd.read_csv(file_path)
        
            m = data.shape[1]

            return render_template('SVM/data.html', data=data.to_html(table_id='myTable', classes='table table-striped', header=True, index=False), m=m)


# SVM END
#--------------------------------------------------------------------------------------------------------------------------
# Kmean start



@app.route('/kmeans', methods=['GET', 'POST'])
def kmeans_index():
    return render_template('kmeans/index.html')

@app.route('/kmeans/data', methods=['GET', 'POST'])
def kmeans_data():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            errors = 'No file part! Please choose 1 file csv !'
            return render_template('kmeans/data.html', errors=errors)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            errors = 'No selected file'
            return render_template('kmeans/data.html', errors=errors)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'],'kmeans_' + str(uuid.uuid4())[:8] + '_' + filename)
            file.save(file_path)

            data = pd.read_csv(file_path)
        
            m = data.shape[1]

            return render_template('kmeans/data.html', data=data.to_html(table_id='myTable', classes='table table-striped', header=True, index=False), m=m)


if __name__ == '__main__':
    app.run(debug=True)


