from flask import Flask, render_template, request, url_for, session
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import csv
from werkzeug.utils import secure_filename
from sklearn.cluster import KMeans
import uuid #Random Short Id
import os

UPLOAD_FOLDER = 'static/uploads/kmeans' #Location is saving uploaded
ALLOWED_EXTENSIONS = {'csv'} #Kind of file

app = Flask(__name__)

app.secret_key = b'_5#y2L"F4Q8z\n\xec]/' #Secret key of Session
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/SVM', methods=['GET', 'POST'])
def svm():
    return render_template('SVM/upload.php')



# @app.route('/SVM/data', methods=['GET', 'POST'])
# def SVM():
#     if request.method == 'POST':
#         file = request.form['csvfile']  
#         if file.filename == '':
#             flash('No selected file')
#             return redirect(request.url)
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#             return redirect(url_for('uploaded_file',
#                                     filename=filename))
        
#         import numpy as np
#         import quadprog as qp
#         import matplotlib.pyplot as plot
#         import math 
#         import pandas as pd
    
        
#         data = pd.read_csv(data_upload)
#         m = data.shape[1]
#         #   Xây dựng tập dữ liệu X và nhãn y
#         X = pd.DataFrame(data.iloc[0:100,[0,1]])
#         y = pd.DataFrame(data.iloc[0:100,4])

#         plot.plot(X.iloc[0:50, 0], X.iloc[0:50, 1], 'bo')
#         plot.plot(X.iloc[50:100, 0], X.iloc[50:100, 1], 'rx')
#         plot.legend()
#         plot.grid()

#     #return render_template('data.html',data=data.to_html(classes='table', header=False, index=False), x=X.to_html(classes='table', header=False, index=False), y=y.to_html(classes='table', header=False, index=False))
#     return render_template('SVM/show-data.html',url='/static/images/plot1.png',data=data.to_html(classes='table', header=False, index=False),accuracy=acc,m=m)

# @app.route('/result', methods=['GET', 'POST'])
# def resultsvm():
#     if request.method == 'POST':
#         # f = request.form['filecsv']
#         col = request.form.getlist('cot')
#         data = data

#         plt.savefig('static/images/plot.png')

#         return render_template('result.html', url='/static/images/plot.png')

@app.route('/kmeans', methods=['GET', 'POST'])
def kmeans_index():
    return render_template('kmeans/index.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4())[:8] + '_' + filename)
            file.save(file_path)

            session['csvfile'] = file_path #Save path file to session
            data = pd.read_csv(file_path)
        
            m = data.shape[1]

            return render_template('kmeans/data.html', data=data.to_html(table_id='myTable', classes='table table-striped', header=True, index=False), m=m)

@app.route('/kmeans/elbow', methods=['GET', 'POST'])
def kmeans_elbow():
    file_path = session.get('csvfile')
    data = pd.read_csv(file_path)

    col = request.form.getlist('cot') #Get values of checkbox form from
    col = np.array(col)
    col1 = col[0]
    col2 = col[1]

    session['col1'] = col1 #Save column to session
    session['col2'] = col2 #Save column to session

    m = data.shape[1]
    haha = 0
    X = data.iloc[int(haha):, [int(col1), int(col2)]].values

    # Tiến hành gom nhóm (Elbow)
    # Chạy thuật toán KMeans với k = (1, 10)

    clusters = []
    for i in range(1, 10):
        km = KMeans(n_clusters=i).fit(X)
        clusters.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(x=list(range(1, 10)), y=clusters, ax=ax)

    ax.set_title("Đồ thị Elbow")
    ax.set_xlabel("Số lượng nhóm")
    ax.set_ylabel("Gía trị Inertia")

    # plt.show()
    # plt.cla()
    image = 'static/images_kmeans/'+ str(uuid.uuid4())[:8] +'_elbow.png'
    plt.savefig(image)

    return render_template('kmeans/elbow.html', data=data.to_html(classes='table table-striped', header=False, index=False), url='/'+image)

if __name__ == '__main__':
    app.run(debug=True)


