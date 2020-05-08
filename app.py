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

UPLOAD_FOLDER = 'static/uploads' #Location is saving uploaded
ALLOWED_EXTENSIONS = {'csv'} #Kind of file

app = Flask(__name__)

app.secret_key = b'_5#y2L"F4Q8z\n\xec]/' #Secret key of Session
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#--------------------------------------------------------------------------------------------------------
#SVM_start

@app.route('/SVM', methods=['GET', 'POST'])
def SVM_index():
    return render_template('SVM/index.html')



@app.route('/SVM/data', methods=['GET', 'POST'])
def SVM_data():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            errors = 'No file part! Please choose 1 file csv !'
            return render_template('SVM/data.html', errors=errors)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            errors = 'No selected file'
            return render_template('SVM/data.html', errors=errors)
        # check file = ""
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'],"SVM_" + str(uuid.uuid4())[:8] + '_' + filename)
            file.save(file_path)

            session['csvfile'] = file_path #Save path file to session

            data = pd.read_csv(file_path)
        
            m = data.shape[1]

            column_names_attribute = []
            for i in range(m):
                col = "col_" + str(i)
                column_names_attribute.append(col)

            data = pd.read_csv(file_path,names = column_names_attribute)


            return render_template('SVM/data.html', data=data.to_html(table_id='myTable', classes='table table-striped', header=True, index=False), m=m)

@app.route('/SVM/result', methods=['GET', 'POST'])
def SVM_result():
    file_path = session.get('csvfile')
    data = pd.read_csv(file_path)
        
    m = data.shape[1]

    column_names_attribute = []

    for i in range(m):
        col = "col_" + str(i)
        column_names_attribute.append(col)

    data = pd.read_csv(file_path,names = column_names_attribute)

    #   Request column_data from form 
    col_data = (request.form.getlist('column_data'))
    #   type of col_data[i] = str, we have type of col_data[i] = int, we can use .iloc with it
    for i in range(len(col_data)):
        col_data[i] = int(col_data[i])
    #   set session col_data
    session['col_data'] = col_data


    #   Request column_label from form 
    col_label = (request.form.get('column_label'))
    #   type of col_label[i] = str, we have type of col_label[i] = int, we can use .iloc with it
    col_label = int(col_label)
    #   set session col_label
    session['col_label'] = col_label


    #   Request kernel from form (linear, rbf, polyminal or sigmoid)
    select_kernel = request.form.get('select_kernel')
    #   set session kernel
    session['kernel'] = select_kernel

    #   Data
    X = data.iloc[:,col_data]
    mx = X.shape[1]

    #   Label
    Y = data.iloc[:,col_label]

    datafull = pd.concat([X, Y],axis=1)

    #   Import model, select data train and test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 40)


    from sklearn import svm

    #   Model
    clf = svm.SVC(kernel = select_kernel ) 
    clf.fit(X_train, y_train)

    #   predict label for X_test
    y_pred = clf.predict(X_test)

    #   Accuracy
    from sklearn import metrics
    accuracy =100* metrics.accuracy_score(y_test, y_pred)
   


    return render_template('SVM/result.html',col_data = col_data, datafull = datafull.to_html(table_id='myTable1', classes='table table-striped', header=True, index=False),mx = mx, acc= accuracy)


@app.route('/SVM/result_end', methods=['GET', 'POST'])
def SVM_test_new():
    el_new = (request.form.getlist('element_new'))

    element_new = []
    for i in range(len(el_new)):
        element_new.append(float(el_new[i]))

    element_new = pd.DataFrame(element_new).T
    print("sdfaasddddddddddddddddddddddddddddddd")
    print(element_new)

    element_new1 = element_new.iloc[0].values



    print("-------------------------------------------")
    print(element_new1)


    file_path = session.get('csvfile')
    col_data = session.get('col_data')
    col_label = session.get('col_label')
    select_kernel = session.get('kernel')


    data = pd.read_csv(file_path)

    X = data.iloc[:,col_data]
    Y = data.iloc[:,col_label]

    datafull = pd.concat([X, Y],axis=1)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 40)

    from sklearn import svm

    #   Model
    clf = svm.SVC(kernel = select_kernel ) 
    clf.fit(X_train, y_train)

    #   predict label for X_test
    y_pred = clf.predict(X_test)

    #   Accuracy
    from sklearn import metrics
    accuracy = 100 * metrics.accuracy_score(y_test, y_pred)

    #   predict label for X_test
    class_new = clf.predict(element_new)

    return render_template('SVM/result_end.html',element_new = element_new1, class_new = class_new, acc=accuracy)








#SVM_end
#---------------------------------------------------------------------------------------------------------
#Kmeans_start






@app.route('/kmeans', methods=['GET', 'POST'])
def kmeans_index():
    return render_template('Kmeans/index.html')

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


