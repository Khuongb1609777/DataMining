#   Nguyen Nhat Khuong B1609777
#   Nguyen Hai Anh B1609759
#   Nguyen Dang Khoa B1611126
#   Tran Nam Duong B1609765


#   Import library Flask, Pandas, Seaborn, Numpy, Os, UUID..
from flask import Flask, render_template, request, url_for, session
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import csv
from werkzeug.utils import secure_filename
from sklearn.cluster import KMeans
import uuid #   Random Short Id
import os
import random
import matplotlib
matplotlib.use('svg')

#   Create UPLOAD_FOLDER is static/uploads
UPLOAD_FOLDER = 'static/uploads' #  Location is saving uploaded
ALLOWED_EXTENSIONS = {'csv','json','xlsx'} #  Kind of file

app = Flask(__name__)


app.secret_key = b'_5#y2L"F4Q8z\n\xec]/' #  Secret key of Session
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#   Create index.html for website
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



#--------------------------------------------------------------------------------------------------------
#   SVM_start

#   create folder SVM for SVM algorithm
@app.route('/SVM', methods=['GET', 'POST'])
def SVM_index():
    return render_template('SVM/index.html')
    #   index.html is upload file of SVM algorithm

#   create folder SVM/data for uploads file
@app.route('/SVM/data', methods=['GET', 'POST'])
def SVM_data():
    if request.method == 'POST':
        #   check if the post request has the file part
        if 'file' not in request.files:
            errors = 'No file part! Please choose 1 file csv !'
            return render_template('SVM/data_SVM.html', errors=errors)
        file = request.files['file']

        #   if user does not select file, browser also
        #   submit an empty part without filename
        if file.filename == '':
            errors = 'No selected file'
            return render_template('SVM/data_SVM.html', errors=errors)
        #   check file = ""
        
        if file and allowed_file(file.filename):
            #   get file_name 
            filename = secure_filename(file.filename)

            #   get file_path
            file_path = os.path.join(app.config['UPLOAD_FOLDER'],"SVM_" + str(uuid.uuid4())[:8] + '_' + filename)

            #   save file
            file.save(file_path)

            #   create session file
            session['csvfile'] = file_path #    Save path file to session

            file_tail = file_path.split(".")[1]

            if file_tail == "csv":
                data = pd.read_csv(file_path)
            elif file_tail == "xlsx":
                data = pd.read_excel(file_path)
            else:
                data = pd.read_excel(file_path)

            #   m is columns_number 
            m = data.shape[1]
            #   Create array_col_data for data.iloc
            array_col_data = []
            for i in range(m):
                array_col_data.append(i)
            #   Create column_names_attribute for name of attribute (col_1, col_2,....)
            column_names_attribute = []
            for i in range(m):
                col = "col_" + str(i)
                column_names_attribute.append(col)
            #   Read data and append column_name

            file_tail = file_path.split(".")[1]

            if file_tail == "csv":
                data = pd.read_csv(file_path,names = column_names_attribute)
            elif file_tail == "xlsx":
                data = pd.read_excel(file_path,names = column_names_attribute)
            else:
                data = pd.read_excel(file_path,names = column_names_attribute)
            #   Return data_SVM.html 
            return render_template('SVM/data_SVM.html', data=data.to_html(table_id='myTable', classes='table table-striped', header=True, index=False),array_col_data = array_col_data, m=m)

#   Create model result -> Accuracy 
@app.route('/SVM/result', methods=['GET', 'POST'])
def SVM_result():
    #   Get file_path from session
    file_path = session.get('csvfile')

    #   Read file with pandas
    file_tail = file_path.split(".")[1]

    if file_tail == "csv":
        data = pd.read_csv(file_path)
    elif file_tail == "xlsx":
        data = pd.read_excel(file_path)
    else:
        data = pd.read_excel(file_path)

    #   Get columns_number of dataset
    m = data.shape[1]

    #   Set columns_name
    column_names_attribute = []
    for i in range(m):
        col = "col_" + str(i)
        column_names_attribute.append(col)

    #   Read file append column_names
    file_tail = file_path.split(".")[1]

    if file_tail == "csv":
        data = pd.read_csv(file_path,names = column_names_attribute)
    elif file_tail == "xlsx":
        data = pd.read_excel(file_path,names = column_names_attribute)
    else:
        data = pd.read_excel(file_path,names = column_names_attribute)

    #   Request column_label from form 
    col_label = (request.form.get('column_label'))

    #   type of col_label[i] = str, we have type of col_label[i] = int, we can use .iloc with it
    col_label = int(col_label)

    #   set session col_label
    session['col_label'] = col_label

     #   Request column_data from form 
    col_data = (request.form.getlist('column_data'))
  
    #   type of col_data[i] = str, we have type of col_data[i] = int, we can use .iloc with it
    for i in range(len(col_data)):
        col_data[i] = int(col_data[i])

    #   Remove col_label
    if col_label in col_data:
        col_data.remove(col_label)

    #   set session col_data
    session['col_data'] = col_data

    #   Request kernel from form (linear, rbf, polyminal or sigmoid)
    select_kernel = request.form.get('select_kernel')

    #   create session kernel
    session['kernel'] = select_kernel

    #   Get data 
    X = data.iloc[:,col_data]

    mx = X.shape[1]

    #   Get Label
    Y = data.iloc[:,col_label]

    #Full data for model (x + y)
    datafull = pd.concat([X, Y],axis=1)

    #   Import model, select data train and test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 40)

    # import model svm
    from sklearn import svm

    #   Create Model
    clf = svm.SVC(kernel = select_kernel ) 
    clf.fit(X_train, y_train)

    #   predict label for X_test
    y_pred = clf.predict(X_test)

    #   Accuracy
    from sklearn import metrics
    accuracy =100* metrics.accuracy_score(y_test, y_pred)
   
    #   Return result_SVM.html
    return render_template('SVM/result_SVM.html',col_data = col_data, datafull = datafull.to_html(table_id='myTable1', classes='table table-striped', header=True, index=False),mx = mx, acc= accuracy)

#   Create model predict
@app.route('/SVM/result_end', methods=['GET', 'POST'])
def SVM_test_new():
    #   get new_element form form
    el_new = (request.form.getlist('element_new'))
    #   Create new_element
    element_new = []
    for i in range(len(el_new)):
        element_new.append(float(el_new[i]))

    element_new = pd.DataFrame(element_new).T
    element_new1 = element_new.iloc[0].values

    #   Create session file_path
    file_path = session.get('csvfile')

    #   Create session col_data
    col_data = session.get('col_data')

    #   Create session col_label
    col_label = session.get('col_label')

    #   Create session kernel
    select_kernel = session.get('kernel')

    #   Read data
    file_tail = file_path.split(".")[1]

    if file_tail == "csv":
        data = pd.read_csv(file_path)
    elif file_tail == "xlsx":
        data = pd.read_excel(file_path)
    else:
        data = pd.read_excel(file_path)


    X = data.iloc[:,col_data]
    Y = data.iloc[:,col_label]

    #Import model
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

    return render_template('SVM/predict_SVM.html',element_new = element_new1, class_new = class_new, acc=accuracy)


#SVM_end
#---------------------------------------------------------------------------------------------------------



#---------------------------------------------------------------------------------------------------------
#Decision_Tree start

@app.route('/DecisionTree', methods=['GET', 'POST'])
def DTree_index():
    return render_template('DecisionTree/index.html')

@app.route('/DecisionTree/data', methods=['GET', 'POST'])
def DTree_data():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            errors = 'No file part! Please choose 1 file csv !'
            return render_template('DecisionTree/data_DecisionTree.html', errors=errors)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            errors = 'No selected file'
            return render_template('DecisionTree/data_DecisionTree.html', errors=errors)
        # check file = ""
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'],"DTree_" + str(uuid.uuid4())[:8] + '_' + filename)
            file.save(file_path)

            #   Create session save path file
            session['csvfile'] = file_path #Save path file to session

            #   Read file with pandas
            file_tail = file_path.split(".")[1]

            if file_tail == "csv":
                data = pd.read_csv(file_path)
            elif file_tail == "xlsx":
                data = pd.read_excel(file_path)
            else:
                data = pd.read_excel(file_path)
        
            m = data.shape[1]
            #   Col_data is data for model
            array_col_data = []
            for i in range(m):
                array_col_data.append(i)

            print (array_col_data)

            #Create column name (col_1, col_2,...)
            column_names_attribute = []
            for i in range(m):
                col = "col_" + str(i)
                column_names_attribute.append(col)

            #   Read file with file_names
            file_tail = file_path.split(".")[1]

            if file_tail == "csv":
                data = pd.read_csv(file_path,names = column_names_attribute)
            elif file_tail == "xlsx":
                data = pd.read_excel(file_path,names = column_names_attribute)
            else:
                data = pd.read_excel(file_path,names = column_names_attribute)

            return render_template('DecisionTree/data_DecisionTree.html', data=data.to_html(table_id='myTable', classes='table table-striped', header=True, index=False),array_col_data = array_col_data, m=m)

@app.route('/DecisionTree/result_DecisionTree', methods=['GET', 'POST'])
def DTree_result():
    #   Get file_path from session
    file_path = session.get('csvfile')
    file_tail = file_path.split(".")[1]

    if file_tail == "csv":
        data = pd.read_csv(file_path)
    elif file_tail == "xlsx":
        data = pd.read_excel(file_path)
    else:
        data = pd.read_excel(file_path)
    m = data.shape[1]

    #   Create columns_name
    column_names_attribute = []

    for i in range(m):
        col = "col_" + str(i)
        column_names_attribute.append(col)
    #   Read file with column_names
    file_tail = file_path.split(".")[1]

    if file_tail == "csv":
        data = pd.read_csv(file_path,names = column_names_attribute)
    elif file_tail == "xlsx":
        data = pd.read_excel(file_path,names = column_names_attribute)
    else:
        data = pd.read_excel(file_path,names = column_names_attribute)

    #   Request column_label from form 
    col_label = (request.form.get('column_label'))

    #   type of col_label[i] = str, we have type of col_label[i] = int, we can use .iloc with it
    col_label = int(col_label)

     #   Request column_data from form 
    col_data = (request.form.getlist('column_data'))
  
    #   type of col_data[i] = str, we have type of col_data[i] = int, we can use .iloc with it
    for i in range(len(col_data)):
        col_data[i] = int(col_data[i])

    #   Remove col_label
    if col_label in col_data:
        col_data.remove(col_label)

    #   set session col_data
    session['col_data'] = col_data

    #   Data
    X = data.iloc[:,col_data]
    mx = X.shape[1]

    #   Label
    Y = data.iloc[:,col_label]

    #   Full data for model (x+y)
    datafull = pd.concat([X, Y],axis=1)

    #   Import model, select data train and test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 40)

    from sklearn import tree

    #   Model
    clf = tree.DecisionTreeClassifier(max_depth=3)
     #max_depth = 3 là độ sâu của cây quyết định
    clf = clf.fit(X_train, y_train)

    #   predict label for X_test
    y_pred = clf.predict(X_test)

    #   Accuracy
    from sklearn import metrics
    accuracy =100* metrics.accuracy_score(y_test, y_pred)
    #Save accuracy in session
    session['acc'] = accuracy
   
    #
    return render_template('DecisionTree/result_DecisionTree.html',col_data = col_data, datafull = datafull.to_html(table_id='myTable1', classes='table table-striped', header=True, index=False),mx = mx, acc= accuracy)


@app.route('/DecisionTree/predict_DecisionTree', methods=['GET', 'POST'])
def DTree_test_new():
    #   Get new_element form form
    el_new = (request.form.getlist('element_new'))

    element_new = []
    for i in range(len(el_new)):
        element_new.append(float(el_new[i]))

    element_new = pd.DataFrame(element_new).T
    element_new1 = element_new.iloc[0].values

    #   Create session file_path
    file_path = session.get('csvfile')

    #   Create session col_data
    col_data = session.get('col_data')

    #   Create session col_label
    col_label = session.get('col_label')

    #   Create session accuracy
    accuracy = session.get('acc')

    #   Read file
    file_tail = file_path.split(".")[1]

    if file_tail == "csv":
        data = pd.read_csv(file_path)
    elif file_tail == "xlsx":
        data = pd.read_excel(file_path)
    else:
        data = pd.read_excel(file_path)

    #   Select data
    X = data.iloc[:,col_data]

    #   Select label
    Y = data.iloc[:,col_label]

    #   Create model
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 40)

    from sklearn import tree

    #   Model
    clf = tree.DecisionTreeClassifier(max_depth=3)
     #max_depth = 3 là độ sâu của cây quyết định
    clf = clf.fit(X_train, y_train)

    #   predict label for X_test
    y_pred = clf.predict(X_test)

    #   predict label for X_test
    class_new = clf.predict(element_new)

    return render_template('DecisionTree/predict_DecisionTree.html',element_new = element_new1, class_new = class_new, acc=accuracy)

#Decision_Tree End
#----------------------------------------------------------------------------------------------------------






#----------------------------------------------------------------------------------------------------------
#Naive_Bayes Start

@app.route('/NaiveBayes', methods=['GET', 'POST'])
def NaiveBayes_index():
    return render_template('NaiveBayes/index.html')

@app.route('/NaiveBayes/data', methods=['GET', 'POST'])
def NaiveBayes_data():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            errors = 'No file part! Please choose 1 file csv !'
            return render_template('NaiveBayes/data_NaiveBayes.html', errors=errors)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            errors = 'No selected file'
            return render_template('NaiveBayes/data_NaiveBayes.html', errors=errors)
        # check file = ""
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'],"NaiveBayes_" + str(uuid.uuid4())[:8] + '_' + filename)
            file.save(file_path)

            #   Create session file_path
            session['csvfile'] = file_path #Save path file to session

            #   Read file
            file_tail = file_path.split(".")[1]

            if file_tail == "csv":
                data = pd.read_csv(file_path)
            elif file_tail == "xlsx":
                data = pd.read_excel(file_path)
            else:
                data = pd.read_excel(file_path)
        
            m = data.shape[1]

            #   Columns_data for model
            array_col_data = []
            for i in range(m):
                array_col_data.append(i)

            #   Create column_name
            column_names_attribute = []
            for i in range(m):
                col = "col_" + str(i)
                column_names_attribute.append(col)

            #   Read data with column_names
            file_tail = file_path.split(".")[1]

            if file_tail == "csv":
                data = pd.read_csv(file_path,names = column_names_attribute)
            elif file_tail == "xlsx":
                data = pd.read_excel(file_path,names = column_names_attribute)
            else:
                data = pd.read_excel(file_path,names = column_names_attribute)

            return render_template('NaiveBayes/data_NaiveBayes.html', data=data.to_html(table_id='myTable', classes='table table-striped', header=True, index=False),array_col_data = array_col_data, m=m)

@app.route('/NaiveBayes/result_NaiveBayes', methods=['GET', 'POST'])
def NaiveBayes_result():

    #   Get file_path from session
    file_path = session.get('csvfile')

    #   Read file
    file_tail = file_path.split(".")[1]

    if file_tail == "csv":
        data = pd.read_csv(file_path)
    elif file_tail == "xlsx":
        data = pd.read_excel(file_path)
    else:
        data = pd.read_excel(file_path)
    m = data.shape[1]

    #   Create columns_names
    column_names_attribute = []

    for i in range(m):
        col = "col_" + str(i)
        column_names_attribute.append(col)

    #   Read file with columns_names
    file_tail = file_path.split(".")[1]

    if file_tail == "csv":
        data = pd.read_csv(file_path,names = column_names_attribute)
    elif file_tail == "xlsx":
        data = pd.read_excel(file_path,names = column_names_attribute)
    else:
        data = pd.read_excel(file_path,names = column_names_attribute)

    #   Request column_label from form 
    col_label = (request.form.get('column_label'))

    #   type of col_label[i] = str, we have type of col_label[i] = int, we can use .iloc with it
    col_label = int(col_label)

     #   Request column_data from form 
    col_data = (request.form.getlist('column_data'))
  
    #   type of col_data[i] = str, we have type of col_data[i] = int, we can use .iloc with it
    for i in range(len(col_data)):
        col_data[i] = int(col_data[i])
    #   Remove col_label
    if col_label in col_data:
        col_data.remove(col_label)

    #   set session col_data
    session['col_data'] = col_data

    #   Data
    X = data.iloc[:,col_data]
    mx = X.shape[1]

    #   Label
    Y = data.iloc[:,col_label]

    #  Full_data for model (X + y)
    datafull = pd.concat([X, Y],axis=1)

    #   Import model, select data train and test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 40)

    #   Create model
    from sklearn.naive_bayes import GaussianNB

    #   Model
    gnb = GaussianNB()
    gnb = gnb.fit(X_train, y_train)

    #   Predict x_test
    y_pred = gnb.predict(X_test)

    #   Accuracy
    from sklearn import metrics
    accuracy =100* metrics.accuracy_score(y_test, y_pred)

    #Save accuracy in session
    session['acc'] = accuracy
   
    return render_template('NaiveBayes/result_NaiveBayes.html',col_data = col_data, datafull = datafull.to_html(table_id='myTable1', classes='table table-striped', header=True, index=False),mx = mx, acc= accuracy)


@app.route('/NaiveBayes/predict_NaiveBayes', methods=['GET', 'POST'])
def NaiveBayes_test_new():
    #   Get new_element from form
    el_new = (request.form.getlist('element_new'))

    element_new = []
    for i in range(len(el_new)):
        element_new.append(float(el_new[i]))

    element_new = pd.DataFrame(element_new).T
    element_new1 = element_new.iloc[0].values

    #   Create session file_path
    file_path = session.get('csvfile')

    #   Create session col_data
    col_data = session.get('col_data')

    #   Create session col_label
    col_label = session.get('col_label')

    #   Create session accuracy
    accuracy = session.get('acc')

    #   Read file
    file_tail = file_path.split(".")[1]

    if file_tail == "csv":
        data = pd.read_csv(file_path)
    elif file_tail == "xlsx":
        data = pd.read_excel(file_path)
    else:
        data = pd.read_excel(file_path)

    #   X and y for model
    X = data.iloc[:,col_data]
    Y = data.iloc[:,col_label]

    #   Create model
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 40)

    from sklearn.naive_bayes import GaussianNB

    #   Model
    gnb = GaussianNB()
    gnb = gnb.fit(X_train, y_train)

    y_pred = gnb.predict(X_test)

    #   predict label for X_test
    class_new = gnb.predict(element_new)

    return render_template('NaiveBayes/predict_NaiveBayes.html',element_new = element_new1, class_new = class_new, acc=accuracy)

#Naive_Bayes end
#----------------------------------------------------------------------------------------------------------



#----------------------------------------------------------------------------------------------------------
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

            session['readFile'] = file_path #Save path file to session

            checkFile = file_path.split(".")[1] #Cut the file extension

            if(checkFile == "csv"):
                data = pd.read_csv(file_path)
            elif(checkFile == "xlsx"):
                data = pd.read_excel(file_path)
            elif(checkFile == "json"):
                data = pd.read_json(file_path)

            m = data.shape[1] #Get amount column

            return render_template('kmeans/data.html', data=data.to_html(table_id='myTable', classes='table table-striped', header=True, index=False), m=m)

@app.route('/kmeans/elbow', methods=['GET', 'POST'])
def kmeans_elbow():
    predColumn = session.get('predColumn') #Get the column 

    file_path = session.get('readFile')

    checkFile = file_path.split(".")[1] #Cut the file extension

    if(checkFile == "csv"):
        data = pd.read_csv(file_path)
    elif(checkFile == "xlsx"):
        data = pd.read_excel(file_path)
    elif(checkFile == "json"):
        data = pd.read_json(file_path)

    col = request.form.getlist('cot') #Get values of checkbox form from
    # col = np.array(col)
    session['predColumn'] = len(col) #Set column predict

    for i in range(len(col)):
        col[i] = int(col[i])

    # print(col)

    session['col'] = col #Save values of checkbox to session
    # col1 = col[0]
    # col2 = col[1]

    # session['col1'] = col1 #Save column to session
    # session['col2'] = col2 #Save column to session

    m = data.shape[1]
    haha = 0
    X = data.iloc[int(haha):, col].values

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
    # image = 'static/images/kmeans/'+ str(uuid.uuid4())[:8] +'_elbow.png'
    image = 'static/images_kmeans/elbow.svg'
    plt.savefig(image)
    plt.clf()

    return render_template('kmeans/elbow.html', data=data.to_html(classes='table table-striped', header=False, index=False), url='/'+image, predColumn=predColumn)

@app.route('/kmeans/result', methods=['GET', 'POST'])
def kmeans_result():
    valuesPred = request.form.getlist('attributePre')

    array_values = []
    for i in range(len(valuesPred)):
        array_values.append(float(valuesPred[i]))
    array_values = np.array([array_values])

    k = request.form.get('cluster')

    file_path = session.get('readFile')

    checkFile = file_path.split(".")[1] #Cut the file extension

    if(checkFile == "csv"):
        data = pd.read_csv(file_path)
    elif(checkFile == "xlsx"):
        data = pd.read_excel(file_path)
    elif(checkFile == "json"):
        data = pd.read_json(file_path)

    # col1  = session.get('col1')
    # col2  = session.get('col2')
    col = session.get('col')

    col1 = col[0]
    col2 = col[1]
    # print(col1, col2)
    haha = 0

    X = data.iloc[int(haha):, col].values

    # print(X)

    name_column = list(data.columns)

    if len(col) <= 2:
        name1 = name_column[int(col1)]
        name2 = name_column[int(col2)]

        km = KMeans(n_clusters=int(k))
        y_means = km.fit_predict(X)
        result = km.predict(array_values)

        data['Clusters'] = y_means

        for i in range(0, int(k)):
            data["Clusters"].replace({i: "C" + str(i+1)}, inplace=True)

        x_coordinates = [0, 1, 2, 3, 4, 5] #random color
        y_coordinates = [0, 1, 2, 3, 4, 5] #random color

        for i in range(0, int(k)):
            for x, y in zip(x_coordinates, y_coordinates): #random color
                rgb = (random.random(), random.random(), random.random())
            plt.scatter(X[y_means == i, 0], X[y_means == i, 1], s=15, marker='h', c=[rgb], label='Cluster ' + str(i+1))
        # plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], s=100, c='red', label='Nhóm 1')
        # plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], s=100, c='blue', label='Nhóm 2')
        # plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1], s=100, c='green', label='Nhóm 3')
        plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=100, marker='*',c='black', label='Centeroid')
        # plt.plot(plt.legend()x,y)

        plt.style.use('fivethirtyeight')
        
        plt.title('K Means Clustering', fontsize=20)

        plt.xlabel(name1)
        plt.ylabel(name2)
        
        plt.legend()
        
        plt.grid()
        # image = 'static/images/kmeans/'+ str(uuid.uuid4())[:8] +'_plot.png'
        image = 'static/images_kmeans/plot.svg'
        plt.savefig(image)

        return render_template('kmeans/result.html', data=data.to_html(table_id='myTable', classes='table table-striped', header=True, index=False), url='/'+image, result=result, array_values=array_values)
    else:
        km = KMeans(n_clusters=int(k))
        y_means = km.fit_predict(X)
        result = km.predict(array_values)

        data['Clusters'] = y_means

        for i in range(0, int(k)):
            data["Clusters"].replace({i: "C" + str(i+1)}, inplace=True)

        return render_template('kmeans/result.html', data=data.to_html(table_id='myTable', classes='table table-striped', header=True, index=False), result=result, array_values=array_values)


if __name__ == '__main__':
    app.run(debug=True)

# End && this is perfect code =)) ------------------------------------------------------------------------