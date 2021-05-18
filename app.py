
#   Import library Flask, Pandas, Seaborn, Numpy, Os, UUID..
from flask import Flask, render_template, request, url_for, session
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import pickle
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


@app.route('/DecisionTree/predict', methods=['GET', 'POST'])
def DTree_test_new():
    #   Get new_element form form
    # parse_boolean = {"yes":1, "no":0}
    gender = int(request.form.get('gender')) # male/female
    age = int(request.form.get('age')) # int
    height = float(request.form.get('height')) # float
    weight = float(request.form.get('weight')) # int
    fhwo = int(request.form.get('fhwo')) # yes/no
    favc = int(request.form.get('favc')) # yes/no
    fcvc = int(request.form.get('fcvc')) # never/ some time/ always
    ncp = int(request.form.get('ncp')) # number
    caec = int(request.form.get('caec')) # no/sometimes/frequently/always
    smoke = int(request.form.get('smoke')) # yes/no
    ch2o = int(request.form.get('ch2o')) # 1/2/3
    print(ch2o)
    print(type(ch2o))
    scc = int(request.form.get('scc')) # yes/no
    faf = int(request.form.get('faf')) # I do not have / 1 or 2 days / week
    tue = int(request.form.get('tue')) # 0 - 2 hours / 3 - 5 hours / More than 5 hours
    calc = int(request.form.get('calc')) # I do not drink / sometimes / Frequently / Always
    mtrans = int(request.form.get('mtrans')) # Walking / Bike / Motorbike / Automobile / Public_Transportation
    new_record = [gender, age, height, weight, fhwo, favc, fcvc, ncp, caec, smoke, ch2o, scc, faf, tue, calc, mtrans]
    new_record = np.array(new_record)
    names = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
       'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE',
       'CALC', 'MTRANS']
    obesity_model = pickle.load(open("model_dtree.sav", 'rb'))
    new_data = pd.DataFrame([new_record], columns=names)
    result = obesity_model.predict(new_data)
    parse_obesity = {
        0: "Insufficient Weight", 
        1: "Normal Weight", 
        2: "Overweight Level I", 
        3: "Overweight Level II", 
        4: "Obesity Type I", 
        5: "Obesity Type II", 
        6: "Obesity Type III"
    }
    result_predict = parse_obesity[result[0]]
    return render_template('DecisionTree/result_DecisionTree.html',result = result_predict)

#Decision_Tree End
#----------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    app.run(debug=True)

#   Done project
# End && this is perfect code =)) ------------------------------------------------------------------------