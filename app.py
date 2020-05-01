from flask import Flask, render_template, request, url_for
import pandas as pd
import csv


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/SVM', methods=['GET', 'POST'])
def svm():
    return render_template('SVM/upload.php')



@app.route('/SVM/data', methods=['GET', 'POST'])
def SVM():
    if request.method == 'POST':
        file = request.form['csvfile']  
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
        
        import numpy as np
        import quadprog as qp
        import matplotlib.pyplot as plot
        import math 
        import pandas as pd
    
        
        data = pd.read_csv(data_upload)
        m = data.shape[1]
        #   Xây dựng tập dữ liệu X và nhãn y
        X = pd.DataFrame(data.iloc[0:100,[0,1]])
        y = pd.DataFrame(data.iloc[0:100,4])

        plot.plot(X.iloc[0:50, 0], X.iloc[0:50, 1], 'bo')
        plot.plot(X.iloc[50:100, 0], X.iloc[50:100, 1], 'rx')
        plot.legend()
        plot.grid()

    #return render_template('data.html',data=data.to_html(classes='table', header=False, index=False), x=X.to_html(classes='table', header=False, index=False), y=y.to_html(classes='table', header=False, index=False))
    return render_template('SVM/show-data.html',url='/static/images/plot1.png',data=data.to_html(classes='table', header=False, index=False),accuracy=acc,m=m)

@app.route('/result', methods=['GET', 'POST'])
def resultsvm():
    if request.method == 'POST':
        # f = request.form['filecsv']
        col = request.form.getlist('cot')
        data = data

        plt.savefig('static/images/plot.png')

        return render_template('result.html', url='/static/images/plot.png')


if __name__ == '__main__':
    app.run(debug=True)


