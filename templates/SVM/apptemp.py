        data_upload = request.form['csvfile']
        data = pd.read_csv(data_upload)
        m = data.shape[1]
        #   Xây dựng tập dữ liệu X và nhãn y
        X = pd.DataFrame(data.iloc[0:100,[0,1]])
        y = pd.DataFrame(data.iloc[0:100,4])


        plot.plot(X.iloc[0:50, 0], X.iloc[0:50, 1], 'bo')
        plot.plot(X.iloc[50:100, 0], X.iloc[50:100, 1], 'rx')
        plot.savefig('static/images/plot1.png')



        #   Import các thư viện cần thiết
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 40)
        from sklearn import svm

        #   Xây dựng mô hình huấn luyện
        clf = svm.SVC(kernel='linear') 
        clf.fit(X_train, y_train)

        w = np.array(clf.coef_[0])
        b = clf.intercept_

        #   Dự đoán nhãn cho tập X_test
        y_pred = clf.predict(X_test)

        #   Đánh giá mô hình
        from sklearn import metrics
        accuracy = metrics.accuracy_score(y_test, y_pred)
        acc = accuracy * 100

        x1 = np.array([4, 7])
        x2 = (-b[0] - w[0]*x1)/w[1]
        plot.plot(x1,x2, 'y')

        x3 = np.array([4,7])
        x4 =  (-b[0] - w[0]*x3 + 0.1)/w[1]
        plot.plot(x3,x4,'b--')

        x5 = np.array([4,7])
        x6 = ( -b[0] - w[0]*x5 -0.1)/w[1]
        plot.plot(x5,x6,'r--')

        plot.plot(X.iloc[0:50, 0], X.iloc[0:50, 1], 'bo')
        plot.plot(X.iloc[50:100, 0], X.iloc[50:100, 1], 'rx')
        plot.legend()
        plot.grid()