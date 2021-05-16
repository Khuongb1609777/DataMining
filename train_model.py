import numpy as np
import pandas as pd
import cv2 as cv2
import os
import numpy as np
import glob
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import pickle

dataset = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")

#  #   Column                          Non-Null Count  Dtype
# ---  ------                          --------------  -----
#  0   Gender                          2111 non-null   object
#  1   Age                             2111 non-null   float64
#  2   Height                          2111 non-null   float64
#  3   Weight                          2111 non-null   float64
#  4   family_history_with_overweight  2111 non-null   object
#  5   FAVC                            2111 non-null   object   Frequent consumption of high caloric food Thường xuyên tiêu thụ thực phẩm giàu calo
#  6   FCVC                            2111 non-null   float64
#  7   NCP                             2111 non-null   float64
#  8   CAEC                            2111 non-null   object
#  9   SMOKE                           2111 non-null   object
#  10  CH2O                            2111 non-null   float64
#  11  SCC                             2111 non-null   object
#  12  FAF                             2111 non-null   float64
#  13  TUE                             2111 non-null   float64
#  14  CALC                            2111 non-null   object
#  15  MTRANS                          2111 non-null   object
#  16  NObeyesdad                      2111 non-null   object

#   Ham chuyen doi du lieu co thu tu
def transform_encoder_level(dataset, feature_name, feature_values, feture_values_level):
    for i in range(len(dataset)):
        for j in range(len(feature_values)):
            if dataset[feature_name][i] == feature_values[j]:
                dataset[feature_name][i] = int(feture_values_level[j])
    return dataset


gender_values = ["Female", "Male"]
gender_values_preprocessing = [0, 1]

dataset = transform_encoder_level(
    dataset,
    "Gender",
    gender_values,
    gender_values_preprocessing,
)


family_history_with_overweight_values = ["yes", "no"]
family_history_with_overweight_values_pre = [1, 0]
dataset = transform_encoder_level(
    dataset,
    "family_history_with_overweight",
    family_history_with_overweight_values,
    family_history_with_overweight_values_pre,
)

dataset = transform_encoder_level(
    dataset,
    "FAVC",
    ["yes", "no"],
    [1, 0],
)

dataset = transform_encoder_level(
    dataset,
    "CAEC",
    ["no", "Sometimes", "Frequently", "Always"],
    [0, 1, 2, 3],
)

dataset = transform_encoder_level(
    dataset,
    "SMOKE",
    ["no", "yes"],
    [0, 1],
)

dataset = transform_encoder_level(
    dataset,
    "SCC",
    ["no", "yes"],
    [0, 1],
)

dataset = transform_encoder_level(
    dataset,
    "CALC",
    ["no", "Sometimes", "Frequently", "Always"],
    [0, 1, 2, 3],
)

dataset = transform_encoder_level(
    dataset,
    "CALC",
    ["no", "Sometimes", "Frequently", "Always"],
    [0, 1, 2, 3],
)

dataset = transform_encoder_level(
    dataset,
    "MTRANS",
    ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"],
    [0, 1, 2, 3, 4],
)

dataset = transform_encoder_level(
    dataset,
    "NObeyesdad",
    [
        "Obesity_Type_I",
        "Obesity_Type_III",
        "Obesity_Type_II",
        "Overweight_Level_II",
        "Overweight_Level_I",
        "Normal_Weight",
        "Insufficient_Weight",
    ],
    [4, 6, 5, 3, 2, 1, 0],
)

X = dataset.iloc[:,0:16]
y = dataset.iloc[:,16]
y=y.astype('int') #Vì y là obj nên k thể phân loại
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True, stratify=y)

#Xác định chỉ số C tốt nhất_-----------------------------
parameter_candidates = [
  {'C': [0.001, 0.01, 0.1, 1, 5, 10, 100, 1000], 'kernel': ['linear','rbf','poly']},]

model_svm = GridSearchCV(estimator=SVC(), param_grid=parameter_candidates, n_jobs=-1)
model_svm.fit(X_train, y_train)

print('Best score model_svm:', model_svm.best_score_) # Best score model_svm: 0.961807382903541
print('Best C model_svm:',model_svm.best_estimator_.C) # Best C model_svm: 1000
print(model_svm.best_params_) # {'C': 1000, 'kernel': 'linear'}

#----------------------------------------------------------------------------------------------------------------------

parameter_dtree = [{'criterion':['gini','entropy'],'splitter': ['best','random'], 'max_depth':[1,2,3,4,5,6,7,8,9,10,12,15],
                'min_samples_split':[0.01,0.1,0.2,0.3,0.5], 'min_samples_leaf':[0.01,0.1,0.2,0.3,0.5]}]
model_dtree = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=parameter_dtree, n_jobs=-1)
model_dtree.fit(X_train, y_train)

print('Best score model_dtree:', model_dtree.best_score_) # Best score model_dtree: 0.8953186477207227
print('Best C model_dtree:',model_dtree.best_estimator_) 
# Best C model_dtree: DecisionTreeClassifier(criterion='entropy', max_depth=12, min_samples_leaf=0.01, min_samples_split=0.01)
print(model_dtree.best_params_) 
# {'criterion': 'entropy', 'max_depth': 12, 'min_samples_leaf': 0.01, 'min_samples_split': 0.01, 'splitter': 'best'}

# max_depth : độ sâu của cây
# min_samples : số lượng mẫu tối thiểu để tách nút
# min_samples_leaf : Số lượng mẫu tối thiểu cần để có ở một nút lá

#----------------------------------------------------------------------------------------------------------------------
model_bayes = GaussianNB()
model_bayes.fit(X_train, y_train)
model_bayes.predict(X_test)


# titles_options = [("Ma trận nhầm lẫn trên tập số", None),
#                   ("Chuẩn hóa về 1", 'true')]
# for title, normalize in titles_options:
#     disp = plot_confusion_matrix(model_bayes,X_test ,y_test ,
#                                  display_labels=[0,1,2,3,4,5,6],
#                                  cmap= plt.cm.Blues,
#                                  normalize=normalize)
#     disp.ax_.set_title(title)
#     print(title)
#     print(disp.confusion_matrix)

model_list = {'model_svm':model_svm, 'model_dtree':model_dtree, 'model_naive_bayes':model_bayes}
for i in range(len(list(model_list))):
    titles_options = [("Ma trận nhầm lẫn trên model" + str(list(model_list.keys())[i]), None),
                    ("model " + str(list(model_list.keys())[i]) +" Chuẩn hóa 1", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(list(model_list.values())[i], X_test, y_test,
                                    display_labels=[0,1,2,3,4,5,6],
                                    cmap=plt.cm.Blues,
                                    normalize=normalize)
        disp.ax_.set_title(title)
        print(title)
        print(disp.confusion_matrix)
    plt.show()

for i in range(len(list(model_list))):
    y_predict = list(model_list.values())[i].predict(X_test)
    accuracy = accuracy_score(y_predict,y_test)
    print(str(list(model_list.keys())[i]),accuracy)

# model_svm 0.9512195121951219
# model_dtree 0.9182209469153515
# model_naive_bayes 0.6197991391678622

for i in range(len(model_list)):
    filename = str(list(model_list.keys())[i]) +".sav"
    pickle.dump(list(model_list.values())[i], open(filename, 'wb'))
