from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import io
import base64

global uname
global X_train, X_test, y_train, y_test
accuracy, precision, recall, fscore = [], [], [], []
graph_data = []

dataset = pd.read_csv("Dataset/student-por.csv")

labels = np.unique(dataset['FinalResult'])
label_encoder = []
columns = dataset.columns
types = dataset.dtypes.values
for i in range(len(types)):
    name = types[i]
    if name == 'object': #finding column with object type
        le = LabelEncoder()
        dataset[columns[i]] = pd.Series(le.fit_transform(dataset[columns[i]].astype(str)))#encode all str columns to numeric
        label_encoder.append([columns[i], le])
dataset.fillna(0, inplace = True)
Y = dataset['FinalResult'].ravel()
dataset.drop(['FinalResult'], axis = 1,inplace=True)
X = dataset.values

sc = StandardScaler()
X = sc.fit_transform(X)

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5) #split dataset into train and test

def calculateMetrics(algorithm, predict, y_test):
    global accuracy, precision, recall, fscore
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(float(round(a, 2)))
    precision.append(float(round(p, 2)))
    recall.append(float(round(r, 2)))
    fscore.append(float(round(f, 2)))

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)
predict = knn.predict(X_test)
calculateMetrics("KNN", predict, y_test)

rf = RandomForestClassifier(n_estimators=1)
rf.fit(X_train, y_train)
predict = rf.predict(X_test)
calculateMetrics("Random Forest", predict, y_test)

svm_cls = svm.SVC(C=1.0, kernel="linear")
svm_cls.fit(X_train, y_train)
predict = svm_cls.predict(X_test)
calculateMetrics("SVM", predict, y_test)

gb_cls = GradientBoostingClassifier(n_estimators=10)
gb_cls.fit(X_train, y_train)
predict = gb_cls.predict(X_test)
calculateMetrics("Gradient Boosting", predict, y_test)

lr_cls = LogisticRegression()
lr_cls.fit(X_train, y_train)
predict = lr_cls.predict(X_test)
calculateMetrics("Logistic Regression", predict, y_test)

xg_cls = XGBClassifier(n_estimators=1)
xg_cls.fit(X_train, y_train)
predict = xg_cls.predict(X_test)
calculateMetrics("XGBoost", predict, y_test)

xg_cls = RandomForestClassifier()
xg_cls.fit(X, Y)
predict = xg_cls.predict(X_test)

def PredictPerformance(request):
    if request.method == 'GET':
       return render(request, 'PredictPerformance.html', {})

def PredictPerformanceAction(request):
    if request.method == 'POST':
        global label_encoder, rf, sc, labels, graph_data, gb_cls, xg_cls
        roll_no = request.POST.get('rollno', False)
        gender = request.POST.get('t1', False)
        age = float(request.POST.get('t2', False).strip())
        mother = request.POST.get('t3', False)
        father = request.POST.get('t4', False)
        reason = request.POST.get('t5', False)
        guardian = request.POST.get('t6', False)
        study = float(request.POST.get('t7', False).strip())
        failure = float(request.POST.get('t8', False).strip())
        school = request.POST.get('t9', False)
        family = request.POST.get('t10', False)
        paid = request.POST.get('t11', False)
        activity = request.POST.get('t12', False)
        internet = request.POST.get('t13', False)
        free = float(request.POST.get('t14', False).strip())
        out = float(request.POST.get('t15', False).strip())
        health = float(request.POST.get('t16', False).strip())
        absent = float(request.POST.get('t17', False).strip())
        score1 = float(request.POST.get('t18', False).strip())
        score2 = float(request.POST.get('t19', False).strip())
        score3 = float(request.POST.get('t20', False).strip())

        data = []
        data.append([gender,age,mother,father,reason,guardian,study,failure,school,family,paid,activity,internet,free,out,health,absent,score1,score2,score3])
        data = pd.DataFrame(data, columns=['sex','age','mother_job','father_job','reason','guardian','studytime','failures','schoolsup','famsup','paid','activities','internet','freetime','goout','health','absences','G1','G2','G3'])
        testData = data.values
        for i in range(len(label_encoder)-1):
            temp = label_encoder[i]
            name = temp[0]
            le = temp[1]
            data[name] = pd.Series(le.transform(data[name].astype(str)))#encode all str columns to numeric
        data.fillna(0, inplace = True)
        data = data.values
        data = sc.transform(data)
        predict = xg_cls.predict(data)[0]
        print(predict)
        print(labels)
        predict = int(predict)
        predict = labels[predict]
        graph_data.append(predict)
        status = ""
        if predict == "Poor":
            status = "Warning! Need  more focus & hardwork"
        output = "Roll No : "+roll_no+"<br/>Overall Predicted Performance ===> "+predict+"<br/>"+status
        context= {'data':output}
        return render(request, 'PredictPerformance.html', context)

def Graphs(request):
    if request.method == 'GET':
        global graph_data
        output = "All Students Performance Graph"
        gd = np.asarray(graph_data)
        unique, count = np.unique(gd, return_counts=True)
        plt.pie(count,labels=unique,autopct='%1.1f%%')
        plt.title('Performance Graph')
        plt.axis('equal')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()    
        context= {'data':output, 'img': img_b64}
        return render(request, 'ViewResult.html', context)    

def TrainML(request):
    if request.method == 'GET':
        output = ''
        output+='<table border=1 align=center width=100%><tr><th><font size="" color="black">Algorithm Name</th><th><font size="" color="black">Accuracy</th><th><font size="" color="black">Precision</th>'
        output+='<th><font size="" color="black">Recall</th><th><font size="" color="black">FSCORE</th></tr>'
        global accuracy, precision, recall, fscore
        algorithms = ['KNN', 'Random Forest', 'SVM', 'Gradient Boosting', 'Logistic Regression', 'XGBoost']
        for i in range(len(algorithms)):
            output+='<td><font size="" color="black">'+algorithms[i]+'</td><td><font size="" color="black">'+str(accuracy[i])+'</td><td><font size="" color="black">'+str(precision[i])+'</td><td><font size="" color="black">'+str(recall[i])+'</td><td><font size="" color="black">'+str(fscore[i])+'</td></tr>'
        output+= "</table></br>"
        df = pd.DataFrame([['KNN','Precision',precision[0]],['KNN','Recall',recall[0]],['KNN','F1 Score',fscore[0]],['KNN','Accuracy',accuracy[0]],
                           ['Random Forest','Precision',precision[1]],['Random Forest','Recall',recall[1]],['Random Forest','F1 Score',fscore[1]],['Random Forest','Accuracy',accuracy[1]],
                           ['SVM','Precision',precision[2]],['SVM','Recall',recall[2]],['SVM','F1 Score',fscore[2]],['SVM','Accuracy',accuracy[2]],
                           ['Gradient Boosting','Precision',precision[3]],['Gradient Boosting','Recall',recall[3]],['Gradient Boosting','F1 Score',fscore[3]],['Gradient Boosting','Accuracy',accuracy[3]],
                           ['Logistic Regression','Precision',precision[3]],['Logistic Regression','Recall',recall[3]],['Logistic Regression','F1 Score',fscore[3]],['Logistic Regression','Accuracy',accuracy[3]],
                           ['XGBoost','Precision',precision[3]],['XGBoost','Recall',recall[3]],['XGBoost','F1 Score',fscore[3]],['XGBoost','Accuracy',accuracy[3]],
                          ],columns=['Algorithms','Metrics','Value'])
        df.pivot_table(index="Algorithms", columns="Metrics", values="Value").plot(kind='bar', figsize=(8, 4))
        plt.title("All Algorithms Performance Graph")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()    
        context= {'data':output, 'img': img_b64}
        return render(request, 'ViewResult.html', context)        

def AdminLogin(request):
    if request.method == 'GET':
       return render(request, 'AdminLogin.html', {})

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def Aboutus(request):
    if request.method == 'GET':
       return render(request, 'Aboutus.html', {})

def LoadDataset(request):
    if request.method == 'GET':
       return render(request, 'LoadDataset.html', {})    

def AdminLoginAction(request):
    if request.method == 'POST':
        global uname
        username = request.POST.get('username', False)
        password = request.POST.get('password', False)
        if username == "admin" and password == "admin":
            context= {'data':'welcome '+username}
            return render(request, 'AdminScreen.html', context)
        else:
            context= {'data':'Invalid login details'}
            return render(request, 'AdminLogin.html', context)          

def LoadDatasetAction(request):
    if request.method == 'POST':
        myfile = request.FILES['t1'].read()
        fname = request.FILES['t1'].name
        if os.path.exists("StudentApp/static/"+fname):
            os.remove("StudentApp/static/"+fname)
        with open("StudentApp/static/"+fname, "wb") as file:
            file.write(myfile)
        file.close()
        dataset = pd.read_csv("StudentApp/static/"+fname)
        columns = dataset.columns
        dataset = dataset.values
        output='<table border=1 align=center width=100%><tr>'
        for i in range(len(columns)):
            output += '<th><font size="" color="black">'+columns[i]+'</th>'
        output += '</tr>'
        for i in range(len(dataset)):
            output += '<tr>'
            for j in range(len(dataset[i])):
                output += '<td><font size="" color="black">'+str(dataset[i,j])+'</td>'
            output += '</tr>'
        output+= "</table></br></br></br></br>"
        #print(output)
        context= {'data':output}
        return render(request, 'ViewResult.html', context)    







        
