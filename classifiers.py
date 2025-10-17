import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

data = pd.read_parquet('training.parquet')
cols=len(data.columns)
nfeatures = cols-1
X = data.iloc[:,0:nfeatures]  
y = data.iloc[:,-1]
print(len(data),cols)

def classifierxgboost(X,y):
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    model = XGBClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    matrix = confusion_matrix(y_test, y_pred)
    print(matrix)
    report = classification_report(y_test, y_pred)
    print(report)

def GaussNB(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    matrix = confusion_matrix(y_test, y_pred)
    print(matrix)
    report = classification_report(y_test, y_pred)
    print(report)

def DeepModel(X,y):
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    model = Sequential([
        Dense(32, input_dim=nfeatures, activation='relu'),  # Input layer
        Dense(16, activation='relu'),               # Hidden layer
        Dense(1, activation='sigmoid')              # Output layer
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    y_pred = model.predict(X_test)
    matrix = confusion_matrix(y_test, y_pred)
    print(matrix)
    report = classification_report(y_test, y_pred)
    print(report)

#Rule Based (Rough set) Classifier
def induce_rules(X, y):
    rules = {}
    for i in range(len(X)):
        condition = tuple(X.iloc[i])
        decision = y.iloc[i]
        rules[condition] = decision
    return rules

def predict(X_test, rules):
    y_pred = []
    for i in range(len(X_test)):
        condition = tuple(X_test.iloc[i])
        if condition in rules:
            y_pred.append(rules[condition])
        else:
            # If no exact match, use majority class as fallback
            y_pred.append(max(set(rules.values()), key=list(rules.values()).count))
    return y_pred
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
rules = induce_rules(X_train, y_train)
y_pred = predict(X_test, rules)
print("Classification Report:")
print(classification_report(y_test, y_pred))
'''
#classifierxgboost(X,y)
#GaussNB(X,y)
#DeepModel(X,y)

def RSTClassifier(X,y,targetclass=y[0]):
    print(targetclass)
    dc=pd.Series(y)
    cm=dc.groupby(dc).apply(lambda px: px.index.tolist()).to_dict()
    D=set(cm[targetclass])
    #print(D)
    purity={}
    headers = X.keys().tolist()

    for j in range(nfeatures):
        header=headers[j]
        hdata={}
        #print("Columns:",j,header)
        dp=pd.Series(X.iloc[:,j])
        md=dp.groupby(dp).apply(lambda px: px.index.tolist()).to_dict()
        K=md.keys()
        #print(K)
        for z in K:
            common_elements=set(md[z]) & D
            val=round(len(common_elements)/len(md[z]),3)
            #print(z,len(md[z]),len(common_elements),val)
            hdata[z]=val
        purity[header]=hdata
    #print(purity['urgent'][0])
    return purity

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

pur=RSTClassifier(X_train,y_train,'Benign')
#print(pur)

def RST_AmbigiousSample(X,pur):
    headers = X.keys().tolist()
    print(headers,len(headers),len(X))
    ambigioussamples={}
    for i in range(len(X)):
        flag=0
        sumpurity=0
        for j in range(len(headers)):
            key=X.iloc[i][headers[j]]
            if (pur[headers[j]][key] == 1 or pur[headers[j]][key] == 0):
                flag=1
                break
            else:
                sumpurity+=pur[headers[j]][key]
        if (flag == 0):
            Avgpurity=round(sumpurity/len(headers),3)
            ambigioussamples[i]=Avgpurity
    return ambigioussamples

AmbSample=RST_AmbigiousSample(X_train,pur)
print(len(AmbSample))
    
print("Mahendra")
