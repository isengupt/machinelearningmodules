import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn import tree, preprocessing
import sklearn.ensemble as ske
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


import statsmodels.api as sm
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
def set_label(cat):
    cause = 0
    natural = ['Lightning']
    accidental = ['Structure','Fireworks','Powerline','Railroad','Smoking','Children','Campfire','Equipment Use','Debris Burning']
    malicious = ['Arson']
    other = ['Missing/Undefined','Miscellaneous']
    if cat in natural:
        cause = 1
    elif cat in accidental:
        cause = 2
    elif cat in malicious:
        cause = 3
    else:
        cause = 4
    return cause


class Connecter:
  def __init__(self, path):
    self.path = path
    self.cnx = self.connect()

  def checkFiles(self):
    print(check_output(["ls", "./input"]).decode("utf8"))

  def connect(self):
    return sqlite3.connect(self.path)

  def createDf(self):
    self.df = pd.read_sql_query("SELECT FIRE_YEAR,STAT_CAUSE_DESCR,LATITUDE,LONGITUDE,STATE,DISCOVERY_DATE,FIRE_SIZE FROM 'Fires'", self.cnx)
    self.df['DATE'] = pd.to_datetime(self.df['DISCOVERY_DATE'] - pd.Timestamp(0).to_julian_date(), unit='D')
    #change data to julien format
    print(self.df.head())

    #save copy of df
    self.df['MONTH'] = pd.DatetimeIndex(self.df['DATE']).month
    self.df['DAY_OF_WEEK'] = self.df['DATE'].dt.day_name()
    self.df_orig = self.df.copy() 
    print(self.df.head())


    #turn labels into numerical values
    le = preprocessing.LabelEncoder()
    self.df['STAT_CAUSE_DESCR'] = le.fit_transform(self.df['STAT_CAUSE_DESCR'])
    self.df['STATE'] = le.fit_transform(self.df['STATE'])
    self.df['DAY_OF_WEEK'] = le.fit_transform(self.df['DAY_OF_WEEK'])
    print(self.df.head())
  
  def selectFeatures(self, feature_cols, target_col):
      self.X = self.df[feature_cols] 
      self.y = self.df[target_col]

  def dropNulls(self):
    self.df = self.df.drop('DATE',axis=1)
    self.df = self.df.dropna()

  def splitData(self):
    self.X = self.df.drop(['STAT_CAUSE_DESCR'], axis=1).values
    self.y = self.df['STAT_CAUSE_DESCR'].values
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y,test_size=0.3, random_state=0) 

  def randomForest(self):
    self.clf_rf = ske.RandomForestClassifier(n_estimators=50)
    self.clf_rf = self.clf_rf.fit(self.X_train, self.y_train)
    print(self.clf_rf.score(self.X_test,self.y_test))

  def labelledRandomForest(self):
    self.df['LABEL'] = self.df_orig['STAT_CAUSE_DESCR'].apply(lambda x: set_label(x)) 
    self.df = self.df.drop('STAT_CAUSE_DESCR',axis=1)
    print(self.df.head())

    self.X = self.df.drop(['LABEL'], axis=1).values
    self.y = self.df['LABEL'].values
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y,test_size=0.3, random_state=0)
    self.clf_rf = ske.RandomForestClassifier(n_estimators=50)
    self.clf_rf = self.clf_rf.fit(self.X_train, self.y_train)
    print(self.clf_rf.score(self.X_test,self.y_test))
 
  def decisionTree(self):
    self.df['LABEL'] = self.df_orig['STAT_CAUSE_DESCR'].apply(lambda x: set_label(x)) 
    self.df = self.df.drop('STAT_CAUSE_DESCR',axis=1)
    print(self.df.head())

    self.X = self.df.drop(['LABEL'], axis=1).values
    self.y = self.df['LABEL'].values
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=1) # 70% training and 30% test
    self.clf = DecisionTreeClassifier()

    
    self.clf = self.clf.fit(self.X_train,self.y_train)

    
    self.y_pred = self.clf.predict(self.X_test)
    print(self.y_pred)
    print("Accuracy:",metrics.accuracy_score(self.y_test, self.y_pred))


  def elbowMethod(self):

    self.df['LABEL'] = self.df_orig['STAT_CAUSE_DESCR'].apply(lambda x: set_label(x)) 
    self.df = self.df.drop('STAT_CAUSE_DESCR',axis=1)
    print(self.df.head())

    self.X = self.df
    self.y = self.df['LABEL']


    cols = self.X.columns
    cs = []
    for i in range(1, 11):
        self.kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
        self.kmeans.fit(self.X)
        cs.append(self.kmeans.inertia_)

    print(cs)

  def kMeans(self):
    self.df['LABEL'] = self.df_orig['STAT_CAUSE_DESCR'].apply(lambda x: set_label(x)) 
    self.df = self.df.drop('STAT_CAUSE_DESCR',axis=1)
    print(self.df.head())

    self.X = self.df
    self.y = self.df['LABEL']


    cols = self.X.columns


    ms = MinMaxScaler()

    self.X = ms.fit_transform(self.X)
    self.X = pd.DataFrame(self.X, columns=[cols])
    print(self.X.head())

    self.kmeans = KMeans(n_clusters=4, random_state=0) 

    self.kmeans.fit(self.X)

    print(self.kmeans.cluster_centers_)
    print(self.kmeans.inertia_)

    labels = self.kmeans.labels_


    self.correct_labels = sum(self.y == labels)

    print("Result: %d out of %d samples were correctly labeled." % (self.correct_labels, self.y.size))

  def GLM(self, regressor):

    self.df['LABEL'] = self.df_orig['STAT_CAUSE_DESCR'].apply(lambda x: set_label(x)) 
    self.df = self.df.drop('STAT_CAUSE_DESCR',axis=1)
    print(self.df.head())

    self.X = self.df.drop(['LABEL'], axis=1).values
    self.y = self.df['LABEL'].values
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y,test_size=0.3, random_state=0)

    if (regressor=="P"):
      poisson_training_results = sm.GLM(self.y_train, self.X_train, family=sm.families.Poisson()).fit()
      print(poisson_training_results.summary())

   


 
     





c1 = Connecter('./wildfires.sqlite')
c1.createDf()
c1.dropNulls()
c1.GLM("P")
#c1.elbowMethod()
#c1.decisionTree()
#c1.labelledRandomForest()
#c1.splitData()
#c1.randomForest()
