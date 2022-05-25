import streamlit as st 
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

dataset = pd.read_csv('heart_failure_clinical_records_dataset.csv')
dataset = dataset[dataset['ejection_fraction']<70]

x = dataset.iloc[:, [4,7,11]].values
y = dataset.iloc[:,-1].values

html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Heart Failure Analysis</h2>
    </div>
    """
st.markdown(html_temp,unsafe_allow_html=True)
st.write("""
# Explore different models
Which one is the best?
""")

classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('Logistic Regression','K-Nearest Neighbors','Support Vector', 'Decision Tree', 'Random Forest')
)


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'K-Nearest Neighbors':
       K = st.sidebar.slider('N_neighbors (6)', 3, 9)
       params['K'] = K
    elif clf_name == 'Support Vector':
        C = st.sidebar.slider('C (0.6)', 0.5, 0.9)
        params['C'] = C   
    elif clf_name == 'Decision Tree':
       M = st.sidebar.slider('Max_leaf_nodes (3)', 2, 9)
       params['M'] = M  
    elif clf_name == 'Random Forest':
       E = st.sidebar.slider('N_estimators (11)', 10, 29)
       params['E'] = E     
    st.sidebar.write('INPUTS :')   
    ejection_fraction=st.sidebar.slider("Ejection fraction",10,70)
    params["ejection_fraction"]=ejection_fraction
    serum_creatine = st.sidebar.text_input("Serum_creatine","0.0")
    params["serum_creatine"]=serum_creatine
    time=st.sidebar.slider("Time (follow-up period in days)",0,300)
    params["time"]=time
    return params

params=add_parameter_ui(classifier_name)    

def get_classifier(clf_name):
    clf = None
    if clf_name == 'Logistic Regression':
        clf = LogisticRegression()
    elif clf_name == 'Decision Tree':
        clf = DecisionTreeClassifier(max_leaf_nodes = params['M'], random_state=0, criterion='entropy')
    elif clf_name == 'K-Nearest Neighbors':
        clf = KNeighborsClassifier(n_neighbors=params['K'], metric='minkowski')   
    elif clf_name == 'Support Vector':
        clf = SVC(C=params['C'],random_state=0, kernel = 'rbf')     
    else:
        clf = RandomForestClassifier(n_estimators = params['E'], criterion='entropy', random_state=0)
    return clf


clf = get_classifier(classifier_name)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train1 = sc.fit_transform(X_train)
x_test1 = sc.transform(X_test)

prediction=""
if(classifier_name=='K-Nearest Neighbors' or classifier_name=='Support Vector'):
  clf.fit(x_train1,y_train)
  y_pred=clf.predict(x_test1)
  res=sc.transform([[params["ejection_fraction"],params["serum_creatine"],params["time"]]])
  prediction = clf.predict([[res[0][0],res[0][1],res[0][2]]])
else:   
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  prediction = clf.predict([[int(params["ejection_fraction"]),float(params["serum_creatine"]),int(params["time"])]])

acc = accuracy_score(y_test, y_pred)

st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy = ',acc)


if st.button("Predict"):
  #prediction = clf.predict([[params["ejection_fraction"],params["serum_creatine"],params["time"]]])
  if(prediction[0]==1):
    st.success('Patient deceased in the follow-up period')
  else:
    st.success('Patient not deceased in the follow-up period')
  mylist2 = ["Logistic Regression", "KNearestNeighbours","SupportVector","DecisionTree","RandomForest"]
  mylist =[0.8833,0.9333,0.9,0.95,0.95]
  fig = plt.figure()
  plt.rcParams['figure.figsize']=15,6 
  sns.set_style("darkgrid")
  ax = sns.barplot(x=mylist2, y=mylist, palette = "rocket", saturation =1.5)
  plt.xlabel("Classifier Models", fontsize = 15 ) 
  plt.ylabel("% of Accuracy", fontsize = 15)
  plt.title("Accuracy of different Classifier Models", fontsize = 17)
  plt.xticks(fontsize = 8, horizontalalignment = 'center', rotation = 12)
  plt.yticks(fontsize = 10)
  for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate(f'{height:.2%}', (x + width/2, y + height*1.02), ha='center', fontsize = 'large')
  st.pyplot(fig)
