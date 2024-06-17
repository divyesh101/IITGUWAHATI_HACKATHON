import numpy as np
import pandas as pd
from collections import Counter
import scipy as sp
from scipy import stats 
import matplotlib.pyplot as plt

path=r"C:\Users\Nazmi\Downloads\dataset and all\training_set_features.csv"
dataframe=pd.read_csv(path)
pd.set_option('display.max_columns',50)  #This shows up to 50 columns in the data
print(dataframe.head())
print(dataframe.isnull())#This is providing a large amoubt of data , better to calculate proportions
print(dataframe.isnull().mean())
print(Counter(dataframe.marital_status))#helps to calculate all possible parameters within feature married
dataframe.groupby(['marital_status'])['marital_status'].count().sort_values(ascending=False).plot.bar()
plt.show()
print(dataframe['marital_status'].value_counts().index[0])
print(Counter(dataframe.rent_or_own))
dataframe.groupby(['rent_or_own'])['rent_or_own'].count().sort_values(ascending=False).plot.bar()
plt.show()
print(dataframe['rent_or_own'].value_counts().index[0])
print(Counter(dataframe.employment_status))
dataframe.groupby(['employment_status'])['employment_status'].count().sort_values(ascending=False).plot.bar()
plt.show()
print(dataframe['employment_status'].value_counts().index[0])
print(Counter(dataframe.education))
dataframe.groupby(['education'])['education'].count().sort_values(ascending=False).plot.bar()
plt.show()
print(dataframe['education'].value_counts().index[0])
print(Counter(dataframe.income_poverty))
dataframe.groupby(['income_poverty'])['income_poverty'].count().sort_values(ascending=False).plot.bar()
plt.show()
print(dataframe.isnull().sum())
# Find mode of xyz_concern
mode_value = dataframe['xyz_concern'].mode()[0]

# Fill NaN values in multiple columns
columns_to_fill_with_mode = [
    'xyz_concern', 'xyz_knowledge', 'behavioral_antiviral_meds', 'behavioral_avoidance',
    'behavioral_face_mask', 'behavioral_wash_hands', 'behavioral_large_gatherings',
    'behavioral_outside_home', 'behavioral_touch_face', 'doctor_recc_xyz', 
    'doctor_recc_seasonal', 'chronic_med_condition', 'child_under_6_months', 
    'health_worker', 'health_insurance', 'opinion_xyz_vacc_effective', 
    'opinion_xyz_risk', 'opinion_xyz_sick_from_vacc'
]

for column in columns_to_fill_with_mode:
    dataframe[column] = dataframe[column].fillna(mode_value)

# Fill specific columns with their respective modes
dataframe['opinion_seas_vacc_effective'] = dataframe['opinion_seas_vacc_effective'].fillna(dataframe['opinion_seas_vacc_effective'].mode()[0])
dataframe['opinion_seas_risk'] = dataframe['opinion_seas_risk'].fillna(mode_value)
dataframe['opinion_seas_sick_from_vacc'] = dataframe['opinion_seas_sick_from_vacc'].fillna(dataframe['opinion_seas_sick_from_vacc'].mode()[0])
dataframe['household_adults'] = dataframe['household_adults'].fillna(dataframe['household_adults'].mode()[0])
dataframe['household_children'] = dataframe['household_children'].fillna(dataframe['household_children'].mode()[0])

print(dataframe.isnull().sum())
dataframe = dataframe.drop(['employment_industry','employment_occupation'],axis=1)
print(dataframe.head())


path2=r"C:\Users\Nazmi\Downloads\dataset and all\training_set_labels.csv"
df2=pd.read_csv(path2)
print(df2.isnull().sum())
out= pd.merge(dataframe, df2, 
                   on='respondent_id', 
                   how='outer')

print(out.head())
X = out.iloc[:,:-2]
y = out['xyz_vaccine']
y1 = out['seasonal_vaccine']
print(y)
X = pd.get_dummies(out,columns=['age_group','education','race','sex','income_poverty','marital_status','rent_or_own','employment_status','hhs_geo_region','census_msa'])
pd.set_option('display.max_columns',100)
print(X)
X.drop(['age_group_18 - 34 Years','education_12 Years','race_Black','sex_Female','income_poverty_<= $75,000, Above Poverty','marital_status_Married','rent_or_own_Own','employment_status_Employed','hhs_geo_region_atmpeygn','census_msa_Non-MSA'], axis = 1,inplace = True) 
print(X)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.10,random_state=10)
model = DecisionTreeClassifier(max_depth=4)
print(model)
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
print(accuracy_score(y_test,y_predict))
print(X)
print(y.value_counts())
print(y1.value_counts())
model.predict([[0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,3.0,1.0,2.0,2.0,1.0,2.0,0.0,0.0,0,0,0,0,1,0,1,0,0,0,0,1,0,0,1,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0]])
print(classification_report(y_test,y_predict))
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
clf = LogisticRegression(solver="liblinear", random_state=0).fit(X_train, y_train)
roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])
print(roc_auc_score(y_train, clf.decision_function(X_train)))#ROC IS ABOUT 99%







