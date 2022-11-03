## pipreqs (to create requirement file needed for streamshare)
## cd to folder
## streamlit run classification_streamlit_app.py
########################## Initialization #####################

import streamlit as st

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib
import seaborn as sns 
import plotly as py
import plotly.graph_objs as go
import pickle

import warnings
import os
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('max_colwidth', 150)
st.set_page_config(layout="wide")


@st.cache(hash_funcs={dict: lambda _: None})
def load_data():
	Data = 0
	XGB_model = 0
	RF_model = 0
	LR_model = 0
	final_Data = 0
	
	Data = pd.read_csv ("train.csv")  

	XGB_model = pickle.load( open('xgb.pkl', 'rb'))
	RF_model = pickle.load(open('RF.pkl', 'rb'))
	LR_model = pickle.load(open('LR.pkl', 'rb'))
	
	
	
	final_Data = Data.copy()
	final_Data = final_Data.dropna()
	final_Data.drop(['Loan_ID','Gender','Dependents','Self_Employed'],axis=1,inplace=True)
	final_Data = final_Data.reset_index(drop=True)
	
	from sklearn.preprocessing import LabelEncoder  
	## sometimes better to use map rather than labelencoder because
	le = LabelEncoder()

	final_Data.Married = le.fit_transform(final_Data.Married)
	final_Data.Education = le.fit_transform(final_Data.Education)
	final_Data.Credit_History = le.fit_transform(final_Data.Credit_History)
	final_Data.Property_Area = le.fit_transform(final_Data.Property_Area)
	final_Data.Loan_Status = le.fit_transform(final_Data.Loan_Status)

	return Data,XGB_model,RF_model,LR_model,final_Data

Data,XGB_model,RF_model,LR_model,final_Data  = load_data()



# ###################### Data Calculations #####################
fig_num = 1
df = Data


@st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
def load_matplotlib_figure(df):

	fig_num = 1

	data_dist = plt.figure(fig_num, figsize = (15 , 5))
	sns.countplot(df['Loan_Status']);
	fig_num +=1

	credit_dist = plt.figure(fig_num,figsize = (10 , 5))
	sns.countplot(df['Credit_History']);
	fig_num +=1

	credit_dist_data = plt.figure(fig_num,figsize = (10 , 5))
	sns.countplot(x='Loan_Status', hue='Credit_History', data=df);
	fig_num +=1

	gender_dist =plt.figure(fig_num,figsize = (10 , 5))
	sns.countplot(df['Gender']);
	fig_num +=1

	gender_dist_data = plt.figure(fig_num,figsize = (10 , 5))
	sns.countplot(x='Loan_Status', hue='Gender', data=df);
	fig_num +=1

	married_dist = plt.figure(fig_num,figsize = (10 , 5))
	sns.countplot(df['Married']);
	fig_num +=1

	married_dist_data = plt.figure(fig_num,figsize = (10 , 5))
	sns.countplot(x='Loan_Status', hue='Married', data=df);
	fig_num +=1


	dependents_dist = plt.figure(figsize = (10 , 5))
	sns.countplot(df['Dependents']);
	fig_num +=1

	dependents_dist_data = plt.figure(figsize = (10 , 5))
	sns.countplot(x='Loan_Status', hue='Dependents', data=df);
	fig_num +=1

	# first if Dependents = 0 , we got higher chance to get a loan ((very hight chance))
	# good feature


	education_dist  = plt.figure(fig_num,figsize = (10 , 5))
	sns.countplot(df['Education']);
	fig_num +=1

	education_dist_data = plt.figure(fig_num,figsize = (10 , 5))
	sns.countplot(x='Loan_Status', hue='Education', data=df);
	fig_num +=1


	employed_dist =plt.figure(figsize = (10 , 5))
	sns.countplot(df['Self_Employed']);
	fig_num +=1

	employed_dist_data =plt.figure(figsize = (10 , 5))
	sns.countplot(x='Loan_Status', hue='Self_Employed', data=df);
	fig_num +=1


	propert_dist = plt.figure(fig_num,figsize = (10 , 5))
	sns.countplot(df['Property_Area']);
	fig_num +=1

	propert_dist_data = plt.figure(fig_num,figsize = (10 , 5))
	sns.countplot(x='Loan_Status', hue='Property_Area', data=df);
	fig_num +=1
	# first if Dependents = 0 , we got higher chance to get a loan ((very hight chance))
	# good feature


	income_dist = plt.figure(fig_num,figsize = (10 , 5))
	plt.scatter(df['ApplicantIncome'], df['Loan_Status']);
	fig_num +=1

	# parameter_distribution = plt.figure(fig_num , figsize = (15 , 6))
	# n = 0 
	# for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
	#     n += 1
	#     plt.subplot(1 , 3 , n)
	#     plt.subplots_adjust(hspace =0.5 , wspace = 0.5)
	#     sns.distplot(df[x] , bins = 20)
	#     plt.title('Distplot of {}'.format(x))

	# fig_num +=1


	
	# gender_plot = plt.figure(fig_num , figsize = (15 , 5))
	# sns.countplot(y = 'Gender' , data = df)	
	# fig_num +=1


	# pairplot = sns.pairplot(df1)



	return data_dist,credit_dist,credit_dist_data,gender_dist,gender_dist_data,married_dist,married_dist_data,dependents_dist,dependents_dist_data,propert_dist,employed_dist,employed_dist_data,propert_dist_data,income_dist,education_dist,education_dist_data

data_dist,credit_dist,credit_dist_data,gender_dist,gender_dist_data,married_dist,married_dist_data,dependents_dist,dependents_dist_data,propert_dist,employed_dist,employed_dist_data,propert_dist_data,income_dist,education_dist,education_dist_data= load_matplotlib_figure(df)

# ########################## Page UI #####################

# ### Overview of Project
tab1, tab2, tab3 = st.tabs([ "WebApp","Project Overview", "Methodology"])

with tab1:
	" ## Binary Classification - Loan Approval Prediction "
	" This web app predicts if your home loan will be sanctioned based on the personal information. It will show the prediction done by multiple models"
	
	"---"
	col_1, col_2, col_3,col_4 = st.columns(4)

	with col_1:
		#name = st.text_input("What is your name ?")
		Income = st.slider('Applicants Monthly Income in $', 0, 20000, 0)
		CoIncome= st.slider('Co-Applicants Monthly Income in $', 0,10000, 0)
	with col_2:
		Married = st.selectbox( 'Marital Status', ('UnMarried', 'Married'))
		Education = st.selectbox( 'Education', ('Not Graduate', 'Graduate'))
	with col_3:
		LoanAmount = st.slider('LoanAmount  in k$', 1, 500, 250)
		LoanTerm = st.slider('Loan_Amount_Term in months', 120, 480, 360,step = 120)	
	with col_4:
		Credit_history = st.selectbox( 'Previous Credit history', ('Yes','No' ))
		Property_Area = st.selectbox( 'Property Area', ('Urban', 'Rular','Semiurban'))

	"---"
	st.write("Your personal information is:- Income",Income,", Co-Applicants Income",CoIncome ,", Marital status-" ,Married,", Education-", 
		Education,", and Previous Credit history- ",Credit_history)	
	st.write("Your loan details:- Loan Amount",LoanAmount,"Loan Duration",LoanTerm,"and Property Area - ", Property_Area)	
	"---"

	if Married=="UnMarried":
		Married_flag = 0
	if Married=="Married":
		Married_flag = 1

	if Education=="Not Graduate":
		Education_flag = 1
	if Education=="Graduate":
		Education_flag = 0

	if Credit_history=="No":
		Credit_history_flag = 0
	if Credit_history=="Yes":
		Credit_history_flag = 1

	if Property_Area=="Urban":
		Property_Area_flag = 2
	if Property_Area=="Rular":
		Property_Area_flag = 0
	if Property_Area=="Semiurban":
		Property_Area_flag = 1

	a = np.array([Married_flag,Education_flag,Income,CoIncome,LoanAmount,LoanTerm,Credit_history_flag,Property_Area_flag])
	a = np.expand_dims(a, 0)
	
	"Hello"
# 	predict_XGB = XGB_model.predict(a)[0]
# 	predict_RF = RF_model.predict(a)[0]
# 	predict_LR =  LR_model.predict(a)[0]
	



    
	'## Prediction'
	col_10, col_20,col_30 = st.columns(3)

	with col_10:
		'##### XGBoost'
		if predict_XGB == 1:
			#predict_XGB = "Yes"
			st.success('Loan will get approve !!!')
		if predict_XGB == 0:
			#predict_XGB = "No"
			st.error('Loan will not get approve XXX')

	with col_20:
		'##### RandomForest'
		if predict_RF == 1:
			#predict_RF = "Yes"
			st.success('Loan will get approve !!!')
		if predict_RF == 0:
			#predict_RF = "No"
			st.error('Loan will not get approve XXX')

	with col_30:
		'##### Logistic Regression'
		if predict_LR == 1:
			#predict_LR = "Yes"
			st.success('Loan will get approve !!!')
		if predict_LR == 0:
			#predict_LR = "No"
			st.error('Loan will not get approve XXX')



with tab2:
	#"## Project Overview"

	'##### Binary Classification - Loan Approval Prediction '
	'Aim of this project is to predict if loan will get approved or not based on the binary classification models'

	"This is the third project in the ML for Data science series. The aim of this project is to"

	"- Practice EDA and Feature Engineering for classifier model"
	"- Learn and explore different classifier models and performance matrices."
	"- How to build robust models using k-fold cross validation"
	"- Learn how to deploy these machine learning models using pickle in webapplication"
	'- Practice web application developmennt and deployment.'

	st.info(' Main Aim is to understand how the data science pipeline works and get used to basic tools, and not to build accurate model')

	"---"
	'## Sources'
	"[Kaggle DataSet](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)"
	"[Github](https://github.com/Kapil3003/03_Classification)"

	# "---"
	# "- Checkout the app in WebApp tab"
	# "- Checkout the step by step analysis in Methodology tab"
	"---"

	" ## Projects"
	" 1. [Clustering - Customer Segmentation](https://kapil3003-01-clustering-clustering-streamlit-app-43fp3b.streamlitapp.com/)"
	" 2. [Forecasting - Banglore House Price Prediction](https://kapil3003-02-regression-regression-housepriceprediction-ifckzh.streamlitapp.com/)"
	" 3. [Binary Classification - Loan Approval Prediction ](https://kapil3003-03-classification-classification-streamlit-app-el6w2c.streamlitapp.com/)"
	" 4. [Hyper parameter Optimisation - Breast Cancer Prediction](https://github.com/Kapil3003/04_Hyperparameter_Optimization)"


with tab3:

	#"# Problem Statement"
	'##### Binary Classification - Loan Approval Prediction '
	'Aim of this project is to predict if loan will get approved or not based on the binary classification models'

	'we are going to work on binary classification problem, where we got some information about sample of peoples, and we need to predict whether we should give someone a loan or not depending on personal information. '
	'Our Dataset is very small(614 entries) , so we wont be focusing on building the accurate prediction model.'

	'main aim of the projects is to practive EDA, get understanding of classification models, there performance matrices and K Fold cross validation'
	
	"---"
	"#### DataSet"
	st.table(df.head())
	st.write("Shape of the data" , df.shape)

	'### Exploratory Data Analysis'

	col_10, col_20 = st.columns([2, 1])
	with col_10:
		st.pyplot(data_dist)

	with col_20:
		st.write('The percentage of Y class : %.2f' % (df['Loan_Status'].value_counts()[0] / len(df)))
		st.write('The percentage of N class : %.2f' % (df['Loan_Status'].value_counts()[1] / len(df)))
		st.write("we can say data is balanced")
	col_10, col_20,col_30 = st.columns(3)
	with col_10:
		st.pyplot(credit_dist)

	with col_20:
		st.pyplot(credit_dist_data)

	with col_30:
		st.write('The percentage with Credit History class : %.2f' % (df['Credit_History'].value_counts()[1] / len(df)))
		st.write('The percentage of Credit History in approved loan class : %.2f' % (df[df["Loan_Status"]=='Y']['Credit_History'].value_counts()[1] / len(df[df["Loan_Status"]=='Y'])))
		st.write('The percentage of Credit History in approved loan class : %.2f' % (df[df["Loan_Status"]=='N']['Credit_History'].value_counts()[1] / len(df[df["Loan_Status"]=='N'])))
		
		'Dataset is unbalanced in terms of credit history - 85% of the data has credit history. Even with this high number we see only 50% of the people with credit history getting their loans rejected' 
	

	col_10, col_20,col_30 = st.columns(3)
	with col_10:
		st.pyplot(credit_dist)

	with col_20:
		st.pyplot(credit_dist_data)

	with col_30:
		st.write('The percentage with Credit History class : %.2f' % (df['Credit_History'].value_counts()[1] / len(df)))
		st.write('The percentage of Credit History in approved loan class : %.2f' % (df[df["Loan_Status"]=='Y']['Credit_History'].value_counts()[1] / len(df[df["Loan_Status"]=='Y'])))
		st.write('The percentage of Credit History in approved loan class : %.2f' % (df[df["Loan_Status"]=='N']['Credit_History'].value_counts()[1] / len(df[df["Loan_Status"]=='N'])))
	

	col_10, col_20,col_30 = st.columns(3)
	with col_10:
		st.pyplot(gender_dist)

	with col_20:
		st.pyplot(gender_dist_data)

	with col_30:
		st.write('The percentage of male class : %.2f' % (df['Gender'].value_counts()[0] / len(df)))

		st.write('The percentage of male with approved loan class : %.2f' % (df[df["Loan_Status"]=='Y']['Gender'].value_counts()[0] / len(df[df["Loan_Status"]=='Y'])))
		st.write('The percentage of male with unapproved loan  class : %.2f' % (df[df["Loan_Status"]=='N']['Gender'].value_counts()[0] / len(df[df["Loan_Status"]=='N'])))

		'Distribution doesnt change much - not important'

	col_10, col_20,col_30 = st.columns(3)
	with col_10:
		st.pyplot(married_dist)

	with col_20:
		st.pyplot(married_dist_data)

	with col_30:
		st.write('The percentage of married class : %.2f' % (df['Married'].value_counts()[0] / len(df)))

		st.write('The percentage married people in approved loan class : %.2f' % (df[df["Loan_Status"]=='Y']['Married'].value_counts()[0] / len(df[df["Loan_Status"]=='Y'])))
		st.write('The percentage married people in unapproved loan  class : %.2f' % (df[df["Loan_Status"]=='N']['Married'].value_counts()[0] / len(df[df["Loan_Status"]=='N'])))

		'Didnt affect too much but we can still consider'

	col_10, col_20,col_30 = st.columns(3)
	with col_10:
		st.pyplot(dependents_dist)

	with col_20:
		st.pyplot(dependents_dist_data)

	with col_30:
		st.write('The percentage of 0 dependents : %.2f' % (df['Dependents'].value_counts()[0] / len(df)))
		# st.write('The percentage of 1 dependents : %.2f' % (df['Dependents'].value_counts()[1] / len(df)))
		# st.write('The percentage of 2 dependents : %.2f' % (df['Dependents'].value_counts()[2] / len(df)))
		# st.write('The percentage of 3 dependents : %.2f' % (df['Dependents'].value_counts()[3] / len(df)))

		st.write('The percentage 0 dependents in approved loan class : %.2f' % (df[df["Loan_Status"]=='Y']['Dependents'].value_counts()[0] / len(df[df["Loan_Status"]=='Y'])))
		st.write('The percentage 0 dependents in unapproved loan  class : %.2f' % (df[df["Loan_Status"]=='N']['Dependents'].value_counts()[0] / len(df[df["Loan_Status"]=='N'])))

		# st.write('The percentage 1 dependents in approved loan class : %.2f' % (df[df["Loan_Status"]=='Y']['Dependents'].value_counts()[1] / len(df[df["Loan_Status"]=='Y'])))
		# st.write('The percentage 1 dependents in unapproved loan  class : %.2f' % (df[df["Loan_Status"]=='N']['Dependents'].value_counts()[1] / len(df[df["Loan_Status"]=='N'])))

		# st.write('The percentage 2 dependents in approved loan class : %.2f' % (df[df["Loan_Status"]=='Y']['Dependents'].value_counts()[2] / len(df[df["Loan_Status"]=='Y'])))
		# st.write('The percentage 2 dependents in unapproved loan  class : %.2f' % (df[df["Loan_Status"]=='N']['Dependents'].value_counts()[2] / len(df[df["Loan_Status"]=='N'])))

		# st.write('The percentage 3+ dependents in approved loan class : %.2f' % (df[df["Loan_Status"]=='Y']['Dependents'].value_counts()[3] / len(df[df["Loan_Status"]=='Y'])))
		# st.write('The percentage 3+ dependents in unapproved loan  class : %.2f' % (df[df["Loan_Status"]=='N']['Dependents'].value_counts()[3] / len(df[df["Loan_Status"]=='N'])))

		'Distribution remains the same, doesnt affect much - not important'

	col_10, col_20,col_30 = st.columns(3)
	with col_10:
		st.pyplot(education_dist)

	with col_20:
		st.pyplot(education_dist_data)

	with col_30:
		st.write('The percentage of graduates in a dataset : %.2f' % (df['Education'].value_counts()[0] / len(df)))
		st.write('The percentage graduates in approved loan class : %.2f' % (df[df["Loan_Status"]=='Y']['Education'].value_counts()[0] / len(df[df["Loan_Status"]=='Y'])))
		st.write('The percentage graduates in unapproved loan  class : %.2f' % (df[df["Loan_Status"]=='N']['Education'].value_counts()[0] / len(df[df["Loan_Status"]=='N'])))

		'Graduate people have little higher chance of approval'

	col_10, col_20,col_30 = st.columns(3)
	with col_10:
		st.pyplot(employed_dist)

	with col_20:
		st.pyplot(employed_dist_data)

	with col_30:
		st.write('The percentage of Self_Employed in a dataset : %.2f' % (df['Self_Employed'].value_counts()[0] / len(df)))
		st.write('The percentage ofSelf_Employed in approved loan class : %.2f' % (df[df["Loan_Status"]=='Y']['Self_Employed'].value_counts()[0] / len(df[df["Loan_Status"]=='Y'])))
		st.write('The percentage of Self_Employed in unapproved loan  class : %.2f' % (df[df["Loan_Status"]=='N']['Self_Employed'].value_counts()[0] / len(df[df["Loan_Status"]=='N'])))
		'Distribution doesnt change much - not important'

	col_10, col_20,col_30 = st.columns(3)
	with col_10:
		st.pyplot(propert_dist)

	with col_20:
		st.pyplot(propert_dist_data)

	with col_30:
		st.write('The percentage of Semiurban : %.2f' % (df['Property_Area'].value_counts()[0] / len(df)),', Urban : %.2f' % (df['Property_Area'].value_counts()[1] / len(df)),'Rural : %.2f' % (df['Property_Area'].value_counts()[2] / len(df))) 
		# st.write('The percentage of Urban : %.2f' % (df['Property_Area'].value_counts()[1] / len(df)))
		# st.write('The percentage of Rural : %.2f' % (df['Property_Area'].value_counts()[2] / len(df)))

		st.write('Approved loan class, Semiurban  : %.2f' % (df[df["Loan_Status"]=='Y']['Property_Area'].value_counts()[0] / len(df[df["Loan_Status"]=='Y'])),', Urban: %.2f' % (df[df["Loan_Status"]=='Y']['Property_Area'].value_counts()[1] / len(df[df["Loan_Status"]=='Y'])),', Rural : %.2f' % (df[df["Loan_Status"]=='Y']['Property_Area'].value_counts()[2] / len(df[df["Loan_Status"]=='Y'])))
		# st.write('The percentage Semiurban in unapproved loan  class : %.2f' % (df[df["Loan_Status"]=='N']['Property_Area'].value_counts()[0] / len(df[df["Loan_Status"]=='N'])))

		# st.write('The percentage Urban in approved loan class : %.2f' % (df[df["Loan_Status"]=='Y']['Property_Area'].value_counts()[1] / len(df[df["Loan_Status"]=='Y'])))
		# st.write('The percentage Urban in unapproved loan  class : %.2f' % (df[df["Loan_Status"]=='N']['Property_Area'].value_counts()[1] / len(df[df["Loan_Status"]=='N'])))

		# st.write('The percentage Rural in approved loan class : %.2f' % (df[df["Loan_Status"]=='Y']['Property_Area'].value_counts()[2] / len(df[df["Loan_Status"]=='Y'])))
		# st.write('The percentage Rural in unapproved loan  class : %.2f' % (df[df["Loan_Status"]=='N']['Property_Area'].value_counts()[2] / len(df[df["Loan_Status"]=='N'])))
	
		'Didnt affect too much but we can still consider'

	# col_10, col_20 = st.columns([2, 1])
	# with col_10:
	# 	st.pyplot(income_dist)

	# with col_20:
	# 	st.write('The percentage of Y class : %.2f' % (df['Loan_Status'].value_counts()[0] / len(df)))
	# 	st.write('The percentage of N class : %.2f' % (df['Loan_Status'].value_counts()[1] / len(df)))	

	col_10, col_20 = st.columns([2, 1])
	with col_10:
		'##### MedianValues'
		st.table(df.groupby('Loan_Status').median())

		x = df.copy()
		x["LoanperIncome"] = x.LoanAmount/x.ApplicantIncome

		x.groupby('Loan_Status').mean()
		'##### Mean Values' 

		st.table(x.groupby('Loan_Status').mean())

	with col_20:

		'Co-Applicants median income suggestes that people with approved loans(Y) have very high Co-Applicants Income'
		''
		''
		''
		''
		''
		''
		''
		'We created a new feature to understand the loan taking capacity of an individual. It is basically the ratio between Loan Amount and monthly income.'
		'Higher the ratio less is the loan repayment capacity'

	# col_10, col_20 = st.columns([3, 1])
	# with col_10:
	# 	st.pyplot(pairplot)
	# with col_20:
	# 	" Correlation plot does not reveal any correlation between any features"s

	"### Data Preparation"

	'Based on EDA, We will be ignoring the following features 1. Gender 2. Dependents 3.Self Employment'

	' We will also convert categorial features into integers using LabelEncoder'

	code = '''from sklearn.preprocessing import LabelEncoder		
le = LabelEncoder()		
data.Married = le.fit_transform(data.Married)
data.Education = le.fit_transform(data.Education)
data.Credit_History = le.fit_transform(data.Credit_History)
data.Property_Area = le.fit_transform(data.Property_Area)
data.Loan_Status = le.fit_transform(data.Loan_Status)'''

	st.code(code, language='python')


	'##### Final Data Before Test_Train_Split'
	st.table(final_Data.head())

	'#### Model Training'
	'Used StartifiedShuffle to split the data in same ratio as original data w.r.t target Distribution'
	st.code('from sklearn.model_selection import StratifiedShuffleSplit', language='python')

	' Using k-fold cross validation technique to varify the robustness of model. and compared the models for their accuracy'

	code = '''from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
from sklearn.model_selection import cross_val_score

Accuracy_XGB = (cross_val_score(XGBClassifier(), X, y, cv=cv)).mean()
Accuracy_RF = (cross_val_score(RandomForestClassifier(), X, y, cv=cv)).mean()
Accuracy_DT = (cross_val_score(DecisionTreeClassifier(), X, y, cv=cv)).mean()
Accuracy_KNN = (cross_val_score(KNeighborsClassifier(), X, y, cv=cv)).mean()
Accuracy_LR = (cross_val_score(LogisticRegression(), X, y, cv=cv)).mean()
'''

	
	st.code(code, language='python')
	'We were able to acheive more than **75%** accuracy for XGB,RF and LR after k-fold'
	'### Performances matrices'
	'1. Confusion Matrix'
	'2. Accuracy score - Precision, Recall, F-1 and support'
