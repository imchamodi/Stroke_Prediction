#!/usr/bin/env python
# coding: utf-8

# # Brain Stroke Dataset

# <u><b>Loading Initial Libraries

# ### <u>Libraries

# In[1]:


# Data manipulation libraries
import pandas as pd 
import numpy as np 

# Visualization libraries
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set_style('darkgrid')
import plotly.express as ex
import plotly.graph_objs as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import os as swd
import matplotlib.ticker as mtick
from sklearn.cross_decomposition import PLSRegression


# <b><u>Importing and observing the dataset

# In[2]:


data = pd.read_csv('E:/3rd year 2nd semester/ST3082-statistical learning/3rd Project/healthcare-dataset-stroke-data.csv')
data.head(5)


# In[3]:


data.drop(['id'],axis=1,inplace=True)
data.head(5)


# In[4]:


data.describe(include='all')


# In[5]:


plt.figure(figsize = (5,5))
sns.heatmap(data.isnull(),cmap='summer')
plt.show()


# In[6]:


pwd


# In[7]:


dir_path = swd.path.dirname('E:\\3rd year 2nd semester\\ST3082-statistical learning\\3rd Project')
print(dir_path)


# In[8]:


#data = data[data['gender'] != 'Other']

# create a list of categorical column names to one-hot encode
#cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
cat_cols = ['gender']
print(data[cat_cols].value_counts(normalize=False))


# <b><u>Imputing missing values in 'bmi' using a regression tree</b>

# In[9]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,LabelEncoder

DT_bmi_pipe = Pipeline( steps=[ 
                               ('scale',StandardScaler()),
                               ('lr',DecisionTreeRegressor(random_state=42))
                              ])
X = data[['age','gender','ever_married','Residence_type','bmi']].copy()
X.gender = X.gender.replace({'Male':0,'Female':1,'Other':-1}).astype(np.uint8)
X.Residence_type = X.Residence_type.replace({'Urban':0,'Rural':1}).astype(np.uint8)
X.ever_married = X.ever_married.replace({'No':0,'Yes':1}).astype(np.uint8)
Missing = X[X.bmi.isna()]
X = X[~X.bmi.isna()]
Y = X.pop('bmi')
DT_bmi_pipe.fit(X,Y)
predicted_bmi = pd.Series(DT_bmi_pipe.predict(Missing[['age','gender','ever_married','Residence_type']]),index=Missing.index)
data.loc[Missing.index,'bmi'] = predicted_bmi


# In[10]:


plt.figure(figsize = (3,3))
sns.heatmap(data.isnull(),cmap='coolwarm')
plt.show()


# # <u>Exploratory Data Analysis 

# ## Univariate

# #### <u> Response Variable : Percentage of People Having Strokes

# In[11]:


x = pd.DataFrame( data.groupby(['stroke'])['stroke'].count())
print(x)
# plot
fig, ax = plt.subplots(figsize = (6,6), dpi = 70)
ax.barh([1], x.stroke[1], height = 0.7, color = '#df1529')
plt.text(-1350,-0.08, 'No Stroke',{'font': 'Serif','weight':'bold','size': 16,'style':'normal', 'color':"#009786"})
plt.text(5000,-0.08, '95%',{'font':'Serif','weight':'bold' ,'size':16,'color':"#009786"})
ax.barh([0], x.stroke[0], height = 0.7, color = "#009786")
plt.text(-1000,1, 'Stroke', {'font': 'Serif','weight':'bold','size': 16,'style':'normal', 'color':'#df1529'})
plt.text(300,1, '5%',{'font':'Serif', 'weight':'bold','size':'16','color':'#df1529'})


plt.text(-1150,1.77, 'Percentage of People Having Strokes' ,{'font': 'Serif', 'size': '25','weight':'bold', 'color':'black'})
plt.text(4650,1.65, 'Stroke ', {'font': 'Serif','weight':'bold','size': '16','weight':'bold','style':'normal', 'color':'#df1529'})
plt.text(5650,1.65, '|', {'color':'black' , 'size':'16', 'weight': 'bold'})
plt.text(5750,1.65, 'No Stroke', {'font': 'Serif','weight':'bold', 'size': '16','style':'normal', 'weight':'bold','color':"#009786"})
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


# #### <u>Categorical variables distribtion

# In[12]:


# Select only the continuous variables

# Select only the continuous variables
continuous_vars = ['age', 'avg_glucose_level', 'bmi']
data_cont = data.loc[:, continuous_vars]

# Create the correlation plot
plt.figure(figsize=(12, 8))
ax = sns.heatmap(data_cont.corr(), annot=True, cmap=sns.diverging_palette(10, 220, sep=80, n=5), vmin=-1, vmax=1)
plt.show()


# #### <u>Numeric/continuous variable distribtion

# ###### 1)Age distribution

# In[13]:


fig = plt.figure(figsize = (24,10), dpi = 60)

gs = fig.add_gridspec(10,24)
gs.update(wspace = 1, hspace = 0.05)


ax2 = fig.add_subplot(gs[1:4,0:8]) #distribution plot

sns.kdeplot(data = data, x = 'age', ax = ax2, shade = True, color = "#009786", alpha = 1,ec='black')
ax2.set_xlabel('Age of a person', fontdict = {'font':'Serif', 'color': 'black', 'size': 16,'weight':'bold' })
ax2.text(-17,0.025,'Overall Age Distribution ', {'font':'Serif', 'color': 'black','weight':'bold','size':24}, alpha = 0.9)


# ###### 2)Avg Gluicose Level

# In[14]:


fig = plt.figure(figsize = (24,10), dpi = 60)

gs = fig.add_gridspec(10,24)
gs.update(wspace = 1, hspace = 0.05)


ax2 = fig.add_subplot(gs[1:4,0:8]) #distribution plot

sns.kdeplot(data = data, x = 'avg_glucose_level', ax = ax2, shade = True, color = "#009786", alpha = 1, ec='black' )
ax2.set_xlabel('Avg. Glucose Levels', fontdict = {'font':'Serif', 'color': 'black', 'size': 16,'weight':'bold' })
ax2.text(-17,0.025,'Overall Avg. Glucose Levels Distribution ', {'font':'Serif', 'color': 'black','weight':'bold','size':24}, alpha = 0.9)


# ###### 2)BMI

# In[15]:


fig = plt.figure(figsize=(24, 10), dpi=60)
gs = fig.add_gridspec(10, 24)
gs.update(wspace=1, hspace=0.05)

ax2 = fig.add_subplot(gs[1:4, 0:8]) #distribution plot
sns.kdeplot(data=data, x='bmi', ax=ax2, shade=True, color="#009786", alpha=1, ec='black')
ax2.set_xlabel('BMI', fontdict={'font': 'Serif', 'color': 'black', 'size': 16, 'weight': 'bold'})

fig.text(0.1, 0.9, 'Overall BMI Distribution', fontdict={'font': 'Serif', 'color': 'black', 'weight': 'bold', 'size': 24}, va='center', ha='left')

plt.show()


# In[41]:


from scipy.stats import spearmanr

# Extract the age and hypertension columns from the DataFrame
age = data['age']
hypertension = data['hypertension']
smoking_status =data['smoking_status']
heart_disease=data['heart_disease']
gender=data['gender']
work_type=data['work_type']

# Perform the Spearman's rank correlation test
correlation_hy, p_value = spearmanr(age, hypertension)

# Print the correlation coefficient and p-value
print("Spearman's rank correlation coefficient hy:", correlation_hy)
print("p-value:", p_value)


# Perform the Spearman's rank correlation test
correlation_smoke, p_value = spearmanr(age, smoking_status)

# Print the correlation coefficient and p-value
print("Spearman's rank correlation coefficient smoke:", correlation_smoke)
print("p-value:", p_value)

# Perform the Spearman's rank correlation test
correlation_heart, p_value = spearmanr(age, heart_disease)

# Print the correlation coefficient and p-value
print("Spearman's rank correlation coefficient heart:", correlation_heart)
print("p-value:", p_value)

correlation_gender, p_value_gender = spearmanr(age, gender)

# Perform the Spearman's rank correlation test
print("Spearman's rank correlation coefficient gender:", correlation_gender)
print("p-value for gender:", p_value_gender)


correlation_work, p_value_work = spearmanr(age, work_type)


print("Spearman's rank correlation coefficient work type:", correlation_work)
print("p-value for work type:", p_value_work)




# In[44]:


from scipy.stats import kruskal

# Extract the age and smoking_status columns from the DataFrame
age = data['age']
smoking_status = data['smoking_status']

# Perform the Kruskal-Wallis test
statistic, p_value = kruskal(*[age[data['smoking_status']==val] for val in data['smoking_status'].unique()])

# Print the results
print("Kruskal-Wallis test statistic age vs smoking_status:", statistic)
print("p-value:", p_value)


heart_disease = data['heart_disease']

# Perform the Kruskal-Wallis test
statistic, p_value = kruskal(age, heart_disease)

# Print the results
print("Kruskal-Wallis test statistic age vs heart_disease:", statistic)
print("p-value:", p_value)


# In[16]:


import pandas as pd
from scipy.stats import chi2_contingency

# Create a contingency table of smoking status and work type
cont_table = pd.crosstab(data['smoking_status'], data['work_type'])

# Print the contingency table
print(cont_table)

# Perform the chi-squared test
chi2, p_value, dof, expected = chi2_contingency(cont_table)

# Print the results
print('Chi-squared statistic:', chi2)
print('P-value:', p_value)
print('Degrees of freedom:', dof)
print('Expected frequencies:', expected)


# In[ ]:





#   ## Bivariabte

# #### <u>Numeric/continuous variableS

# ###### 1)Age - Stroke

# In[52]:


fig = plt.figure(figsize = (24,10), dpi = 60)

gs = fig.add_gridspec(10,24)
gs.update(wspace = 1, hspace = 0.05)

ax3 = fig.add_subplot(gs[6:9, 0:8]) #hue distribution plot
sns.kdeplot(data = data[data['stroke'] == 0], x = 'age',ax = ax3, shade = True,  alpha = 1, color = "#009786", ec='black' )
sns.kdeplot(data = data[data['stroke'] == 1], x = 'age',ax = ax3, shade = True,  alpha = 0.8, color = "#df1529", ec='black')



ax3.set_xlabel('Age of a person', fontdict = {'font':'Serif', 'color': 'black', 'weight':'bold','size': 16})

ax3.text(-17,0.0525,'Age-Stroke Distribution', {'font':'Serif', 'weight':'bold','color': 'black', 'size':24}, alpha= 0.9)
ax3.text(100,0.043, 'Stroke ', {'font': 'Serif','weight':'bold','size': '16','weight':'bold','style':'normal', 'color':"#df1529"})
ax3.text(117,0.043, '|', {'color':'black' , 'size':'16', 'weight': 'bold'})
ax3.text(120,0.043, 'No Stroke', {'font': 'Serif','weight':'bold', 'size': '16','style':'normal', 'weight':'bold','color':"#009786"})
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(4, 4))

sns.boxplot(x="stroke", y="age", data=data, palette={0:"#009786", 1:"#df1529"})

plt.xlabel("Stroke", fontdict={"fontsize": 14, "fontweight": "bold"})
plt.ylabel("Age", fontdict={"fontsize": 14, "fontweight": "bold"})

plt.show()



# ###### 2)Avg Glucose Level - Stroke

# In[64]:


fig = plt.figure(figsize = (24,10), dpi = 60)

gs = fig.add_gridspec(10,24)
gs.update(wspace = 1, hspace = 0.05)

ax3 = fig.add_subplot(gs[5:10, 0:10]) #hue distribution plot
sns.kdeplot(data = data[data['stroke'] == 0], x = "avg_glucose_level",ax = ax3, shade = True,  alpha = 1, color = "#009786", ec='black' )
sns.kdeplot(data = data[data['stroke'] == 1], x = "avg_glucose_level",ax = ax3, shade = True,  alpha = 0.8, color = "#df1529", ec='black')

ax3.set_xlabel('Avg. Glucose Level', fontdict = {'font':'Serif', 'color': 'black', 'weight':'bold','size': 16})

ax3.text(-17,0.0195,'Avg. Glucose Level - Stroke Distribution', {'font':'Serif', 'weight':'bold','color': 'black', 'size':24}, alpha= 0.9)
ax3.text(240,0.0174,  'Stroke ', {'font': 'Serif','weight':'bold','size': '16','weight':'bold','style':'normal', 'color':"#df1529"})
ax3.text(290,0.0174, '|', {'color':'black' , 'size':'16', 'weight': 'bold'})
ax3.text(300,0.0174,'No Stroke', {'font': 'Serif','weight':'bold', 'size': '16','style':'normal', 'weight':'bold','color':"#009786"})
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(4, 4))

sns.boxplot(x="stroke", y="avg_glucose_level", data=data, palette={0:"#009786", 1:"#df1529"})

plt.xlabel("Stroke", fontdict={"fontsize": 14, "fontweight": "bold"})
plt.ylabel("Avg. Glucose Level", fontdict={"fontsize": 14, "fontweight": "bold"})

plt.show()


# ###### 2)BMI - Stroke

# In[63]:


fig = plt.figure(figsize = (24,10), dpi = 60)

gs = fig.add_gridspec(10,24)
gs.update(wspace = 1, hspace = 0.05)

ax3 = fig.add_subplot(gs[6:9, 0:8]) #hue distribution plot
sns.kdeplot(data = data[data['stroke'] == 0], x = "bmi",ax = ax3, shade = True,  alpha = 1, color = "#009786", ec='black' )
sns.kdeplot(data = data[data['stroke'] == 1], x = "bmi",ax = ax3, shade = True,  alpha = 0.8, color = "#df1529", ec='black')

ax3.set_xlabel('BMI', fontdict = {'font':'Serif', 'color': 'black', 'weight':'bold','size': 16})

ax3.text(-15,0.12,'BMI - Stroke Distribution', {'font':'Serif', 'weight':'bold','color': 'black', 'size':24}, alpha= 0.9)
ax3.text(80,0.095,  'Stroke ', {'font': 'Serif','weight':'bold','size': '16','weight':'bold','style':'normal', 'color':'#df1529'})
ax3.text(95,0.095, '|', {'color':'black' , 'size':'16', 'weight': 'bold'})
ax3.text(97,0.095,'No Strokey', {'font': 'Serif','weight':'bold', 'size': '16','style':'normal', 'weight':'bold','color':"#009786"})
plt.show()



import seaborn as sns
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(4, 4))

sns.boxplot(x="stroke", y="bmi", data=data, palette={0:"#009786", 1:"#df1529"})

plt.xlabel("Stroke", fontdict={"fontsize": 14, "fontweight": "bold"})
plt.ylabel("BMI", fontdict={"fontsize": 14, "fontweight": "bold"})

plt.show()


# #### <u>Categorical variables

# ###### 1)hypertention- Stroke (do not use this)

# In[20]:


str_only = data[data['stroke'] == 1]
no_str_only = data[data['stroke'] == 0]

fig = plt.figure(figsize = (30,10), dpi = 60)

gs = fig.add_gridspec(10,24)
gs.update(wspace = 1, hspace = 0.05)

ax8 = fig.add_subplot(gs[6:9, 0:8])

positive = pd.DataFrame(str_only["hypertension"].value_counts())
positive["Percentage"] = positive["hypertension"].apply(lambda x: x/sum(positive["hypertension"])*100)
negative = pd.DataFrame(no_str_only["hypertension"].value_counts())
negative["Percentage"] = negative["hypertension"].apply(lambda x: x/sum(negative["hypertension"])*100)

x = np.arange(len(positive))
ax8.text(-0.45, 120,'Hypertension - Stroke Distribution', {'font':'Serif', 'weight':'bold','color': 'black', 'size':24}, alpha= 0.9)
ax8.grid(color='gray', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
ax8.bar(x, height=positive["Percentage"], zorder=3, color= '#ff00bf', width=0.4)
ax8.bar(x+0.4, height=negative["Percentage"], zorder=3, color="#0080ff", width=0.4)
ax8.set_xticks(x + 0.4 / 2)
ax8.set_xticklabels(['No Hypertension','Hypertension'],fontdict = {'font':'Serif', 'color': 'black', 'weight':'bold','size': 16})
ax8.yaxis.set_major_formatter(mtick.PercentFormatter())
ax8.yaxis.set_major_locator(mtick.MultipleLocator(20))
for i,j in zip([0, 1], positive["Percentage"]):
    ax8.annotate(f'{j:0.0f}%',xy=(i, j/2), color='black', size=16 ,font='Serif',horizontalalignment='center', verticalalignment='center')
for i,j in zip([0, 1], negative["Percentage"]):
    ax8.annotate(f'{j:0.0f}%',xy=(i+0.4, j/2), color='black',size=16 ,font='Serif', horizontalalignment='center', verticalalignment='center')

ax8.text(1.6, 100, 'Stroke ', {'font': 'Serif','weight':'bold','size': '16','style':'normal', 'color':'#ff00bf'})
ax8.text(1.84, 100, '|', {'color':'black' , 'size':'16', 'weight': 'bold'})
ax8.text(1.88, 100,'No Stroke', {'font': 'Serif','weight':'bold', 'size': '16','style':'normal', 'color':"#0080ff"})
   
    
plt.show()

    


# In[21]:


#27% of people who have stroke also have Hypertension
#while only 9% of people who do not have stroke have hypertention


# ###### 1)Hypertention- Stroke

# In[22]:


have_hyper = data[data['hypertension'] == 1]
donot_have_hyper = data[data['hypertension'] == 0]

fig = plt.figure(figsize = (30,10), dpi = 60)

gs = fig.add_gridspec(10,24)
gs.update(wspace = 1, hspace = 0.05)

ax8 = fig.add_subplot(gs[6:9, 0:8])

positive = pd.DataFrame(have_hyper["stroke"].value_counts())
positive["Percentage"] = positive["stroke"].apply(lambda x: x/sum(positive["stroke"])*100)
negative = pd.DataFrame(donot_have_hyper["stroke"].value_counts())
negative["Percentage"] = negative["stroke"].apply(lambda x: x/sum(negative["stroke"])*100)

x = np.arange(len(positive))
ax8.text(-0.45, 120,'Hypertension - Stroke Distribution', {'font':'Serif', 'weight':'bold','color': 'black', 'size':24}, alpha= 0.9)
ax8.grid(color='gray', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
ax8.bar(x, height=positive["Percentage"], zorder=3, color= '#ff00bf', width=0.4)
ax8.bar(x+0.4, height=negative["Percentage"], zorder=3, color="#0080ff", width=0.4)
ax8.set_xticks(x + 0.4 / 2)
ax8.set_xticklabels(['No Stroke','Stroke'],fontdict = {'font':'Serif', 'color': 'black', 'weight':'bold','size': 16})
ax8.yaxis.set_major_formatter(mtick.PercentFormatter())
ax8.yaxis.set_major_locator(mtick.MultipleLocator(20))
for i,j in zip([0, 1], positive["Percentage"]):
    ax8.annotate(f'{j:0.0f}%',xy=(i, j/2),color='black', horizontalalignment='center', verticalalignment='center')
for i,j in zip([0, 1], negative["Percentage"]):
    ax8.annotate(f'{j:0.0f}%',xy=(i+0.4, j/2), color='black', horizontalalignment='center', verticalalignment='center')

ax8.text(1.88, 100, 'Hypertension ', {'font': 'Serif','weight':'bold','size': '16','style':'normal', 'color':'#ff00bf'})
ax8.text(1.88, 110,'No Hypertension', {'font': 'Serif','weight':'bold', 'size': '16','style':'normal', 'color':"#0080ff"})
   
    
plt.show()


########################################### 2nd graph
stroke_hyper = data[data['stroke'] == 1]['hypertension'].value_counts()
healthy_hyper = data[data['stroke'] == 0]['hypertension'].value_counts()

no = data['hypertension'].value_counts().values[0]
yes =  data['hypertension'].value_counts().values[1]

stroke_no = stroke_hyper.values[0] / no * 100
stroke_yes = stroke_hyper.values[1] / yes * 100
healthy_no = healthy_hyper.values[0] / no * 100
healthy_yes = healthy_hyper.values[1] / yes * 100

no_per = no / (no+yes) * 100
yes_per = yes / (no+yes) * 100

groups = ['No Hypertension', 'Hypertension']
values1 = [healthy_no , healthy_yes]
values2 = [stroke_no, stroke_yes]

fig = plt.figure(figsize = (30,20), dpi = 60)

gs = fig.add_gridspec(10,24)
gs.update(wspace = 1, hspace = 0.05)

ax = fig.add_subplot(gs[6:9, 0:8])
ax.text(-0.45, 120,'Hypertension - Stroke Distribution', {'font':'Serif', 'weight':'bold','color': 'black', 'size':24}, alpha= 0.9)

# Stacked bar chart
ax.bar(groups, values1, color='#009786')
ax.bar(groups, values2, bottom = values1, color='#df1529')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.yaxis.set_major_locator(mtick.MultipleLocator(20))

# Labels
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2 + bar.get_y(),
            str(round(bar.get_height(), 1)) + '%',  # Add percentage symbol to the label
            ha='center', color='black',  size=16 ,font='Serif')

ax.text(1.88, 100, 'Stroke ', {'font': 'Serif','weight':'bold','size': '16','style':'normal', 'color':'#df1529'})
ax.text(1.88, 110,'No Stroke', {'font': 'Serif','weight':'bold', 'size': '16','style':'normal', 'color':"#009786"})
ax.set_xlabel('Hypertenion', fontdict = {'font':'Serif', 'color': 'black', 'weight':'bold','size': 16})
ax.set_ylabel('Precentage', fontdict = {'font':'Serif', 'color': 'black', 'weight':'bold','size': 16})


# plt.show() 


# In[23]:


#if you have hypertention 13% to stoke 
#No hypertention only 4% stroke 


# In[24]:


########################################### 2nd graph
stroke_hyper = data[data['stroke'] == 1]['gender'].value_counts()
healthy_hyper = data[data['stroke'] == 0]['gender'].value_counts()

no = data['gender'].value_counts().values[0]
yes =  data['gender'].value_counts().values[1]

stroke_no = stroke_hyper.values[0] / no * 100
stroke_yes = stroke_hyper.values[1] / yes * 100
healthy_no = healthy_hyper.values[0] / no * 100
healthy_yes = healthy_hyper.values[1] / yes * 100

no_per = no / (no+yes) * 100
yes_per = yes / (no+yes) * 100

groups = ['No Hypertension', 'Hypertension']
values1 = [healthy_no , healthy_yes]
values2 = [stroke_no, stroke_yes]

fig = plt.figure(figsize = (30,20), dpi = 60)

gs = fig.add_gridspec(10,24)
gs.update(wspace = 1, hspace = 0.05)

ax = fig.add_subplot(gs[6:9, 0:8])
ax.text(-0.45, 120,'Hypertension - Stroke Distribution', {'font':'Serif', 'weight':'bold','color': 'black', 'size':24}, alpha= 0.9)

# Stacked bar chart
ax.bar(groups, values1, color='#009786')
ax.bar(groups, values2, bottom = values1, color='#df1529')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.yaxis.set_major_locator(mtick.MultipleLocator(20))

# Labels
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2 + bar.get_y(),
            str(round(bar.get_height(), 1)) + '%',  # Add percentage symbol to the label
            ha='center', color='black',  size=16 ,font='Serif')

ax.text(1.88, 100, 'Stroke ', {'font': 'Serif','weight':'bold','size': '16','style':'normal', 'color':'#df1529'})
ax.text(1.88, 110,'No Stroke', {'font': 'Serif','weight':'bold', 'size': '16','style':'normal', 'color':"#009786"})
ax.set_xlabel('Hypertenion', fontdict = {'font':'Serif', 'color': 'black', 'weight':'bold','size': 16})
ax.set_ylabel('Precentage', fontdict = {'font':'Serif', 'color': 'black', 'weight':'bold','size': 16})


# plt.show() 


# ###### 2)Heart Disease- Stroke

# In[25]:


Unhelthy_heart = data[data['heart_disease'] == 1]
Healthy_heart = data[data['heart_disease'] == 0]

fig = plt.figure(figsize = (30,10), dpi = 60)

gs = fig.add_gridspec(10,24)
gs.update(wspace = 1, hspace = 0.05)

ax8 = fig.add_subplot(gs[6:9, 0:8])

positive = pd.DataFrame(Unhelthy_heart["stroke"].value_counts())
positive["Percentage"] = positive["stroke"].apply(lambda x: x/sum(positive["stroke"])*100)
negative = pd.DataFrame(Healthy_heart ["stroke"].value_counts())
negative["Percentage"] = negative["stroke"].apply(lambda x: x/sum(negative["stroke"])*100)

x = np.arange(len(positive))
ax8.text(-0.45, 120,'Heart Disease - Stroke Distribution', {'font':'Serif', 'weight':'bold','color': 'black', 'size':24}, alpha= 0.9)
ax8.grid(color='gray', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
ax8.bar(x, height=positive["Percentage"], zorder=3, color= '#ff00bf', width=0.4)
ax8.bar(x+0.4, height=negative["Percentage"], zorder=3, color="#0080ff", width=0.4)
ax8.set_xticks(x + 0.4 / 2)
ax8.set_xticklabels(['No Stroke','Stroke'],fontdict = {'font':'Serif', 'color': 'black', 'weight':'bold','size': 16})
ax8.yaxis.set_major_formatter(mtick.PercentFormatter())
ax8.yaxis.set_major_locator(mtick.MultipleLocator(20))
for i,j in zip([0, 1], positive["Percentage"]):
    ax8.annotate(f'{j:0.0f}%',xy=(i, j/2),color='black', horizontalalignment='center', verticalalignment='center')
for i,j in zip([0, 1], negative["Percentage"]):
    ax8.annotate(f'{j:0.0f}%',xy=(i+0.4, j/2), color='black', horizontalalignment='center', verticalalignment='center')

ax8.text(1.88, 100, 'Unhealty Heart ', {'font': 'Serif','weight':'bold','size': '16','style':'normal', 'color':'#ff00bf'})
ax8.text(1.88, 110,'Healthy Heart', {'font': 'Serif','weight':'bold', 'size': '16','style':'normal', 'color':"#0080ff"})
   
    
plt.show()

#################################### 2ND PLOT
stroke_heart = data[data['stroke'] == 1]['heart_disease'].value_counts()
healthy_heart = data[data['stroke'] == 0]['heart_disease'].value_counts()

no = data['heart_disease'].value_counts().values[0]
yes =  data['heart_disease'].value_counts().values[1]

stroke_no = stroke_heart.values[0] / no * 100
stroke_yes = stroke_heart.values[1] / yes * 100
healthy_no = healthy_heart.values[0] / no * 100
healthy_yes = healthy_heart.values[1] / yes * 100

no_per = no / (no+yes) * 100
yes_per = yes / (no+yes) * 100

groups = ['No Heart Disease', ' Heart Disease']
values1 = [healthy_no , healthy_yes]
values2 = [stroke_no, stroke_yes]

fig = plt.figure(figsize = (30,20), dpi = 60)

gs = fig.add_gridspec(10,24)
gs.update(wspace = 1, hspace = 0.05)

ax = fig.add_subplot(gs[6:9, 0:8])
ax.text(-0.45, 120,'Heart Disease - Stroke Distribution', {'font':'Serif', 'weight':'bold','color': 'black', 'size':24}, alpha= 0.9)

# Stacked bar chart
ax.bar(groups, values1, color='#009786')
ax.bar(groups, values2, bottom = values1, color='#df1529')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.yaxis.set_major_locator(mtick.MultipleLocator(20))

# Labels
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2 + bar.get_y(),
            str(round(bar.get_height(), 1)) + '%',  # Add percentage symbol to the label
            ha='center', color='black',size=16 ,font='Serif')

ax.text(1.88, 100, 'Stroke ', {'font': 'Serif','weight':'bold','size': '16','style':'normal', 'color':'#df1529'})
ax.text(1.88, 110,'No Stroke', {'font': 'Serif','weight':'bold', 'size': '16','style':'normal', 'color':"#009786"})
ax.set_xlabel('Heart Disease', fontdict = {'font':'Serif', 'color': 'black', 'weight':'bold','size': 16})
ax.set_ylabel('Precentage', fontdict = {'font':'Serif', 'color': 'black', 'weight':'bold','size': 16})


plt.show() 


# In[26]:


#Unhealthy heart 17% have stroke
#healthy heart only 4% stroke


# ###### 3)Smoking status- Stroke

# In[49]:


smoke = data['smoking_status'].value_counts()
stroke_smoke = data[data['stroke'] == 1]['smoking_status'].value_counts()
healthy_smoke = data[data['stroke'] == 0]['smoking_status'].value_counts()

never = smoke.values[0]
unknown =  smoke.values[1]
former = smoke.values[2]
smokes = smoke.values[3]

stroke_never = stroke_smoke.values[0] / never * 100
stroke_unknown = stroke_smoke.values[2]  / unknown *100
stroke_former = stroke_smoke.values[1]  / former * 100
stroke_smokes = stroke_smoke.values[3]  / smokes *100

healthy_never = healthy_smoke.values[0] / never * 100
healthy_unknown = healthy_smoke.values[1] / unknown *100
healthy_former = healthy_smoke.values[2] / former * 100
healthy_smokes = healthy_smoke.values[3]/ smokes *100


never_per = never/(never+unknown+former+smokes) * 100
unknown_per = unknown/(never+unknown+former+smokes)* 100
former_per = former/(never+unknown+former+smokes) * 100
smokes_per = smokes/(never+unknown+former+smokes)* 100


groups = ['Never Smoked', 'Formerly Smoked','Smokes','Unknown']
values1 = [healthy_never ,healthy_former,healthy_smokes,healthy_unknown]
values2 = [stroke_never ,stroke_former,stroke_smokes,stroke_unknown]

fig = plt.figure(figsize = (30,20), dpi = 60)

gs = fig.add_gridspec(10,24)
gs.update(wspace = 1, hspace = 0.05)

ax = fig.add_subplot(gs[6:9, 0:8])
ax.text(-0.45, 120,'Smoking status- Stroke Distribution', {'font':'Serif', 'weight':'bold','color': 'black', 'size':24}, alpha= 0.9)

# Stacked bar chart
ax.bar(groups, values1, color='#009786')
ax.bar(groups, values2, bottom = values1, color='#df1529')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.yaxis.set_major_locator(mtick.MultipleLocator(20))

# Labels
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2 + bar.get_y(),
            str(round(bar.get_height(), 1)) + '%',  # Add percentage symbol to the label
            ha='center', color='black',size=16 ,font='Serif')

ax.text(5.88, 100, 'Stroke ', {'font': 'Serif','weight':'bold','size': '16','style':'normal', 'color':'#df1529'})
ax.text(5.88, 110,'No Stroke', {'font': 'Serif','weight':'bold', 'size': '16','style':'normal', 'color':"#009786"})
ax.set_xlabel('Smoking Status', fontdict = {'font':'Serif', 'color': 'black', 'weight':'bold','size': 16})
ax.set_ylabel('Precentage', fontdict = {'font':'Serif', 'color': 'black', 'weight':'bold','size': 16})


# In[62]:


smoke = data['smoking_status'].value_counts()
stroke_smoke = data[data['stroke'] == 1]['smoking_status'].value_counts()
healthy_smoke = data[data['stroke'] == 0]['smoking_status'].value_counts()

never = smoke.values[0]
unknown = smoke.values[1]
former = smoke.values[2]
smokes = smoke.values[3]

eversmoke = former + smokes

stroke_eversmoke = (stroke_smoke.values[1]+stroke_smoke.values[3]) / eversmoke * 100
stroke_nosmoke = stroke_smoke.values[0]/ never * 100

healthy_eversmoke = (healthy_smoke.values[2]+ healthy_smoke.values[3]) / eversmoke * 100
healthy_nosmoke = healthy_smoke.values[0] / never * 100

groups = ['Smoke', 'Not Smoke']
values1 = [healthy_eversmoke ,healthy_nosmoke]
values2 = [stroke_eversmoke,stroke_nosmoke]

fig = plt.figure(figsize = (30,20), dpi = 60)

gs = fig.add_gridspec(10,24)
gs.update(wspace = 1, hspace = 0.05)

ax = fig.add_subplot(gs[6:9, 0:8])
ax.text(-0.45, 120,'Smoking Status - Stroke Distribution', {'font':'Serif', 'weight':'bold','color': 'black', 'size':24}, alpha= 0.9)

# Stacked bar chart
ax.bar(groups, values1, color='#009786')
ax.bar(groups, values2, bottom = values1, color='#df1529')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.yaxis.set_major_locator(mtick.MultipleLocator(20))

# Labels
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2 + bar.get_y(),
            str(round(bar.get_height(), 1)) + '%',  # Add percentage symbol to the label
            ha='center', color='black',size=16 ,font='Serif')

ax.text(5.88, 100, 'Stroke ', {'font': 'Serif','weight':'bold','size': '16','style':'normal', 'color':'#df1529'})
ax.text(5.88, 110,'No Stroke', {'font': 'Serif','weight':'bold', 'size': '16','style':'normal', 'color':"#009786"})
ax.set_xlabel('Work Type', fontdict = {'font':'Serif', 'color': 'black', 'weight':'bold','size': 16})
ax.set_ylabel('Precentage', fontdict = {'font':'Serif', 'color': 'black', 'weight':'bold','size': 16})


# ###### 3)Work Type- Stroke

# In[39]:


work = data['work_type'].value_counts()
print(work)
stroke_work = data[data['stroke'] == 1]['work_type'].value_counts()
print(stroke_work)
healthy_work = data[data['stroke'] == 0]['work_type'].value_counts()
print(healthy_work)

private = work.values[0]
self =  work.values[1]
child = work.values[2]
gov = work.values[3]
never = work.values[4]

stroke_private = stroke_work.values[0] / private * 100
stroke_self = stroke_work.values[1]  / self *100
stroke_child = stroke_work.values[3]  / child * 100
stroke_gov = stroke_work.values[2]  / gov *100
stroke_never = 0

healthy_private = healthy_work.values[0] / private * 100
healthy_self = healthy_work.values[1] / self *100
healthy_child = healthy_work.values[2] / child * 100
healthy_gov = healthy_work.values[3]/ gov *100
healthy_never = healthy_work.values[4]/ never *100

private_per =private/(private+self+child+gov+never) * 100
self_per = self/(private+self+child+gov+never)* 100
child_per =child/(private+self+child+gov+never) * 100
gov_per = gov/(private+self+child+gov+never)* 100
never_per =never/(private+self+child+gov+never)* 100


groups = ['Private', 'Self','Children','Goverment','Never Worked']
values1 = [healthy_private ,healthy_self,healthy_child,healthy_gov,healthy_never]
values2 = [stroke_private,stroke_self,stroke_child,stroke_gov,stroke_never]

fig = plt.figure(figsize = (30,20), dpi = 60)

gs = fig.add_gridspec(10,24)
gs.update(wspace = 1, hspace = 0.05)

ax = fig.add_subplot(gs[6:9, 0:8])
ax.text(-0.45, 120,'Work Type - Stroke Distribution', {'font':'Serif', 'weight':'bold','color': 'black', 'size':24}, alpha= 0.9)

# Stacked bar chart
ax.bar(groups, values1, color='#009786')
ax.bar(groups, values2, bottom = values1, color='#df1529')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.yaxis.set_major_locator(mtick.MultipleLocator(20))

# Labels
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2 + bar.get_y(),
            str(round(bar.get_height(), 1)) + '%',  # Add percentage symbol to the label
            ha='center', color='black',size=16 ,font='Serif')

ax.text(5.88, 100, 'Stroke ', {'font': 'Serif','weight':'bold','size': '16','style':'normal', 'color':'#df1529'})
ax.text(5.88, 110,'No Stroke', {'font': 'Serif','weight':'bold', 'size': '16','style':'normal', 'color':"#009786"})
ax.set_xlabel('Work Type', fontdict = {'font':'Serif', 'color': 'black', 'weight':'bold','size': 16})
ax.set_ylabel('Precentage', fontdict = {'font':'Serif', 'color': 'black', 'weight':'bold','size': 16})


# ###### 3)Work Type by working and non working- Stroke

# In[41]:


work = data['work_type'].value_counts()
stroke_work = data[data['stroke'] == 1]['work_type'].value_counts()
healthy_work = data[data['stroke'] == 0]['work_type'].value_counts()

private = work.values[0]
self =  work.values[1]
child = work.values[2]
gov = work.values[3]
never = work.values[4]

working= (private + self+ gov)
nonworking = (child + never)
print(nonworking)

stroke_working= (stroke_work.values[0]+stroke_work.values[1] + stroke_work.values[2] )  / working * 100
stroke_nonworking = (stroke_work.values[3]+0) / nonworking *100


healthy_working = (healthy_work.values[0]+healthy_work.values[1] + healthy_work.values[3] )  / working * 100
healthy_nonworking =(healthy_work.values[2]+healthy_work.values[4]) / nonworking *100


groups = ['Working', 'Non Working']
values1 = [healthy_working ,healthy_nonworking]
values2 = [stroke_working,stroke_nonworking]

fig = plt.figure(figsize = (30,20), dpi = 60)

gs = fig.add_gridspec(10,24)
gs.update(wspace = 1, hspace = 0.05)

ax = fig.add_subplot(gs[6:9, 0:8])
ax.text(-0.45, 120,'Work Type - Stroke Distribution', {'font':'Serif', 'weight':'bold','color': 'black', 'size':24}, alpha= 0.9)

# Stacked bar chart
ax.bar(groups, values1, color='#009786')
ax.bar(groups, values2, bottom = values1, color='#df1529')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.yaxis.set_major_locator(mtick.MultipleLocator(20))

# Labels
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2 + bar.get_y(),
            str(round(bar.get_height(), 1)) + '%',  # Add percentage symbol to the label
            ha='center', color='black',size=16 ,font='Serif')

ax.text(2, 100, 'Stroke ', {'font': 'Serif','weight':'bold','size': '16','style':'normal', 'color':'#df1529'})
ax.text(2, 110,'No Stroke', {'font': 'Serif','weight':'bold', 'size': '16','style':'normal', 'color':"#009786"})
ax.set_xlabel('Work Type', fontdict = {'font':'Serif', 'color': 'black', 'weight':'bold','size': 16})
ax.set_ylabel('Precentage', fontdict = {'font':'Serif', 'color': 'black', 'weight':'bold','size': 16})


# In[13]:


pwd


# In[53]:


data_onehot=data.copy()
data_onehot = pd.get_dummies(data_onehot, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'])
data_onehot.to_excel('preprocessed_data_new.xlsx', index=False)


# In[54]:


data_onehot.head()


# In[57]:


from sklearn.model_selection import train_test_split

np.random.seed(42)  # set the random seed for reproducibility

X = data_onehot.drop(['stroke'], axis=1)
y = data_onehot['stroke']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[58]:


from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

# Separate the features and target variable
X = data_onehot.drop('stroke', axis=1)
y = data_onehot['stroke']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PLS regression
pls = PLSRegression(n_components=2)
pls.fit(X_scaled, y)


# In[59]:


# Calculate the variable importance scores
vip = np.abs(pls.coef_).sum(axis=1)

# Normalize the scores
vip /= vip.sum()

# Sort the scores in descending order and get the corresponding feature names
sorted_idx = vip.argsort()[::-1]
sorted_features = X.columns[sorted_idx]

# Plot the scores
plt.figure(figsize=(8, 6))
plt.bar(sorted_features, vip[sorted_idx], color="#009786")
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Variable Importance')
plt.title('Variable Importance Plot')
plt.show()


print(sorted_features, vip[sorted_idx])


# In[109]:


# Plot the loadings for the input features
fig, ax = plt.subplots(figsize=(8, 6))
plt.scatter(pls.x_loadings_[:, 0], pls.x_loadings_[:, 1], color="#009786")
plt.scatter(pls.y_loadings_[:, 0], pls.y_loadings_[:, 1],color="#df1529")
plt.xlabel('First PLS Component')
plt.ylabel('Second PLS Component')

# Add the proportion of variation explained by each component to the title
var_explained = f'{x_var_explained*100:.2f}% of X variance, {y_var_explained*100:.2f}% of Y variance'
plt.title('Loadings Plot')

for i, var in enumerate(X.columns):
    plt.annotate(var, (pls.x_loadings_[i, 0], pls.x_loadings_[i, 1]))
plt.annotate('stroke', (pls.y_loadings_[0, 0], pls.y_loadings_[0, 1]))

ax.axhline(y=0, linestyle='--', color='gray')
ax.axvline(x=0, linestyle='--', color='gray')
plt.show()


x_scores = pls.x_scores_
y_scores = pls.y_scores_

# Calculate the proportion of variance explained by each component
total_var = np.var(X_scaled, axis=0).sum()
var_explained_1 = np.var(x_scores[:, 0]) / total_var
var_explained_2 = np.var(x_scores[:, 1]) / total_var

# Print the results
print(f"Proportion of variance explained by first PLS component: {var_explained_1:.2%}")
print(f"Proportion of variance explained by second PLS component: {var_explained_2:.2%}")



# Plot the scores for the first two PLS components
plt.figure(figsize=(8, 6))
colors = ['#df1529' if i == 1 else "#009786" for i in y]
plt.scatter(x_scores[:, 0], x_scores[:, 1], c=colors)
plt.xlabel('PLS Component 1')
plt.ylabel('PLS Component 2')
plt.colorbar(label='Stroke')
plt.title('Scores Plot')
plt.show()


# In[61]:


from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import matplotlib.pyplot as plt

# Fit a PLS-DA model with two LVs
plsda = PLSRegression(n_components=2)
X_scores, y_scores = plsda.fit_transform(X, y)

# Fit a linear discriminant analysis (LDA) model to the PLS-DA scores
lda = LinearDiscriminantAnalysis()
lda.fit(X_scores, y)



# In[114]:


# Extract the PLS components
X_pls = pls.transform(X_scaled)

# Create a new data frame with the PLS scores and outcome variable
df_pls = pd.DataFrame(X_pls, columns=['PLS1', 'PLS2'])
df_pls['Stroke'] = y.values

# Create a scatter plot of the PLS components
plt.figure(figsize=(8, 6))


# Plot the stroke=0 points in green
plt.scatter(df_pls.loc[df_pls['Stroke']==0, 'PLS1'], 
            df_pls.loc[df_pls['Stroke']==0, 'PLS2'], 
            color="#009786", alpha=0.7, label='Stroke=0')

# Plot the stroke=1 points in red
plt.scatter(df_pls.loc[df_pls['Stroke']==1, 'PLS1'], 
            df_pls.loc[df_pls['Stroke']==1, 'PLS2'], 
            color='#df1529', alpha=0.7, label='Stroke=1')


plt.xlabel('PLS Component 1')
plt.ylabel('PLS Component 2')
plt.title('Scores Plot')
plt.legend()
plt.show()


# In[111]:


plt.figure(figsize=(12,8))
ax = sns.heatmap(data.corr(), annot=True)
plt.show()


plt.figure(figsize=(12, 8))
ax = sns.heatmap(data.corr(), annot=True, cmap=sns.diverging_palette(10, 220, sep=80, n=5), vmin=-1, vmax=1)
ax.set_title('Correlation Heatmap', fontsize=18, fontweight='bold')
plt.show()

