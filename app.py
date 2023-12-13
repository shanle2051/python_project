#!/usr/bin/env python
# coding: utf-8

# # Final Project 

# ### Define problem
# + Analyze data on social media data among the US public and build a model that takes predicts whether someone uses LinkedIn.

# ### Import packages

# In[88]:


import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# ### Q1 Read data

# In[89]:


s = pd.read_csv("social_media_usage.csv")

# Check dimensions 
s.shape


# In[90]:


s.head


# #### Q2: clean_sm Function 

# In[91]:


def clean_sm (x):
     return np.where(x == 1, 1, 0)


# In[92]:


# toy data frame
toy_df = pd.DataFrame({
    "blue": [10, 1, 12],
    "green": [15, 18, 23],
    "yellow": [1, 9, 19]})
    
toy_df


# In[93]:


toy_cleaned = clean_sm(toy_df)

toy_cleaned


# #### Q3: ss Data frame

# In[94]:


# Read column names
s.columns


# In[95]:


target_column = "web1h"
ss = s.copy()


# In[96]:


ss["sm_li"] = ss[target_column].apply(clean_sm)


# In[97]:


ss


# In[98]:


# Drop missing values
ss = ss.dropna()
ss


# In[99]:


ss.isna


# In[100]:


# Remove missing values based on criteria


# In[101]:


ss = ss[(ss['income'] >= 1) & (ss['income'] <= 9) &
        (ss['educ2'] >= 1) & (ss['educ2'] <= 8) &
        (ss['age'] <= 98)].copy()

ss


# In[102]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[103]:


# Explore data
ss.head


# #### Q4: Target and Feature

# In[155]:


features = ["income","educ2","par", "marital", "age", "gender"]

X = ss[features]
y = ss["sm_li"]


# #### Q5: Split Data

# In[156]:


# Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,      
                                                    test_size=0.2,   
                                                    random_state=987) 

# X_train: 80% of "ss" data and the features used to predict the a response variable during model training 
# y_train: 80% of "ss" data and the target (y) that we will predict using the explanatory variables during model training

# X_test: 20% of "ss" data and the features from the model on unseen data
# y_test: 20% of "ss" data and the target (y) to predict and evaluate performance


# In[157]:


X_train.shape


# In[158]:


X_test.shape


# In[159]:


y_train.shape


# In[160]:


y_test.shape


# #### Q6: Train Model

# In[161]:


# Initialize algorithm 
lr = LogisticRegression(class_weight='balanced')


# In[162]:


# Fit model
lr.fit(X_train, y_train)


# #### Q7: Evaluate Model

# In[163]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[164]:


# Make predictions using the model and the testing data
y_pred = lr.predict(X_test)


# In[165]:


accuracy_score(y_test, y_pred)


# In[166]:


# Generate a confusion matrix 
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix


# In[167]:


# Class 0 (Negative Class):
# True Negatives (TN): 109, representing the model correctly predicted Class 0; correct negative predictions
# False Positives (FP): 59 representing the model predicted Class 1, but the true class was Class 0; correct positive predictions

# Class 1 (Positive Class):
# False Negatives (FN): 22 representing the model predicted Class 0, but the true class was Class 1; incorrect positive predictions
# True Positives (TP): 62 representing the model correctly predicted Class 1confusion_matrix(y_test, y_pred);  correct positive predictions


# #### Q8: Confusion matrix as a Dataframe

# In[168]:


pd.DataFrame(confusion_matrix(y_test, y_pred),
            columns=["Predicted negative", "Predicted positive"],
            index=["Actual negative","Actual positive"]).style.background_gradient(cmap="PiYG")


# #### Q9: precision, recall, and F1 score

# In[169]:


## recall: TP/(TP+FN)
Recall = 62/(62+22)
Recall


# In[170]:


## precision: TP/(TP+FP)
Precision = 62/(62+59)
Precision


# In[171]:


## F1 Score: 
## 2*[(0.51239*0.73809)/(0.51239+0.73809)]
 
0.51239*0.73809


# In[172]:


0.51239+0.73809


# In[173]:


0.37818993510000004/1.25048


# In[174]:


f1 = 2*0.3024358127279125
f1


# In[175]:


# Create a classification_report using sklearn
print(classification_report(y_test, y_pred))


# In[176]:


# Classification report results match manual calculations 


# #### Q10: Predictions

# In[177]:


# New data for features
person = [8, 7, 2, 1, 42, 2]

# Predict class, given input features
predicted_class = lr.predict([person])

# Generate probability of positive class (=1)
probs = lr.predict_proba([person])


# In[178]:


predicted_class[0]


# In[179]:


probs[0][1]


# In[180]:


# New data for features
person1 = [8, 7, 2, 1, 82, 2]

# Predict class, given input features
predicted_class1 = lr.predict([person1])

# Generate probability of positive class (=1)
probs1 = lr.predict_proba([person1])


# In[182]:


predicted_class1[0]


# In[185]:


probs1[0][1]


# In[186]:


from joblib import dump

dump(lr, "model_trained")

