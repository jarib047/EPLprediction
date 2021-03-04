#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import itertools

get_ipython().run_line_magic('matplotlib', 'inline')


# In[19]:


#reading data from csv into a dataframe
raw_data_1 = pd.read_csv('E:/2000-01.csv')
raw_data_2 = pd.read_csv('E:/2001-02.csv')
raw_data_3 = pd.read_csv('E:/2002-03.csv')
raw_data_4 = pd.read_csv('E:/2003-04.csv')
raw_data_5 = pd.read_csv('E:/2004-05.csv')
raw_data_6 = pd.read_csv('E:/2005-06.csv')
raw_data_7 = pd.read_csv('E:/2006-07.csv')
raw_data_8 = pd.read_csv('E:/2007-08.csv')
raw_data_9 = pd.read_csv('E:/2008-09.csv')
raw_data_10 = pd.read_csv('E:/2009-10.csv')
raw_data_11 = pd.read_csv('E:/2010-11.csv')
raw_data_12 = pd.read_csv('E:/2011-12.csv')
raw_data_13 = pd.read_csv('E:/2012-13.csv')
raw_data_14 = pd.read_csv('E:/2013-14.csv')
raw_data_15 = pd.read_csv('E:/2014-15.csv')
raw_data_16 = pd.read_csv('E:/2015-16.csv')
raw_data_17 = pd.read_csv('E:/2016-17.csv')
raw_data_18 = pd.read_csv('E:/2017-18.csv')
raw_data_19 = pd.read_csv('E:/2018-19.csv')
raw_data_20 = pd.read_csv('E:/2019-20.csv')



raw_data_20.head()


# In[20]:


#getting all the stats related to gameplay

columns_req=['Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR']

playing_stats_1=raw_data_1[columns_req]
playing_stats_2=raw_data_2[columns_req]
playing_stats_3=raw_data_3[columns_req]
playing_stats_4=raw_data_4[columns_req]
playing_stats_5=raw_data_5[columns_req]
playing_stats_6=raw_data_6[columns_req]
playing_stats_7=raw_data_7[columns_req]
playing_stats_8=raw_data_8[columns_req]
playing_stats_9=raw_data_9[columns_req]
playing_stats_10=raw_data_10[columns_req]
playing_stats_11=raw_data_11[columns_req]
playing_stats_12=raw_data_12[columns_req]
playing_stats_13=raw_data_13[columns_req]
playing_stats_14=raw_data_14[columns_req]
playing_stats_15=raw_data_15[columns_req]
playing_stats_16=raw_data_16[columns_req]
playing_stats_17=raw_data_17[columns_req]
playing_stats_18=raw_data_18[columns_req]
playing_stats_19=raw_data_19[columns_req]
playing_stats_20=raw_data_20[columns_req]


# In[21]:


df=playing_stats_20
df.head()


# In[22]:


# Gets the goals scored agg arranged by teams and matchweek
def get_goals_scored(playing_stat):
    teams = {}
    for i in playing_stat.groupby('HomeTeam').mean().T.columns:
        teams[i] = []
    
   
    for i in range(len(playing_stat)):
        HTGS = playing_stat.iloc[i]['FTHG']
        ATGS = playing_stat.iloc[i]['FTAG']
        teams[playing_stat.iloc[i].HomeTeam].append(HTGS)
        teams[playing_stat.iloc[i].AwayTeam].append(ATGS)
    
    GoalsScored = pd.DataFrame(data=teams, index = [i for i in range(1,39)]).T
    GoalsScored[0] = 0
    # Aggregate to get uptil that point
    for i in range(2,39):
        GoalsScored[i] = GoalsScored[i] + GoalsScored[i-1]
    return GoalsScored



# Gets the goals conceded agg arranged by teams and matchweek
def get_goals_conceded(playing_stat):
    teams = {}
    for i in playing_stat.groupby('HomeTeam').mean().T.columns:
        teams[i] = []
    
    for i in range(len(playing_stat)):
        ATGC = playing_stat.iloc[i]['FTHG']
        HTGC = playing_stat.iloc[i]['FTAG']
        teams[playing_stat.iloc[i].HomeTeam].append(HTGC)
        teams[playing_stat.iloc[i].AwayTeam].append(ATGC)
    
    GoalsConceded = pd.DataFrame(data=teams, index = [i for i in range(1,39)]).T
    GoalsConceded[0] = 0
    for i in range(2,39):
        GoalsConceded[i] = GoalsConceded[i] + GoalsConceded[i-1]
    return GoalsConceded

def get_gss(playing_stat):
    #get goal diff stats
    GC = get_goals_conceded(playing_stat)
    GS = get_goals_scored(playing_stat)
   
    j = 0
    HTGS = []
    ATGS = []
    HTGC = []
    ATGC = []

    for i in range(380):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        HTGS.append(GS.loc[ht][j])
        ATGS.append(GS.loc[at][j])
        HTGC.append(GC.loc[ht][j])
        ATGC.append(GC.loc[at][j])
        
        if ((i + 1)% 10) == 0:
            j = j + 1
        
    playing_stat['HTGS'] = HTGS
    playing_stat['ATGS'] = ATGS
    playing_stat['HTGC'] = HTGC
    playing_stat['ATGC'] = ATGC
    
    return playing_stat


playing_stats_1 = get_gss(playing_stats_1)
playing_stats_2 = get_gss(playing_stats_2)
playing_stats_3 = get_gss(playing_stats_3)
playing_stats_4 = get_gss(playing_stats_4)
playing_stats_5 = get_gss(playing_stats_5)
playing_stats_6 = get_gss(playing_stats_6)
playing_stats_7 = get_gss(playing_stats_7)
playing_stats_8 = get_gss(playing_stats_8)
playing_stats_9 = get_gss(playing_stats_9)
playing_stats_10 = get_gss(playing_stats_10)
playing_stats_11 = get_gss(playing_stats_11)
playing_stats_12 = get_gss(playing_stats_12)
playing_stats_13 = get_gss(playing_stats_13)
playing_stats_14 = get_gss(playing_stats_14)
playing_stats_15 = get_gss(playing_stats_15)
playing_stats_16 = get_gss(playing_stats_16)
playing_stats_17 = get_gss(playing_stats_17)
playing_stats_18 = get_gss(playing_stats_18)
playing_stats_19 = get_gss(playing_stats_19)
playing_stats_20 = get_gss(playing_stats_20)


# In[23]:


playing_stats_1.tail(20)


# In[24]:


# Convert the result of the game into a score, win three points, score one point, lose no points
def get_points(result):
    if result == 'W':
        return 3
    elif result == 'D':
        return 1
    else:
        return 0
    

def get_cuml_points(matchres):
    matchres_points = matchres.applymap(get_points)
    for i in range(2,39):
        matchres_points[i] = matchres_points[i] + matchres_points[i-1]
        
    matchres_points.insert(column =0, loc = 0, value = [0*i for i in range(20)])
    return matchres_points


def get_matchres(playing_stat):
    teams = {}
    for i in playing_stat.groupby('HomeTeam').mean().T.columns:
        teams[i] = []
        
    # Record the results of the match in the home team and the away team



    for i in range(len(playing_stat)):
        if playing_stat.iloc[i].FTR == 'H':
            teams[playing_stat.iloc[i].HomeTeam].append('W')
            teams[playing_stat.iloc[i].AwayTeam].append('L')
        elif playing_stat.iloc[i].FTR == 'A':
            teams[playing_stat.iloc[i].AwayTeam].append('W')
            teams[playing_stat.iloc[i].HomeTeam].append('L')
        else:
            teams[playing_stat.iloc[i].AwayTeam].append('D')
            teams[playing_stat.iloc[i].HomeTeam].append('D')
            
    return pd.DataFrame(data=teams, index = [i for i in range(1,39)]).T

def get_agg_points(playing_stat):
    matchres = get_matchres(playing_stat)
    cum_pts = get_cuml_points(matchres)
    HTP = []
    ATP = []
    j = 0
    for i in range(380):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        HTP.append(cum_pts.loc[ht][j])
        ATP.append(cum_pts.loc[at][j])

        if ((i + 1)% 10) == 0:
            j = j + 1

    
    playing_stat['HTP'] = HTP
    playing_stat['ATP'] = ATP
    return playing_stat

playing_stats_1 = get_agg_points(playing_stats_1)
playing_stats_2 = get_agg_points(playing_stats_2)
playing_stats_3 = get_agg_points(playing_stats_3)
playing_stats_4 = get_agg_points(playing_stats_4)
playing_stats_5 = get_agg_points(playing_stats_5)
playing_stats_6 = get_agg_points(playing_stats_6)
playing_stats_7 = get_agg_points(playing_stats_7)
playing_stats_8 = get_agg_points(playing_stats_8)
playing_stats_9 = get_agg_points(playing_stats_9)
playing_stats_10 = get_agg_points(playing_stats_10)
playing_stats_11 = get_agg_points(playing_stats_11)
playing_stats_12 = get_agg_points(playing_stats_12)
playing_stats_13 = get_agg_points(playing_stats_13)
playing_stats_14 = get_agg_points(playing_stats_14)
playing_stats_15 = get_agg_points(playing_stats_15)
playing_stats_16 = get_agg_points(playing_stats_16)
playing_stats_17 = get_agg_points(playing_stats_17)
playing_stats_18 = get_agg_points(playing_stats_18)
playing_stats_19 = get_agg_points(playing_stats_19)
playing_stats_20 = get_agg_points(playing_stats_20)


# In[25]:


playing_stats_12.head(20)


# In[26]:


def get_form(playing_stat,num):
    form = get_matchres(playing_stat)
    form_final = form.copy()
    for i in range(num,39):
        form_final[i] = ''
        j = 0
        while j < num:
            form_final[i] += form[i-j]
            j += 1           
    return form_final

def add_form(playing_stat,num):
    form = get_form(playing_stat,num)
    h = ['M' for i in range(num * 10)]  
    a = ['M' for i in range(num * 10)]
    
    j = num
    for i in range((num*10),380):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        
        past = form.loc[ht][j]              
        h.append(past[num-1])                    
        
        past = form.loc[at][j]              
        a.append(past[num-1])                 
        
        if ((i + 1)% 10) == 0:
            j = j + 1

    playing_stat['HM' + str(num)] = h                 
    playing_stat['AM' + str(num)] = a

    
    return playing_stat


def add_form_df(playing_stats):
    playing_stats = add_form(playing_stats,1)
    playing_stats = add_form(playing_stats,2)
    playing_stats = add_form(playing_stats,3)
    playing_stats = add_form(playing_stats,4)
    playing_stats = add_form(playing_stats,5)
    return playing_stats   

playing_stats_1 = add_form_df(playing_stats_1)
playing_stats_2 = add_form_df(playing_stats_2)
playing_stats_3 = add_form_df(playing_stats_3)
playing_stats_4 = add_form_df(playing_stats_4)
playing_stats_5 = add_form_df(playing_stats_5)
playing_stats_6 = add_form_df(playing_stats_6)
playing_stats_7 = add_form_df(playing_stats_7)
playing_stats_8 = add_form_df(playing_stats_8)
playing_stats_9 = add_form_df(playing_stats_9)
playing_stats_10 = add_form_df(playing_stats_10)
playing_stats_11 = add_form_df(playing_stats_11)
playing_stats_12 = add_form_df(playing_stats_12)
playing_stats_13 = add_form_df(playing_stats_13)
playing_stats_14 = add_form_df(playing_stats_14)
playing_stats_15 = add_form_df(playing_stats_15)
playing_stats_16 = add_form_df(playing_stats_16)
playing_stats_17 = add_form_df(playing_stats_17)
playing_stats_18 = add_form_df(playing_stats_18)
playing_stats_19 = add_form_df(playing_stats_19)
playing_stats_20 = add_form_df(playing_stats_20)


# In[27]:


# Rearranging columns
cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTGS', 'ATGS', 'HTGC', 'ATGC', 'HTP', 'ATP', 'HM1', 'HM2', 'HM3',
        'HM4', 'HM5', 'AM1', 'AM2', 'AM3', 'AM4', 'AM5' ]

playing_stats_1 = playing_stats_1[cols]
playing_stats_2 = playing_stats_2[cols]
playing_stats_3 = playing_stats_3[cols]
playing_stats_4 = playing_stats_4[cols]
playing_stats_5 = playing_stats_5[cols]
playing_stats_6 = playing_stats_6[cols]
playing_stats_7 = playing_stats_7[cols]
playing_stats_8 = playing_stats_8[cols]
playing_stats_9 = playing_stats_9[cols]
playing_stats_10 = playing_stats_10[cols]
playing_stats_11 = playing_stats_11[cols]
playing_stats_12 = playing_stats_12[cols]
playing_stats_13 = playing_stats_13[cols]
playing_stats_14 = playing_stats_14[cols]
playing_stats_15 = playing_stats_15[cols]
playing_stats_16 = playing_stats_16[cols]
playing_stats_17 = playing_stats_17[cols]
playing_stats_18 = playing_stats_18[cols]
playing_stats_19 = playing_stats_19[cols]
playing_stats_20 = playing_stats_20[cols]


# In[28]:


playing_stats_1.tail()


# In[26]:


def get_mw(playing_stat):
    j = 1
    MatchWeek = []
    for i in range(380):
        MatchWeek.append(j)
        if ((i + 1)% 10) == 0:
            j = j + 1
    playing_stat['MW'] = MatchWeek
    return playing_stat

playing_stats_1 = get_mw(playing_stats_1)
playing_stats_2 = get_mw(playing_stats_2)
playing_stats_3 = get_mw(playing_stats_3)
playing_stats_4 = get_mw(playing_stats_4)
playing_stats_5 = get_mw(playing_stats_5)
playing_stats_6 = get_mw(playing_stats_6)
playing_stats_7 = get_mw(playing_stats_7)
playing_stats_8 = get_mw(playing_stats_8)
playing_stats_9 = get_mw(playing_stats_9)
playing_stats_10 = get_mw(playing_stats_10)
playing_stats_11 = get_mw(playing_stats_11)
playing_stats_12 = get_mw(playing_stats_12)
playing_stats_13 = get_mw(playing_stats_13)
playing_stats_14 = get_mw(playing_stats_14)
playing_stats_15 = get_mw(playing_stats_15)
playing_stats_16 = get_mw(playing_stats_16)
playing_stats_17 = get_mw(playing_stats_17)
playing_stats_18 = get_mw(playing_stats_18)


# In[27]:


playing_stat = pd.concat([playing_stats_1,
                          playing_stats_2,
                          playing_stats_3,
                          playing_stats_4,
                          playing_stats_5,
                          playing_stats_6,
                          playing_stats_7,
                          playing_stats_8,
                          playing_stats_9,
                          playing_stats_10,
                          playing_stats_11,
                          playing_stats_12,
                          playing_stats_13,
                          playing_stats_14,
                          playing_stats_15,
                          playing_stats_16,
                          playing_stats_17,
                          playing_stats_18
                          ], ignore_index=True)


# Gets the form points.
def get_form_points(string):
    sum = 0
    for letter in string:
        sum += get_points(letter)
    return sum

playing_stat['HTFormPtsStr'] = playing_stat['HM1'] + playing_stat['HM2'] + playing_stat['HM3'] + playing_stat['HM4'] + playing_stat['HM5']
playing_stat['ATFormPtsStr'] = playing_stat['AM1'] + playing_stat['AM2'] + playing_stat['AM3'] + playing_stat['AM4'] + playing_stat['AM5']

playing_stat['HTFormPts'] = playing_stat['HTFormPtsStr'].apply(get_form_points)
playing_stat['ATFormPts'] = playing_stat['ATFormPtsStr'].apply(get_form_points)

# Identify Win/Loss Streaks if any.
def get_3game_ws(string):
    if string[-3:] == 'WWW':
        return 1
    else:
        return 0
    
def get_5game_ws(string):
    if string == 'WWWWW':
        return 1
    else:
        return 0
    
def get_3game_ls(string):
    if string[-3:] == 'LLL':
        return 1
    else:
        return 0
    
def get_5game_ls(string):
    if string == 'LLLLL':
        return 1
    else:
        return 0
    
playing_stat['HTWinStreak3'] = playing_stat['HTFormPtsStr'].apply(get_3game_ws)
playing_stat['HTWinStreak5'] = playing_stat['HTFormPtsStr'].apply(get_5game_ws)
playing_stat['HTLossStreak3'] = playing_stat['HTFormPtsStr'].apply(get_3game_ls)
playing_stat['HTLossStreak5'] = playing_stat['HTFormPtsStr'].apply(get_5game_ls)

playing_stat['ATWinStreak3'] = playing_stat['ATFormPtsStr'].apply(get_3game_ws)
playing_stat['ATWinStreak5'] = playing_stat['ATFormPtsStr'].apply(get_5game_ws)
playing_stat['ATLossStreak3'] = playing_stat['ATFormPtsStr'].apply(get_3game_ls)
playing_stat['ATLossStreak5'] = playing_stat['ATFormPtsStr'].apply(get_5game_ls)

playing_stat.keys()


# In[28]:


# Get Goal Difference
playing_stat['HTGD'] = playing_stat['HTGS'] - playing_stat['HTGC']
playing_stat['ATGD'] = playing_stat['ATGS'] - playing_stat['ATGC']

# Diff in points
playing_stat['DiffPts'] = playing_stat['HTP'] - playing_stat['ATP']
playing_stat['DiffFormPts'] = playing_stat['HTFormPts'] - playing_stat['ATFormPts']


# In[29]:


# Scale DiffPts , DiffFormPts, HTGD, ATGD by Matchweek.
cols = ['HTGD','ATGD','DiffPts','DiffFormPts','HTP','ATP']
playing_stat.MW = playing_stat.MW.astype(float)

for col in cols:
    playing_stat[col] = playing_stat[col] / playing_stat.MW


# In[30]:


def only_hw(string):
    if string == 'H':
        return 'H'
    else:
        return 'NH'
    
playing_stat['FTR'] = playing_stat.FTR.apply(only_hw)

# Testing set (2015-16 season)
playing_stat_test = playing_stat[5700:]


# In[31]:


playing_stat.to_csv('finall_dataset.csv')


# In[32]:


playing_stat_test.to_csv("testt_set.csv")


# In[33]:


dataset = pd.read_csv('finall_dataset.csv')


# In[34]:


dataset.head()


# In[35]:


dataset.keys()


# In[36]:


plt.figure(figsize=(20,10)) 
sns.heatmap(dataset.corr(), annot= True)


# In[37]:


# Remove few column
dataset2 = dataset.copy().drop(columns =['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG',
       'HTGS', 'ATGS', 'HTGC', 'ATGC',
       'HM4', 'HM5','AM4', 'AM5', 'MW', 'HTFormPtsStr',
       'ATFormPtsStr', 'HTFormPts', 'ATFormPts', 'HTWinStreak3',
       'HTWinStreak5', 'HTLossStreak3', 'HTLossStreak5', 'ATWinStreak3',
       'ATWinStreak5', 'ATLossStreak3', 'ATLossStreak5',
       'DiffPts'] )


# In[38]:


dataset2.keys()


# In[39]:


dataset2.head(10)


# In[40]:



n_matches = dataset2.shape[0]

# Calculate number of features. -1 because we are saving one as the target variable (win/lose/draw)
n_features = dataset2.shape[1] - 1

n_homewins = len(dataset2[dataset2.FTR == 'H'])

win_rate = (float(n_homewins) / (n_matches)) * 100

print("Total number of matches: {}".format(n_matches))
print ("Number of features: {}".format(n_features))
print( "Number of matches won by home team: {}".format(n_homewins))
print ("Win rate of home team: {:.2f}%".format(win_rate))


# In[41]:


# Visualising distribution of data
from pandas.plotting import scatter_matrix


scatter_matrix(dataset2[['HTGD','ATGD','HTP','ATP','DiffFormPts']], figsize=(15,15))


# In[42]:


#FTR = Full Time Result (H=Home Win, D=Draw, A=Away Win)
X_all = dataset2.drop(['FTR'],1)
y_all = dataset2['FTR']

from sklearn.preprocessing import scale

#Center to the mean and component wise scale to unit variance.
cols = [['HTGD','ATGD','HTP','ATP']]
for col in cols:
    X_all[col] = scale(X_all[col])
    


# In[43]:


#last 3 wins for both sides
X_all.HM1 = X_all.HM1.astype('str')
X_all.HM2 = X_all.HM2.astype('str')
X_all.HM3 = X_all.HM3.astype('str')
X_all.AM1 = X_all.AM1.astype('str')
X_all.AM2 = X_all.AM2.astype('str')
X_all.AM3 = X_all.AM3.astype('str')

def preprocess_features(X):
    
    output = pd.DataFrame(index = X.index)

    for col, col_data in X.iteritems():

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix = col)
                    
        output = output.join(col_data)
    
    return output

X_all = preprocess_features(X_all)
print ("Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns)))


# In[44]:


X_all.head(10)


# In[45]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, 
                                                    test_size = 0.3,
                                                    random_state = 2,
                                                    stratify = y_all)


# In[46]:


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# In[47]:


Y_pred = classifier.predict(X_test)


# In[48]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, Y_pred)


# In[49]:


sns.heatmap(cm, annot=True,fmt='d')


# In[50]:


print(classification_report(y_test, Y_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




