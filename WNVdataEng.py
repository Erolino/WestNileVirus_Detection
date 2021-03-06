#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 14:11:00 2019

@author: eran
"""

import numpy as np
import pandas as pd
import os

directory_path='/Users/eran/Galvanize_more_repositories/WestNileVirus'

train=pd.read_csv(os.path.join(directory_path,'all/train.csv'))

'''##########################################################'''

''' TRAIN + SPRAY PART - CLEAN, ENGINEER AND MERGE'''

'''##########################################################'''

# create dummies for species
train1=pd.get_dummies(train,columns=['Species'],drop_first=True)    
train1['Date']=pd.to_datetime(train1['Date'])

# EDA function, visualization of where and when collections and viruses occured:
import matplotlib.pyplot as plt
import seaborn as sns
train2=train1.copy()
def pltloglat(period,yearr=2007):
    if period=='month':
        period=['May','June','July', 'August','September','October']
    if period=='year':
        period=[2007,2009,2011,2013]
    DATA=train2
    fig,ax = plt.subplots(nrows=1,ncols=len(period), sharey=True)
    if len(period)==6:
        fig.set_size_inches(15, 4)
        print('Year -',yearr)
    elif len(period)==4:
        fig.set_size_inches(15, 5)        
    for ii,ob in enumerate(period):
        if len(period)==6: 
            train_0=DATA[(DATA['year']==yearr)&(DATA['WnvPresent']==0)&(DATA['month']==ii+5)]
            train_1=DATA[(DATA['year']==yearr)&(DATA['WnvPresent']==1)&(DATA['month']==ii+5)]
        elif len(period)==4:
            train_0=DATA[(DATA['year']==period[ii])&(DATA['WnvPresent']==0)]
            train_1=DATA[(DATA['year']==period[ii])&(DATA['WnvPresent']==1)]   
        ax[ii].set_title(ob)
        ax[ii].set_xlim([-88.0,-87.5])
        sns.regplot('Longitude','Latitude',data=train_0,fit_reg=False,ax=ax[ii],color='steelblue')
        sns.regplot('Longitude','Latitude',data=train_1,fit_reg=False,ax=ax[ii],color='orange',marker='+',)

        
spray=pd.read_csv(os.path.join(directory_path,'all/spray.csv'))

# Let's create a function that finds the days since the most recent spray, 
# given a date of collection minus dates of spraying:
def recent(delMin,delMax):
    if (delMin>=0) & (delMax<0):
        dell=delMin
    elif (delMax>=0) & (delMin<0):
        dell=delMax
    elif (delMax<0) & (delMin<0):
        dell=3650
    elif (delMax>=0) & (delMin>=0):
        if delMax<delMin:
            dell=delMax
        elif delMax>delMin:
            dell=delMin
        else:
            dell=delMin
    elif delMin.isnull() or delMax.isnull():
        dell=3650
    return(dell)
#def engineerSP(spray):
 

spray1=spray.copy()
# because geo-location has such high resolution, we can't find matches between the data 
#sets (which doesn't allow us to merge the tables accordingly). so 3 digits after the 
# point seems reasonable - equates to 100 meters in resolution ( according to quick 
# exploration on google maps: .001 ~=100 Meters)
spray1['Longitude']=spray['Longitude'].round(3) ## rounding to match longitudes in train data (resolution of 100 M)
spray1['Latitude']=spray['Latitude'].round(3)
train1['Longitude']=train1['Longitude'].round(3)
train1['Latitude']=train1['Latitude'].round(3)
spray1['Date']=pd.to_datetime(spray1['Date']) ## change to datetime object so operations could be done 
spray2=spray1.groupby(['Longitude','Latitude']).Date.agg(['count','min','max',np.ptp]).reset_index() ## get the range of spraying for each site
spray2=spray2.sort_values('ptp',ascending=False)
#merge train with spray:
sptrainright=pd.merge(spray2,train1,on=['Longitude','Latitude'],how='right',indicator=True) ## merge train with spray 
sptrainright['Date']=pd.to_datetime(sptrainright['Date']) 
sptrainright['delmin']=sptrainright['Date']-sptrainright['min']
sptrainright['delmin']=sptrainright['delmin'].dt.days
sptrainright['delmax']=sptrainright['Date']-sptrainright['max']
sptrainright['delmax']=sptrainright['delmax'].dt.days
sptrainright['most_recent_spray_(days)']=3650
## previously to create a column with the most recent spray in days
# after 962 obdservations with spray, the rest is without data on spray
# Make an inner merge to concentrate on the spray&train intersection (and avoid dealing with NaT and NaN:
sptraininner=pd.merge(spray2,train1,on=['Longitude','Latitude'],how='inner',indicator=True)    
sptraininner['Date']=pd.to_datetime(sptraininner['Date'])
sptraininner['delmin']=sptraininner['Date']-sptraininner['min']
sptraininner['delmin']=sptraininner['delmin'].dt.days
sptraininner['delmax']=sptraininner['Date']-sptraininner['max']
sptraininner['delmax']=sptraininner['delmax'].dt.days
sptraininner['most_recent_spray_(days)']=sptraininner.apply(lambda x: recent(x['delmin'],x['delmax']),axis=1)
# Now conocat the dfs
sptrain=pd.concat([sptraininner,sptrainright.loc[963:,:]])
# arrange columns name
'''
colls=sptrain.columns
colis=list(colls)
colsnew=colis[:2]+colis[3:5]+[colis[-1]]+colis[6:14]+colis[15:21]+colis[14:15] 
sptrain=sptrain[colsnew]
'''
sptrain.rename(columns={'Date':'Date_of_collection'},inplace=True)
sptrain.drop(['_merge'],1,inplace=True)
# let' add a column - whether the area was recently sprayed (i.e. <150 days)
sptrain['Recently_sprayed']=(sptrain['most_recent_spray_(days)']<150).astype(int)
sptrain['Recently_sprayed'].value_counts()

'''##########################################################'''

''' WEATHER PART - CLEAN, ENGINEER AND MERGE 
    (on: 'day of collection'=='day of forecast) '''

'''##########################################################'''

weather_csv=os.path.join(directory_path,"all/weather.csv")
weather=pd.read_csv(weather_csv)

## Create weather station column in the train data (2 different stations: 1 and 2)
## first create a function to get classify which weather station fits which trap collection (by geo-location)
# assumption: long and lat have the same resolution when converted to distance.
# for distance calculation using: distance^2=long^2+lat^2

# Reminder the geo-location of the stations: 
# Station 1: CHICAGO O'HARE INTERNATIONAL AIRPORT Lat: 41.995 Lon: -87.933 Elev: 662 ft. above sea level
# Station 2: CHICAGO MIDWAY INTL ARPT Lat: 41.786 Lon: -87.752 Elev: 612 ft. above sea level
import random
def stationing(long,lat):
    sta1={'long':-87.933,'lat': 41.995}
    sta2={'long':-87.752,'lat': 41.786}
    longdelt1=long-sta1['long']
    latdelt1=lat-sta1['lat']
    dist1 = np.sqrt((longdelt1)**2+(latdelt1)**2)
    #print("dist1 =",dist1)
    longdelt2=long-sta2['long']
    latdelt2=lat-sta2['lat']
    dist2 = np.sqrt((longdelt2)**2+(latdelt2)**2)
    #print("dist2 =",dist2)
    if dist1<dist2:
        #print("stat1",dist1)
        station=1
    elif dist1>dist2:
        station=2
        #print("stat2",dist2)
    else:
        station=random.choice([1,2])
    return(station)
    
sptrainW=sptrain.copy() # make a copy of sptrainW so we can add weather (W) features to it. (starting with station column) 
sptrainW['station']=sptrainW.apply(lambda x: stationing(x.Longitude,x.Latitude),axis=1)  
## using stationing function to match station number to every observation
## now we have station column. ~80% of the traps are closer to station 2
weather['Date']=pd.to_datetime(weather['Date'])

# merging weather and sptrain data 
sptrainW0=pd.merge(sptrainW,weather,left_on=['Date_of_collection','station'],right_on=['Date','Station'],how='left',indicator=True)
## sptrainW0 is the merged dataset before any engineering and cleanning of it
sptrainW1=sptrainW0.copy()
## let's start engineering:
sptrainW1.drop(['station','Date'],1,inplace=True)
sptrainW1.drop(['_merge',],1,inplace=True)
sptrainW1.drop(['Depart',],1,inplace=True)# Depart - # Most is M missing (8223), so droping that column
#Since number of missing values are small (26), mode or median would make sense as replacement. 
# But after doing quick search online, we can approximate WetBulb from DewPoint and 
# temperature (that we have) with this formula - TAVG-((TAVG-DEWPOINT)/3).
# (resource: http://theweatherprediction.com/habyhints/170/ )
# Writing a function for wetbulb approximation:
def wetbulb(tavg,dp,wb):
    if wb=='M':
        wb=tavg-(tavg-dp)/3
    else: 
        pass
    return(wb)


sptrainW1['Tavg']=sptrainW1['Tavg'].astype(int) ## numbers are stored as str - so turn to int, to manipulate 
# applying it to the df:  
sptrainW1['WetBulb']=sptrainW1.apply(lambda x: wetbulb(x['Tavg'],x['DewPoint'],x['WetBulb']),axis=1)
sptrainW1['WetBulb']=sptrainW1['WetBulb'].astype(int)
## continuing switching str into int in other columns
sptrainW1['Heat']=sptrainW1['Heat'].astype(int)
sptrainW1['Cool']=sptrainW1['Cool'].astype(int)
sptrainW1.drop(['Sunrise','Sunset'],1,inplace=True) ## mostly empty
#Function to turn codes into 2 groups good (' ') and bad weather ( all other codes)
# the assumption is that all codes are related to bad weather..
def codes(col):
    if col==' ':
        col='Norm'
    else:
        col='Bad'
    return(col)
codes(' ')

sptrainW1['weather_type']=sptrainW1.apply(lambda x: codes(x['CodeSum']), axis=1)
sptrainW1=pd.get_dummies(sptrainW1,columns=['weather_type'],drop_first=True)
# to check: type sptrainW1.weather_type_Norm.value_counts()
sptrainW1.drop(['Water1','SnowFall'],1,inplace=True) ## see summary, mostly missing values
# PrecipTotal - convert T (trace) to 0.005 (look at summary):
sptrainW1['PrecipTotal']=sptrainW1['PrecipTotal'].apply(lambda x: 0.005 if x=='  T' else x)
#convert 'M' to mode 
import statistics as st
mode=st.mode(sptrainW1['PrecipTotal']) # mode is '0'
sptrainW1['PrecipTotal']=sptrainW1['PrecipTotal'].apply(lambda x: mode if x=='M' else x)
sptrainW1['PrecipTotal']=sptrainW1['PrecipTotal'].astype(float) # converting to type float.
 # Depth:
sptrainW1.drop('Depth',inplace=True,axis=1) # drop, mostly 'M' rest 0 (see summary)
# Stn Pressure:
moud=st.mode(sptrainW1['StnPressure'])
sptrainW1['StnPressure']=sptrainW1['StnPressure'].apply(lambda x: moud if x=='M' else x)
sptrainW1['StnPressure']=sptrainW1['StnPressure'].astype(float)
# Sealevel
sptrainW1['SeaLevel']=sptrainW1['SeaLevel'].astype(float)
## 'ResultSpeed', 'ResultDir' are good to go (floats no missing value)
sptrainW1['AvgSpeed']=sptrainW1['AvgSpeed'].astype(float) # turn to float
# date of colletion
# let's split the date to day of the month, day of the week, month, year the strongest 
# effect would probably be the month (where the hottest months will have highest frequency of WNV), 
# and maybe recent years have been hotter, maybe traps collected on Monday have more mosquitos in them? etc.

sptrainW1['Day_of_month']=sptrainW1['Date_of_collection'].apply(lambda x: x.to_pydatetime().day)
sptrainW1['month']=sptrainW1['Date_of_collection'].apply(lambda x: x.to_pydatetime().month)
sptrainW1['year']=sptrainW1['Date_of_collection'].apply(lambda x: x.to_pydatetime().year)
sptrainW1['Day_of_week']=sptrainW1['Date_of_collection'].apply(lambda x: x.to_pydatetime().weekday())
sptrainW1['year']=sptrainW1['year']-(min(sptrainW1['year'])+1)

## df almost ready. Let's arange and drop columns:
sptrainW1.drop(['count','Block','min', 'max','ptp','Address',
                'Street','Trap','AddressNumberAndStreet','CodeSum','delmin', 'delmax'],axis=1,inplace=True)

## most recent spray (days)" feature is really effecting the feature space (because of the 
 ## majority of fabricated 3500 days). let's turn this column into 3 categories: 
 ## recently sprayed (this season <180 days), sprayed 2 yrs ago, and never sprayed (=3650).
mid=sptrainW1['most_recent_spray_(days)'][(
        sptrainW1['most_recent_spray_(days)']<3650)&(sptrainW1['most_recent_spray_(days)']>180)]
sptrainW1['sprayed_2_yrs_ago']=sptrainW1['most_recent_spray_(days)'].apply(lambda x: 1 if sum(x==mid)>0 else 0)
sptrainW1['never_sprayed']=sptrainW1['most_recent_spray_(days)'].apply(lambda x: 1 if x>3640 else 0)    
    
sptrainW2=sptrainW1.copy()
sptrainW2=sptrainW2[['Longitude', 'Latitude', 'Date_of_collection', 'AddressAccuracy',
       'NumMosquitos', 'Species_CULEX PIPIENS',
       'Species_CULEX PIPIENS/RESTUANS', 'Species_CULEX RESTUANS',
       'Species_CULEX SALINARIUS', 'Species_CULEX TARSALIS',
       'Species_CULEX TERRITANS', 'most_recent_spray_(days)',
       'Recently_sprayed', 'Station', 'Tmax', 'Tmin', 'Tavg', 'DewPoint',
       'WetBulb', 'Heat', 'Cool', 'PrecipTotal', 'StnPressure', 'SeaLevel',
       'ResultSpeed', 'ResultDir', 'AvgSpeed', 'weather_type_Norm',
       'Day_of_month', 'month', 'year', 'Day_of_week', 'sprayed_2_yrs_ago',
       'never_sprayed', 'WnvPresent']]

'''###################################

Weather - 2 weeks feature engineering 

#####################################" '''


# Functions to use inside weath_eng func:
#1) getting ag temp when there is 'M' - missing data
def tavg_fix(tavg_col,max_col,min_col):
   # Mind=np.where(W1['Tavg']=='M')
    if tavg_col=='M':
        tavg_col=(max_col+min_col)/2
    else: 
        pass
    return(tavg_col)
    
#2)
    # Wetbulb:
#Since number of missing values are small (26), mode or median would make sense as replacement. 
# But after doing quick search online, we can approximate WetBulb from DewPoint and 
# temperature (that we have) with this formula - TAVG-((TAVG-DEWPOINT)/3).
# Writing a function for wetbulb approximation:
def wetbulb(tavg,dp,wb):
    if wb=='M':
        wb=tavg-(tavg-dp)/3
    else: 
        pass
    return(wb)
    
#3)
import statistics as st

def M_rid(col,num,thing='M'):
    if col==thing:
        col=num
    else: 
        pass
    return(col)

#4)
#Function to turn codes into 2 groups good (' ') and bad weather ( all other codes)
def codes(col):
    if col==' ':
        col='Norm'
    else:
        col='Bad'
    return(col)
codes(' ')



####
def weath_eng(weather_raw):
###

    W10=weather_raw.copy()
    
    W10.drop(['Depart',],1,inplace=True)# Depart - # Most is M missing (8223), so droping that column
    
         
    W10['Tavg']=W10.apply(lambda x: tavg_fix(x['Tavg'],x['Tmax'],x['Tmin']),axis=1)
    
    W10['Tavg']=W10['Tavg'].astype(float) ## numbers are stored as str - so turn to float, to manipulate 
      
    # applying it to the df:  
    W10['WetBulb']=W10.apply(lambda x: wetbulb(x['Tavg'],x['DewPoint'],x['WetBulb']),axis=1)
    W10['WetBulb']=W10['WetBulb'].astype(float)
    
        
    W10['Heat']=W10.apply(lambda x: M_rid(x['Heat'],'0'),axis=1)
    ## continuing switching str into int in other columns
    W10['Heat']=W10['Heat'].astype(int)
    
    W10['Cool']=W10.apply(lambda x: M_rid(x['Cool'],' 0'),axis=1)
    
    W10['Cool']=W10['Cool'].astype(int)
    
    W10.drop(['Sunrise','Sunset'],1,inplace=True) ## mostly empty
    
    W10['weather_type']=W10.apply(lambda x: codes(x['CodeSum']), axis=1)

    W10=pd.get_dummies(W10,columns=['weather_type'],drop_first=True)
    
    # to check: type sptrainW1.weather_type_Norm.value_counts()
    W10.drop(['Water1','SnowFall'],1,inplace=True) ## see summary, mostly missing values
    # PrecipTotal - convert T (trace) to 0.005 (look at summary):
    W10['PrecipTotal']=W10['PrecipTotal'].apply(lambda x: 0.005 if x=='  T' else x)
    #convert 'M' to mode 
    mode=st.mode(W10['PrecipTotal']) # mode is '0'
    W10['PrecipTotal']=W10['PrecipTotal'].apply(lambda x: mode if x=='M' else x)
    W10['PrecipTotal']=W10['PrecipTotal'].astype(float) # converting to type float.
     # Depth:
    W10.drop('Depth',inplace=True,axis=1) # drop, mostly 'M' rest 0 (see summary)
    # Stn Pressure:
    moud=st.mode(W10['StnPressure'])
    W10['StnPressure']=W10['StnPressure'].apply(lambda x: moud if x=='M' else x)
    W10['StnPressure']=W10['StnPressure'].astype(float)
    # Sealevel
    mod=st.mode(W10['SeaLevel'])
    W10['SeaLevel']=W10['SeaLevel'].apply(lambda x: mod if x=='M' else x)
    W10['SeaLevel']=W10['SeaLevel'].astype(float)

    ## 'ResultSpeed', 'ResultDir' are good to go (floats no missing value)
    mod=st.mode(W10['AvgSpeed'])
    W10['AvgSpeed']=W10['AvgSpeed'].apply(lambda x: mod if x=='M' else x)
    W10['AvgSpeed']=W10['AvgSpeed'].astype(float) # turn to float
    
    return(W10)

weath_out=weath_eng(weather)


'''#####################################

Functions to engineer weather data into summaries (mean,std etc..) of 
last 14 days for every trap collection (observation) 
 
########################################'''

# functions to make 14 day summary 
# summary of one day
## important - column need to be clean, and dtype: int or float
import scipy.stats as scist

def get_summary(colum):
    des=colum.describe()  # use describe to get summary
    des=pd.DataFrame(des)  # turn into data frame
    desT=des.T  # transpose describe to a table 
    desT.rename(columns={'mean':desT.index[0]+'.'+'mean','std':desT.index[0]+'.'+'std',
                    '50%':desT.index[0]+'.'+'50%'},inplace=True) ## rename columns
    desT.drop(['count','min','25%','75%','max'],1,inplace=True)
    desT[desT.index[0]+'.'+'mean-median']=desT[desT.index[0]+'.'+'mean']-desT[desT.index[0]+'.'+'50%'] # add mean-median
    outliers_low=sum((scist.zscore(colum)<-2))
    outliers_high=sum((scist.zscore(colum)>2))
    desT[desT.index[0]+'.'+'outliers_low']=outliers_low ## add outliers
    desT[desT.index[0]+'.'+'outliers_high']=outliers_high
    desT.reset_index(drop=True,inplace=True)
    return desT

## using summary of one day to run on 14 days and make a summary:
def fourteen(col_date,station,cols,num_days=14):   # get the 14 day batch of a collection date
    # col_date is one Date of collection 
    # list of columns we'd like to use for the summary of the batch 
    ind=np.where((col_date==weath_out['Date'])
             &(station==weath_out['Station']))
    indt=int(ind[0])
    datee=weath_out.loc[indt,'Date']
    datee=pd.Series(datee)
    stat=pd.Series(station)
    dd=weath_out.iloc[(indt-(num_days*2)):indt,]
    ddd=dd[dd['Station']==station] # the 14 day (default) batch
    ss=pd.DataFrame()
    ss=pd.concat([ss,datee,stat],axis=1)
    for col in cols:
        summ=get_summary(ddd[col])
        ss=pd.concat([ss,summ],axis=1)
#     print(ss)
#     print('shape',ss.shape)
#     print('type',type(ss))
#     print('####')
    return(ss)

## script to create the engineered features for 14 days:

int_feat=['Tmax', 'Tmin', 'Tavg', 'DewPoint', 
          'WetBulb','Heat', 'Cool', 'PrecipTotal', 'StnPressure',
          'ResultSpeed','ResultDir', 'AvgSpeed', 'weather_type_Norm']
# tried to do it with apply and lambda, but there seems to be a bug. have to use for loop to run on the whole data set
# which will be very slow.
# this is how it should be done with apply:
#  rrr.iloc[34:46]=sptrainW2.iloc[0:3,:].apply(lambda x: fourteen(x['Date_of_collection'],x['Station'],
#                        ['ResultSpeed','Tmax']),axis=1)
import time

def make_features(clean_weath,feat_of_interest):   ## input clean weather data, and list of feature names to engineer
    tt=time.time()
    sso=pd.DataFrame()
    for i in range(clean_weath.shape[0]):
        row=fourteen(sptrainW2.loc[i,'Date_of_collection'],sptrainW2.loc[i,'Station'],feat_of_interest)
        sso=pd.concat([sso,row],axis=0)
        #if i==12:
         #   break
    sso.drop(['weather_type_Norm.std','weather_type_Norm.50%','weather_type_Norm.mean-median',
              'weather_type_Norm.outliers_low','weather_type_Norm.outliers_high'],1,inplace=True)
    sso.rename(columns={0:'Date_colect',1:'station'},inplace=True)
    ttt=time.time()
    print('time of running:',ttt-tt)
    print('new data shape: ',sso.shape)
    return(sso)

''' Let's run the function and get the engineered weather data '''
## warnning: this could take a while (approx 241 seconds)
## turn statement into True to run the function
if 1==0:
    eng_weath=make_features(weath_out,int_feat) 

#repository path: directory_path
if 1==0:
    eng_weath.to_csv(os.path.join(directory_path,'eng_weath.csv'))
    eng_weath=pd.read_csv("/Users/eran/Galvanize_more_repositories/WestNileVirus/eng_weath.csv")
#eng_weath.shape=(2944, 75)
#sptrainW2.shape=(10506, 35)

# merging spray&weather df (with weather on the day of) with engneered weather
sptrain_engW=pd.merge(sptrainW2,eng_weath,left_on=['Date_of_collection','Station'],right_on=['Date_colect','station'],indicator=True)
## Getting a massive df because of duplicates. getting rid of them:
# sptrain_engW.shape=(293598, 111)
yep=sptrain_engW.drop_duplicates()
#yep.shape=(9296, 111)

## DROP THE "OF THE DAY" WEATHER DATA FEATURES.
yep.drop(['Tmax', 'Tmin', 'Tavg', 'DewPoint',
       'WetBulb', 'Heat', 'Cool', 'PrecipTotal', 'StnPressure', 'SeaLevel',
       'ResultSpeed', 'ResultDir', 'AvgSpeed', 'weather_type_Norm'],1,inplace=True)

yep.drop(['Date_colect','Station','_merge'],1,inplace=True)

yep.drop(['Date_of_collection'],1,inplace=True)

#saving the data sets:
# sptrainW_day_of.csv - spray+train+weather data (the day of collection)
# sptrainW_14_days.csv - spray+train+weather data (summary of weather of 14 days before collection)
if 1==0:
    sptrainW2.drop(['Date_of_collection','Station'],1,inplace=True)
    sptrainW2.to_csv(os.path.join(directory_path,'sptrainW_day_of.csv'))
    yep.to_csv(os.path.join(directory_path,'sptrainW_14_days.csv'))
    trainBasic=train1[['Latitude','Longitude','AddressAccuracy','NumMosquitos','Species_CULEX PIPIENS','Species_CULEX PIPIENS/RESTUANS','Species_CULEX RESTUANS','Species_CULEX SALINARIUS','Species_CULEX TARSALIS','Species_CULEX TERRITANS','WnvPresent']]
    trainBasic.to_csv(os.path.join(directory_path,'train_baseline.csv'))
    