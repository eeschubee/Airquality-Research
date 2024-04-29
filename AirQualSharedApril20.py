import pandas as pd
import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime 
from datetime import timedelta 
import plotly.express as px 
import pytz
import latex
import matplotlib as mpl

#Animaiton to video/gif packages
import numpy as np
from scipy.spatial import Delaunay
import plotly.graph_objects as go
#import  moviepy.editor as mpy
import io 
from PIL import Image


#display function was giving me issues seems like it might be a function integrated in other python enviornments. 
from IPython.display import display



mpl.rcParams.update(mpl.rcParamsDefault)

#Specify file with location
df_main = pd.read_csv('outdoor_2021.csv')
df_main.shape
df_main.dtypes
df_main.describe()
df_main.head()



df_main['datetime_utc'] = pd.to_datetime(df_main['datetime_utc'])
df_main.isna().sum()
df_main.duplicated('location_label').sum()
df_main.loc[df_main.duplicated('location_label')]


# Identify unique location (sensor) labels 
temp = df_main.location_label.unique()
temp
temp.shape

#create a dictionary using unique location labels. This will be used to split the data frame 
df_dict = {sale_v: df_main[df_main['location_label'] == sale_v] for sale_v in df_main.location_label.unique()}
df_dict.keys()

#print(df_dict)

#Split Data Frames Using the Dictionary

# There doesnt seem to be a 'BLR_Water_Barn' or 'Blue_Lake_City' in the csv 
#df_blr = df_dict['BLR_Water_Barn']
#df_blc = df_dict['Blue_Lake_City']

df_butler = df_dict['Butler_Creek']
df_cecilville = df_dict['CARB_Cecilville']
df_happy_camp_cc = df_dict['SAFE_Happy_Camp_Community_Center']
df_sawyer = df_dict['CARB_Sawyers_Bar']
df_forks = df_dict['Forks_Of_Salmon']
df_kdnr_out = df_dict['KDNR_Outdoor']


# There doesnt seem to be a MKWC_Outdoor in the csv 
#df_mkwc = df_dict['MKWC_Outdoor']


df_somesbar = df_dict['SAFE_Somes_Bar']
df_sandybar = df_dict['SAFE_Sandy_Bar_Creek']
df_swillup = df_dict['SAFE_Swillup_Creek']

#These need to be changed depending on MAC or windows file organization is different.
#Also what is the Point of creating this CSV file? 
#df_swillup.to_csv('/Users/erikschubert/Desktop/AirQualityResearch/code/swillup_creek_aq_21.csv')

#What is the point of these?
df_happy_camp_cc.head()
df_butler.head()

#This DF doesn not exsists. Looks at line 69.
#df_mkwc.head()

#Create sensor location names for later use in graphs 
sensor_location_names = ["Butler_Creek","CARB_Cecilville",
"SAFE_Happy_Camp_Community_Center","CARB_Sawyer","Forks_Of_Salmon","Orleans_KDNR_Outdoor","SAFE_Somes_Bar","SAFE_Sandy_Bar_Creek","SAFE_Swillup_Creek"]

#used as a check 

sensor_location_names[0]

#create a list of the needed data frames from the ones selected 
sensor_dfs = (df_swillup, df_butler,df_cecilville,df_happy_camp_cc,df_sawyer,df_forks,df_kdnr_out,df_somesbar,df_sandybar)  
   
# this is another visual check    

'''
for df in sensor_dfs:
   display(df.describe())
''' 

'''
# another visual check to find begin and end of logged date time
for i,df in enumerate(sensor_dfs):
   display(sensor_location_names[i])
   display(df.datetime_utc.min(),df.datetime_utc.max())
   display(df.shape)
'''

# Re-index with 15 minute intervals to capture missing data 
from datetime import date
#defining the function for subtracting 
def get_difference(startdate, enddate):
    diff = enddate - startdate
    return diff.days
#initializing dates
# this was the first and last date of the data set. -- hard coded 
startdate = date(2021, 7, 1)
enddate = date(2021, 12, 31)
#storing the result and calling the function
days = get_difference(startdate, enddate)+1
#displaying the result

print(f'Difference is {days} days')
print(f'total 15 minute intervals of: {days*24*4}')

#the output from days*24*4 is used to create a time series that starts juy 1 and is every 15 min 
#reindex - generate 15 minute interval time index

sensor_list_df =  (df_butler,df_cecilville,df_happy_camp_cc,df_sawyer,df_forks,df_kdnr_out,df_somesbar,df_sandybar,df_swillup)

date_index2 = pd.date_range('2021/07/01', periods=17664, freq='15T')

sensor_list_gf = (df_butler,df_cecilville,df_happy_camp_cc,df_sawyer,df_forks,df_kdnr_out,df_somesbar,df_sandybar,df_swillup)

name_label = ["Butler_Creek","CARB_Cecilville",
"SAFE_Happy_Camp_Community_Center","CARB_Sawyer","Forks_Of_Salmon","Orleans_KDNR_Outdoor","SAFE_Somes_Bar","SAFE_Sandy_Bar_Creek","SAFE_Swillup_Creek"]
#this short list was for the poster.
short_list = (df_butler, df_forks,df_kdnr_out,df_somesbar)

short_name = ["Butler Creek","Forks Of Salmon","Orleans","Somes_Bar" ]

for df in sensor_list_df:
     # setting first name as index column
     # indexing by time here. 
     #re indexing by the 15 min time series helps identify missing time stamps. 
   df.set_index(["datetime_utc"], inplace = True,append = False, drop = True)
   df = df.reindex(date_index2)
   # this is a check to see how many of the 15 min intervals have N/A data.
   checkna = df['longitude'].isna()
   #returns number on none values. 
  # print(checkna.value_counts())


#    
for df in sensor_list_df:
   #these colums already exsisted for 2021 because it was pre cleaned 2022 dose this manually with a data cleaning code. 
     # setting first name as index column
   df['check'] = (df['ab_deviation_absolute'] > 5.0) & (df['ab_deviation_fraction'] > 0.7)
   # createsa  col in df called check that either T/F

   #display(df.head())
  # display(df['check'].value_counts())
   


#set average is NaN if check if True
for df in sensor_list_df:
   #checking the location where check is True

   # this is there to check if ab deviation ok col yielding same result as the check coll created in previous for loop. 
   display(df['ab_deviation_OK'].loc[df[df['check']==True].index.values])
   display(df['pm25_avg'].loc[df[df['check']==True].index.values])
   #No need for the code below because the data already had deviation check
   #If not, uncomment the code below
   #df.loc[df.check == True,'pm25_avg'] = np.nan

#EPA correction equation. The PM values were CF Atm. Need to change to PM CF= 1 for better accuracy
# No need to update the PM source value


for df in sensor_list_df:
   df['corrected'] = 0.534*(1*(df['pm25_avg']))-0.0844*df['humidity']+5.604


'''
for df in sensor_list_df:
   display(df.head())
'''
 
#scatter plot between the average and corrected pm2.5 concentration
for i, df in enumerate(sensor_list_df):
   #dataframe with negative corrections
   df_neg_corrected=df[(df['corrected'] < 0)]
   df_neg_corrected.plot.scatter('pm25_avg','corrected', title=name_label[i])
# the previous correction created negative values for very low PM 25


for i, df in enumerate(sensor_list_df):
   #Changing the negative corrected values to zero
   df.loc[df.corrected < 0,'corrected'] = 0
   print(name_label[i],'Negatives Found:')
   print(df['corrected'].where(df['corrected'] < 0).count())
   display(df.head())
   #Change index time from utc to local time (PST)

   #THIS PART OF CODE NEEDS TO BE LOOKED AT TO FIX BUG
   df.index = df.index.tz_localize('UTC').tz_convert('US/Pacific')
   
## Orleans event identification troublshooting 


'''
print(df_kdnr_out.head)
x = df_kdnr_out.loc['2021-09-06 18:30:00-07:00':'2021-09-09 18:30:00-07:00']
y = df_kdnr_out['corrected'].loc['2021-09-06 18:30:00-07:00':'2021-09-09 18:30:00-07:00']
# How many of the data points were above concentration of 150 
temp = (df_kdnr_out['corrected'].loc['2021-09-06 18:30:00-07:00':'2021-09-09 18:30:00-07:00']>150)
print(temp.shape)


#Using Valerie's Template (For LSAMP and UCLA Poster) 

#

i =0 
plt.figure(figsize=(15,10))
plt.title('July through December, 2021.  3-hour Avg PM2.5 Conc',fontsize = 24,weight = 'bold')
plt.xlabel("Date",fontsize=18 ,weight = 'bold')
plt.ylabel('PM2.5 micro-grams/${m^3}$',fontsize=18,weight = 'bold')

   #plt.ylabel(r'PM2.5 micro-grams/$\boldsymbol{m^3}$',fontsize=18,weight = 'bold')
plt.xticks(size = 5,rotation=45,fontsize=20) # This was important to limit the number of days displayed on the x axis
plt.yticks(size = 5,fontsize=20)
plt.tick_params('both', length=20, width=2, which='major')
df_temp = pd.DataFrame()
for dfgraph in short_list:
   # resample evry 3H creates new moving ave 
   df_temp['corrected'] = dfgraph['corrected'].resample('3H').mean()
   display(df_temp)
    # Plot the data with Matplotlib Plt
    # this retruns time 
   x = df_temp['corrected'].loc['2021-01-01':'2021-12-31'].index
   # this returns coresponding concentration at that time.
   y = df_temp['corrected'].loc['2021-01-01':'2021-12-31']
   plt.plot(x,y,label=short_name[i])
   #plt.title(sensor_location_names[i])
   i = i +1 

plt.legend(loc='upper right')
plt.rc('legend', fontsize = 20)
plt.show()

'''

months_for_15_min_avg = [
   { 
   'month_name':'July',
   'start_date':'2021/7/1',
   'end_date': '2021/7/31'
   }, 
   { 
   'month_name':'August',
   'start_date':'2021/8/1',
   'end_date': '2021/8/31'
   }, 
   { 
   'month_name':'September',
   'start_date':'2021/9/1',
   'end_date': '2021/9/30'
   }, 
   { 
   'month_name':'October',
   'start_date':'2021/10/1',
   'end_date': '2021/10/31'
   }, 
   { 
   'month_name':'November',
   'start_date':'2021/11/1',
   'end_date': '2021/11/30'
   }, 
   { 
   'month_name':'December',
   'start_date':'2021/12/1',
   'end_date': '2021/12/31'
   }, 
   # these rest of these could prob just be left out since they look like they were checks designed to fix september but september only has 30 days which was the orgininal prob. 
   { 
   'month_name':'September 1-10',
   'start_date':'2021/9/1',
   'end_date': '2021/9/10'
   }, 
    { 
   'month_name':'September 11-20',
   'start_date':'2021/9/11',
   'end_date': '2021/9/20'
   }, 
    { 
   'month_name':'September 21-30',
   'start_date':'2021/9/21',
   'end_date': '2021/9/30'
   }
]

# this is mostly for visualization to see change in the short term.

def min_avg_PM25_months(start_date,end_date,month_name,sensor_list_gf):
   i = 0  
#plt.figure(figsize=(300,240))

   plt.title(month_name + '15 min Avg PM2.5 Conc, ug/m3')
   plt.xlabel("Date")
   plt.ylabel("PM2.5 micro-grams/m3")
   plt.xticks(rotation=45) # This was important to limit the number of days displayed on the x axis

   for df in sensor_list_gf:
      df.head()
      # Plot the data with Matplotlib Plt
      x = df['pm25_epa'].sort_index().loc[start_date:end_date].index
      y = df['pm25_epa'].sort_index().loc[start_date:end_date]
      plt.scatter(x,y,label=name_label[i])
      #plt.title(sensor_location_names[i])
      i = i +1 
   plt.legend()
   plt.show()
   return
'''
for i in months_for_15_min_avg:
   min_avg_PM25_months(i['start_date'],i['end_date'],i['month_name'],sensor_list_gf) 
'''
   # ********************************************************************************************************************
   # ********************************************************************************************************************
   # ********************************************************************************************************************
   # **** extreme event identification
   # ********************************************************************************************************************
   # ********************************************************************************************************************
   # ********************************************************************************************************************
   
def convert_timedelta(duration):
   days, seconds = duration.days, duration.seconds
   hours = days * 24 + seconds // 3600
   minutes = (seconds % 3600) // 60
   seconds = (seconds % 60)
   return hours, minutes, seconds

Events_and_concentrations = [
   { 
   'Event_level': '3 (55.5)',
   'concentration_param': 55.5
   }, 
   { 
   'Event_level': '2 (150.5)',
   'concentration_param': 150.5
   }, 
   { 
   'Event_level': '1 (250.5)',
   'concentration_param': 250.5
   }
   ]

def Tracking_Events_Greater_Than_Given_Perameters(concentration_param,Event_level,sensor_list_gf):
   print(r'Events greater than '+str(concentration_param)+' micro-grams/$m^3$ - 6, 30 minute intervals out of 48 30 minute intervals (1 day)')  
   dfs_leveltwo = {}
   #colors = mpl.colormaps.get_cmap('viridis').resampled(20).colors
   for k,df in enumerate(sensor_list_gf):
      # Extreme Event Identification 

      
      plt.figure(figsize=(10,8))
      plt.title('Level ' + Event_level + ' Events:  ' + sensor_location_names[k],fontsize = 24,weight = 'bold')
      plt.xlabel('Date',fontsize=18,weight = 'bold')
      plt.ylabel('PM2.5 micro-grams/${m^3}$',fontsize=18,weight = 'bold')
      plt.xticks(rotation=45,fontsize=20) # This was important to limit the number of days displayed on the x axis
      plt.yticks(fontsize=20)
      df.head()
      df_resampled = pd.DataFrame()
      df_resampled['pm25_epa'] = df['pm25_epa'].resample('30min').mean()

      # use a for loop to apply the corrected equation to the pm25_epa values and create a new column called 'corrected'
      df_resampled['corrected'] = 0.534*(1*(df_resampled['pm25_epa']))-0.0844*df['humidity'].resample('30min').mean()+5.604
      print(df_resampled.head())  # Print the first few rows of the resampled dataframe to inspect its structure

      # created new col called time stamp from the index.
      df_resampled['time_stamp'] = df_resampled.index
      # reset the index so that the index stats from 0. No longer a time series. Int index starting from 0.
      df_resampled.reset_index(drop=True, inplace=True)
      # visualize 

      # Counting 30 min concentrations > 55.5 if present 3 hours out of 48 * 30 min intervals. 
      # we delcare an vents data frame that has these cols 
      events=pd.DataFrame(columns = ['index','time_stamp','third','fourth'])
      # initialize first row to be - 50
      events.loc[0]=-50

      i =0 
      #for i,f in enumerate(df_nan['pm2.5_cf_1_a']):

      for i in df_resampled.index:
         
         #EDIT 1
         # if i < 23: 
            # j = i + 24
            # a = 0 
         # its looking forword to determine
         # elif len(df_resampled)-24 -1 < i:
            # j = len(df_resampled)-1 
            # a = i - 24
         # else: 
            # a = i - 24
            # j = i + 23
            
         # window of 48 30 min intervals
         # if the start index is less than 0, set it to 0
         start_index = max(0, i - 24)
         # if the end index is greater than the length of the data frame, set it to the last index
         end_index = min(len(df_resampled) - 1, i + 23)

         #Changes made By jordan.
         # check if the start index is greater than 6 intervals greater than 0

         if start_index > 6:
                # if it is, set the start index to 6 intervals before the current index
                start_index -= 6
         # check if the end index is less than 6 intervals less than the last index
         if end_index < len(df_resampled) - 6:
                    # if it is, set the end index to 6 intervals after the current index
                    end_index += 6
         
         #original
         '''
         a=i
         j=i+47
         '''
         ## Event count is now looking at the correvted values instead of the avg. 

         # this is the number of values that are greater than the concentration param
         df_resampled.loc[i,'event_count'] = (df_resampled['corrected'].loc[start_index:end_index + 1]>concentration_param).sum()
         
         
         
      df_resampled.describe()
      df_resampled
      df_resampled['event_count'].describe()
      # looks to see if greater t
      countlist = (df_resampled[df_resampled['event_count']>6].index.values)

      print(name_label[k])
      print(countlist)

      # for i in countlist:
      #    print({df_resampled['time_stamp'].loc[i]})

      j = 0
      # Now separate events by 48 - 30 minute intervals
      for i,f in enumerate(countlist):
         ''
         if i ==0:
            events.loc[j,'index'] = countlist[i] 
            events.loc[j,'time_stamp'] = df_resampled['time_stamp'].loc[countlist[i]]
            j = j+1
            
         if countlist[i]> (events['index'].loc[j-1] + 48):
            events.loc[j,'index'] = countlist[i] 
            events.loc[j,'time_stamp'] = df_resampled['time_stamp'].loc[countlist[i]]
            j = j+1
            
      # print(events['index'])
      # for i,f in enumerate(events):
         # print(i,f['index'].loc[i],f['time_stamp'].loc[i])
      display(events)
      
      # for i,f in enumerate(events):
      #    print({df_resampled['time_stamp'].loc[f]})
      consolidated_events=pd.DataFrame(columns = ['index','time_start','time_end','days'])
      
      # Identify and isolate consecutive events 
    
      j = 0 
      for p,f in enumerate(events['index']):
         #print(p)

         ##
         if p ==0:
            consolidated_events.loc[j,'index'] = events['index'].loc[p] 
            consolidated_events.loc[j,'time_start'] = events['time_stamp'].loc[p]
            j = j+1
            #print(p,j)
            continue
            
         diff = events['time_stamp'].loc[p] -  events['time_stamp'].loc[p-1]  
         hours, minutes, seconds = convert_timedelta(diff)
         total_hours = hours + minutes/60 + seconds/3600
         
         if total_hours > 25:
            consolidated_events.loc[j,'index'] = events['index'].loc[p] 
            consolidated_events.loc[j,'time_start'] = events['time_stamp'].loc[p]
            j = j+1
            #print(p,j,total_hours)
         else:


            consolidated_events.loc[j-1,'time_end'] = events['time_stamp'].loc[p]
            #print(p,j, total_hours)
      # Convert the datetime columns to datetime64[ns]


      consolidated_events['time_start'] = pd.to_datetime(consolidated_events['time_start'])
      consolidated_events['time_end'] = pd.to_datetime(consolidated_events['time_end'])

      #this For loop could be causing some errors.
      for i in consolidated_events.index:
         
      
         if  (consolidated_events['time_end'].loc[i] is pd.NaT):
            consolidated_events['time_end'].loc[i] = (consolidated_events['time_start'].loc[i]) + (timedelta(days=1)) 


         diff = consolidated_events['time_end'].loc[i] -  consolidated_events['time_start'].loc[i]  
         hours, minutes, seconds = convert_timedelta(diff)
         consolidated_events['days'].loc[i]= (hours + minutes/60 + seconds/3600)/24
     # my_dict['name']='Nick'
      #display(df_resampled.head())

      dfs_leveltwo[str(name_label[k])+ '_leveltwo'] =  consolidated_events
      dfevent = dfs_leveltwo[str(name_label[k])+ '_leveltwo']

      #display(dfevent.head())
      
      # Added some commas to because to the end of the three lines below.
      colors = ['red','purple','green','yellow','orange','brown','black','violet','slategrey','khaki',
                'gray','silver','whitesmoke','rosybrown','firebrick','darksalmon','sienna','sandybrown',
                'olivedrab','chartreuse','palegreen','darkgreen','seagreen','navy','peachpuff','darkorange',
                'navajowhite','darkgoldenrod','lemonchiffon','mediumseagreen','cadetblue','skyblue','dodgerblue','slategray']

      '''

      print(df_resampled.head())
      print(df_resampled['corrected'].loc[0])
      print(df_resampled['corrected'].loc[6])
      '''

      for i in consolidated_events.index:


         
         sz = len(consolidated_events.index)-1


         #Issues where arrising for empty consolidated events. 
         if sz <=0  :
            continue



         # Plot the data with Matplotlib Plt
         # plot start to end first 
         
         if i==0:
            #endindex = consolidated_events['index'].loc[i] + int((consolidated_events['days'].loc[i])*48)

            endindex = consolidated_events['index'].loc[sz] + int((consolidated_events['days'].loc[sz])*48) + 48 #Added one extra day at the end

            x = df_resampled['time_stamp'].loc[consolidated_events['index'].loc[i]:endindex]
            y = df_resampled['corrected'].loc[consolidated_events['index'].loc[i]:endindex]
            plt.plot(x,y,label='PM 2.5')

            x = [df_resampled['time_stamp'].loc[consolidated_events['index'].loc[i]], df_resampled['time_stamp'].loc[endindex]]
            #Below is the threshold line. 
            y = [concentration_param,concentration_param]

            plt.plot(x,y,linewidth=1, linestyle='dashed',label = 'Level '+ Event_level +' Criteria Concentration')

         

         ## this may be casuing issues with our graph because the start of an event is not defined by the first event that occurs when 

         

         start_index = consolidated_events['index'].loc[i] 

         end_index = start_index + int((consolidated_events['days'].loc[i])*48) #48 because 30 min intervals 

         #here we should iterate through d_f starting at the indes 24 * 30 min intervals before the event start and find the first event that exceeds the concentration value and set thaht as start 
         start1 = max(0, start_index - 24)
         end1 = min(len(df_resampled) - 1, end_index + 23)

         for a in range(24):
            
            if df_resampled['corrected'].loc[start1] >= concentration_param: 
               break
            start1 += 1

         for b in range(23):
            
            if df_resampled['corrected'].loc[end1] >= concentration_param:
               
               break
            end1 -= 1


         begindex = start1
         # if the end index is greater than the length of the data frame, set it to the last index
         endindex = end1

         print(begindex)
         print(endindex)

         print(df_resampled['time_stamp'].loc[begindex])
         print(df_resampled['time_stamp'].loc[endindex])

         
         
         x = [df_resampled['time_stamp'].loc[begindex],df_resampled['time_stamp'].loc[begindex]]
         y = [0,800]   
         plt.plot(x,y,linewidth=1,color = colors[i])
         #color = (0, i / 20.0, 0, 1)
         #color=plt.cm.RdYlBu(i)
         
         x = [df_resampled['time_stamp'].loc[begindex],df_resampled['time_stamp'].loc[endindex]]
         y= [800,800]
         plt.plot(x,y,linewidth=1,color =  colors[i])
         
         x = [df_resampled['time_stamp'].loc[endindex],df_resampled['time_stamp'].loc[endindex]]
         y = [0,800]
         # Added This Label

         #More readable. 
         lable_start_date = consolidated_events['time_start'].loc[i].strftime("%b %d, %Y, %I:%M %p")
         lable_end_date =  consolidated_events['time_end'].loc[i].strftime("%b %d, %Y, %I:%M %p")
         duration = "{:.2f}".format(consolidated_events['days'].loc[i])

         labels = f"Evnt {i + 1}, Dur {duration}, {lable_start_date} : {lable_end_date}"

         plt.plot(x,y,linewidth=1,color =  colors[i],label=labels)

      plt.legend(loc='upper right')
      plt.rc('legend', fontsize = 14)


      table_name = f"{Event_level}_{concentration_param}_{name_label[k]}.png"

      plt.savefig(table_name)
      plt.close()
      #plt.show()

      

      for i,dataf in enumerate(dfevent['index']):
            dfevent['slope_first'] = -10.10
            dfevent['slope_second'] = -10.10
            dfevent['slope_third'] = -10.10
            dfevent['slope_fourth'] = -10.10
            
            if dfevent['index'].loc[i]<= 6:
               dfevent.loc[i] = np.nan
               continue
               
               #print(slope)
            if dfevent['index'].loc[i]<=12:
               dfevent.loc[i,'slope_first'] = (df_resampled['corrected'].loc[6] -df_resampled['corrected'].loc[0])/((6-0)*30)
               continue
            if dfevent['index'].loc[i]<=18:
               dfevent.loc[i,'slope_first'] = ((df_resampled['corrected'].loc[6] -df_resampled['corrected'].loc[0])/((6-0)*30))
               dfevent.loc[i,'slope_second'] = (df_resampled['corrected'].loc[12] -df_resampled['corrected'].loc[0])/((12-0)*30)
               continue
            if dfevent['index'].loc[i]<=24:
               dfevent.loc[i,'slope_third'] = (df_resampled['corrected'].loc[18] -df_resampled['corrected'].loc[0])/((18-0)*30)
               dfevent.loc[i,'slope_second'] = (df_resampled['corrected'].loc[12] -df_resampled['corrected'].loc[0])/((12-0)*30)
               dfevent.loc[i,'slope_first'] = (df_resampled['corrected'].loc[6] -df_resampled['corrected'].loc[0])/((6-0)*30)
               continue
            if dfevent['index'].loc[i]>24:
               ind = dfevent['index'].loc[i]
               dfevent.loc[i,'slope_fourth'] = (df_resampled['corrected'].loc[ind] -df_resampled['corrected'].loc[ind-24])/((24-0)*30)
               dfevent.loc[i,'slope_third'] = (df_resampled['corrected'].loc[ind] -df_resampled['corrected'].loc[ind-18])/((18-0)*30)
               dfevent.loc[i,'slope_second'] = (df_resampled['corrected'].loc[ind] -df_resampled['corrected'].loc[ind-12])/((12-0)*30)
               dfevent.loc[i,'slope_first'] = (df_resampled['corrected'].loc[ind] -df_resampled['corrected'].loc[ind-6])/((6-0)*30)

      print(name_label[k])
      display(dfevent)
      dfs_leveltwo[str(name_label[k])+ '_leveltwo'] = dfevent.copy('deep')
      
      #= dfs_leveltwo.append{str(name_label[k])+ '_leveltwo': consolidated_events}
   #print(dfs_leveltwo['SAFE_Swillup_Creek_leveltwo'])
   dfs_leveltwo.keys()
   print(dfs_leveltwo['SAFE_Swillup_Creek_leveltwo'])
   display(dfevent)
   return

# Calling the func for graphing event concentrations using a list of the level and associated concentration
for i in Events_and_concentrations:
   Tracking_Events_Greater_Than_Given_Perameters(i['concentration_param'],i['Event_level'],sensor_list_gf)
   
quit()

 # Header needs to have a var. line 720 
 # line 738 air qual paramm
 # needs sensor_list_gf
 #needs pd

'''

 ###Start of block 
print(r'Eents greater than 55.5 micro-grams/$m^3$ - 6, 30 minute intervals out of 48 30 minute intervals (1 day)')  
dfs_leveltwo = {}
#colors = mpl.colormaps.get_cmap('viridis').resampled(20).colors
for k,df in enumerate(sensor_list_gf):
   # Extreme Event Identification 

   
   plt.figure(figsize=(10,8))
   plt.title('Level 3 Events:  ' + sensor_location_names[k],fontsize = 24,weight = 'bold')
   plt.xlabel('Date',fontsize=18,weight = 'bold')
   plt.ylabel('PM2.5 micro-grams/${m^3}$',fontsize=18,weight = 'bold')
   plt.xticks(rotation=45,fontsize=20) # This was important to limit the number of days displayed on the x axis
   plt.yticks(fontsize=20)
   df.head()
   df_resampled = df.resample('30T').mean()
   df_resampled['time_stamp'] = df_resampled.index
   df_resampled.reset_index(drop=True, inplace=True)
   df_resampled.head()
   plt.show()
   

   # Counting 30 min concentrations > 55.5 if present 3 out of 48 hours. 
   events=pd.DataFrame(columns = ['index','time_stamp','third','fourth'])
   events.loc[0]=-50

   i =0 
   #for i,f in enumerate(df_nan['pm2.5_cf_1_a']):
   for i in df_resampled.index:
      j=i+47
      df_resampled.loc[i,'event_count'] = (df_resampled['pm25_epa'].loc[i:j]>55.5).sum()
      
   df_resampled.describe()
   df_resampled
   df_resampled['event_count'].describe()
   countlist = (df_resampled[df_resampled['event_count']>6].index.values)
   print(name_label[k])
   print(countlist)
   # for i in countlist:
   #    print({df_resampled['time_stamp'].loc[i]})

   j = 0
   # Now separate events by 48 - 30 minute intervals
   for i,f in enumerate(countlist):
      if i ==0:
         events.loc[j,'index'] = countlist[i] 
         events.loc[j,'time_stamp'] = df_resampled['time_stamp'].loc[countlist[i]]
         j = j+1
         
      if countlist[i]> (events['index'].loc[j-1] + 48):
         events.loc[j,'index'] = countlist[i] 
         events.loc[j,'time_stamp'] = df_resampled['time_stamp'].loc[countlist[i]]
         j = j+1
         
   # print(events['index'])
   # for i,f in enumerate(events):
      # print(i,f['index'].loc[i],f['time_stamp'].loc[i])
   display(events)
   
   # for i,f in enumerate(events):
   #    print({df_resampled['time_stamp'].loc[f]})
   consolidated_events=pd.DataFrame(columns = ['index','time_start','time_end','days'])
   
   # Identify and isolate consecutive events 


 
 
   j = 0 
   for p,f in enumerate(events['index']):
      #print(p)
      if p ==0:
         consolidated_events.loc[j,'index'] = events['index'].loc[p] 
         consolidated_events.loc[j,'time_start'] = events['time_stamp'].loc[p]
         j = j+1
         #print(p,j)
         continue
         
      diff = events['time_stamp'].loc[p] -  events['time_stamp'].loc[p-1]  
      hours, minutes, seconds = convert_timedelta(diff)
      total_hours = hours + minutes/60 + seconds/3600
      
      if total_hours > 25:
         consolidated_events.loc[j,'index'] = events['index'].loc[p] 
         consolidated_events.loc[j,'time_start'] = events['time_stamp'].loc[p]
         j = j+1
         #print(p,j,total_hours)
      else:
         consolidated_events.loc[j-1,'time_end'] = events['time_stamp'].loc[p]
         #print(p,j, total_hours)
   consolidated_events['time_start'] = consolidated_events['time_start'].astype('datetime64')
   consolidated_events['time_end'] = consolidated_events['time_end'].astype('datetime64')

   for i in consolidated_events.index:
      
   
      if  (consolidated_events['time_end'].loc[i] is pd.NaT):
         consolidated_events['time_end'].loc[i] = (consolidated_events['time_start'].loc[i]) + (timedelta(days=1)) 
      diff = consolidated_events['time_end'].loc[i] -  consolidated_events['time_start'].loc[i]  
      hours, minutes, seconds = convert_timedelta(diff)
      consolidated_events['days'].loc[i]= (hours + minutes/60 + seconds/3600)/24
  # my_dict['name']='Nick'
   display(df_resampled.head())
   dfs_leveltwo[str(name_label[k])+ '_leveltwo'] =  consolidated_events
   dfevent = dfs_leveltwo[str(name_label[k])+ '_leveltwo']
   display(dfevent.head())
   
   # Added some commas to because to the end of the three lines below.
   colors = ['red','purple','green','yellow','orange','brown','black','violet','slategrey','khaki',
             'gray','silver','whitesmoke','rosybrown','firebrick','darksalmon','sienna','sandybrown',
             'olivedrab','chartreuse','palegreen','darkgreen','seagreen','navy','peachpuff','darkorange',
             'navajowhite','darkgoldenrod','lemonchiffon','mediumseagreen','cadetblue','skyblue','dodgerblue','slategray']
   
   for i in consolidated_events.index:
      
      sz = len(consolidated_events.index)-1
      # Plot the data with Matplotlib Plt
      # plot start to end first 
      
      if i==0:
         #endindex = consolidated_events['index'].loc[i] + int((consolidated_events['days'].loc[i])*48)
         endindex = consolidated_events['index'].loc[sz] + int((consolidated_events['days'].loc[sz])*48) + 48 #Added one extra day at the end
         x = df_resampled['time_stamp'].loc[consolidated_events['index'].loc[i]:endindex]
         y = df_resampled['corrected'].loc[consolidated_events['index'].loc[i]:endindex]
         plt.plot(x,y,label='PM 2.5')
         x = [df_resampled['time_stamp'].loc[consolidated_events['index'].loc[i]], df_resampled['time_stamp'].loc[endindex]]
         y = [150.5,150.5]
         plt.plot(x,y,linewidth=1, linestyle='dashed',label = 'Level 2 Criteria Concentration')

      begindex = consolidated_events['index'].loc[i]
      endindex = begindex + int((consolidated_events['days'].loc[i])*48)
      
      x = [df_resampled['time_stamp'].loc[begindex],df_resampled['time_stamp'].loc[begindex]]
      y = [0,800]   
      plt.plot(x,y,linewidth=1,color = colors[i])
      #color = (0, i / 20.0, 0, 1)
      #color=plt.cm.RdYlBu(i)
      
      x= [df_resampled['time_stamp'].loc[begindex],df_resampled['time_stamp'].loc[endindex]]
      y= [800,800]
      plt.plot(x,y,linewidth=1,color =  colors[i])

      
      
      x = [df_resampled['time_stamp'].loc[endindex],df_resampled['time_stamp'].loc[endindex]]
      y = [0,800]
      plt.plot(x,y,linewidth=1,color =  colors[i],label=f'Event {i +1}')

   plt.legend(loc='upper right')
   plt.rc('legend', fontsize = 14)
   plt.show()
   
   
   
   for i,dataf in enumerate(dfevent['index']):
         dfevent['slope_first'] = -10.10
         dfevent['slope_second'] = -10.10
         dfevent['slope_third'] = -10.10
         dfevent['slope_fourth'] = -10.10
         
         if dfevent['index'].loc[i]<= 6:
            dfevent.loc[i] = np.nan
            continue
            
            #print(slope)
         if dfevent['index'].loc[i]<=12:
            dfevent.loc[i,'slope_first'] = (df_resampled['corrected'].loc[6] -df_resampled['corrected'].loc[0])/((6-0)*30)
            continue
         if dfevent['index'].loc[i]<=18:
            dfevent.loc[i,'slope_first'] = ((df_resampled['corrected'].loc[6] -df_resampled['corrected'].loc[0])/((6-0)*30))
            dfevent.loc[i,'slope_second'] = (df_resampled['corrected'].loc[12] -df_resampled['corrected'].loc[0])/((12-0)*30)
            continue
         if dfevent['index'].loc[i]<=24:
            dfevent.loc[i,'slope_third'] = (df_resampled['corrected'].loc[18] -df_resampled['corrected'].loc[0])/((18-0)*30)
            dfevent.loc[i,'slope_second'] = (df_resampled['corrected'].loc[12] -df_resampled['corrected'].loc[0])/((12-0)*30)
            dfevent.loc[i,'slope_first'] = (df_resampled['corrected'].loc[6] -df_resampled['corrected'].loc[0])/((6-0)*30)
            continue
         if dfevent['index'].loc[i]>24:
            ind = dfevent['index'].loc[i]
            dfevent.loc[i,'slope_fourth'] = (df_resampled['corrected'].loc[ind] -df_resampled['corrected'].loc[ind-24])/((24-0)*30)
            dfevent.loc[i,'slope_third'] = (df_resampled['corrected'].loc[ind] -df_resampled['corrected'].loc[ind-18])/((18-0)*30)
            dfevent.loc[i,'slope_second'] = (df_resampled['corrected'].loc[ind] -df_resampled['corrected'].loc[ind-12])/((12-0)*30)
            dfevent.loc[i,'slope_first'] = (df_resampled['corrected'].loc[ind] -df_resampled['corrected'].loc[ind-6])/((6-0)*30)
   print(name_label[k])
   display(dfevent)
   dfs_leveltwo[str(name_label[k])+ '_leveltwo'] = dfevent.copy('deep')
            
   #= dfs_leveltwo.append{str(name_label[k])+ '_leveltwo': consolidated_events}
#print(dfs_leveltwo['SAFE_Swillup_Creek_leveltwo'])
dfs_leveltwo.keys()
print(dfs_leveltwo['SAFE_Swillup_Creek_leveltwo'])
display(dfevent)
#end block 
 #****************************************************************#
 #****************************************************************#
 #****************************************************************#
 #****************************************************************#  
quit()  

   
print(r'Eents greater than 150.5 micro-grams/$m^3$ - 6, 30 minute intervals out of 48 30 minute intervals (1 day)')  
dfs_leveltwo = {}
#colors = mpl.colormaps.get_cmap('viridis').resampled(20).colors
for k,df in enumerate(sensor_list_gf):
   # Extreme Event Identification 

   
   plt.figure(figsize=(10,8))
   plt.title('Level 2 Events:  ' + sensor_location_names[k],fontsize = 24,weight = 'bold')
   plt.xlabel('Date',fontsize=18,weight = 'bold')
   plt.ylabel('PM2.5 micro-grams/${m^3}$',fontsize=18,weight = 'bold')
   plt.xticks(rotation=45,fontsize=20) # This was important to limit the number of days displayed on the x axis
   plt.yticks(fontsize=20)
   df.head()
   df_resampled = df.resample('30T').mean()
   df_resampled['time_stamp'] = df_resampled.index
   df_resampled.reset_index(drop=True, inplace=True)
   df_resampled.head()
   plt.show()
   

   # Counting 30 min concentrations > 150.5 if present 3 out of 48 hours. 
   events=pd.DataFrame(columns = ['index','time_stamp','third','fourth'])
   events.loc[0]=-50

   i =0 
   #for i,f in enumerate(df_nan['pm2.5_cf_1_a']):
   for i in df_resampled.index:
      j=i+47
      df_resampled.loc[i,'event_count'] = (df_resampled['pm25_epa'].loc[i:j]>150.5).sum()
      
   df_resampled.describe()
   df_resampled
   df_resampled['event_count'].describe()
   countlist = (df_resampled[df_resampled['event_count']>6].index.values)
   print(name_label[k])
   print(countlist)
   # for i in countlist:
   #    print({df_resampled['time_stamp'].loc[i]})

   j = 0
   # Now separate events by 48 - 30 minute intervals
   for i,f in enumerate(countlist):
      if i ==0:
         events.loc[j,'index'] = countlist[i] 
         events.loc[j,'time_stamp'] = df_resampled['time_stamp'].loc[countlist[i]]
         j = j+1
         
      if countlist[i]> (events['index'].loc[j-1] + 48):
         events.loc[j,'index'] = countlist[i] 
         events.loc[j,'time_stamp'] = df_resampled['time_stamp'].loc[countlist[i]]
         j = j+1
         
   # print(events['index'])
   # for i,f in enumerate(events):
      # print(i,f['index'].loc[i],f['time_stamp'].loc[i])
   display(events)
   
   # for i,f in enumerate(events):
   #    print({df_resampled['time_stamp'].loc[f]})
   consolidated_events=pd.DataFrame(columns = ['index','time_start','time_end','days'])
   
   # Identify and isolate consecutive events 


 
 
   j = 0 
   for p,f in enumerate(events['index']):
      #print(p)
      if p ==0:
         consolidated_events.loc[j,'index'] = events['index'].loc[p] 
         consolidated_events.loc[j,'time_start'] = events['time_stamp'].loc[p]
         j = j+1
         #print(p,j)
         continue
         
      diff = events['time_stamp'].loc[p] -  events['time_stamp'].loc[p-1]  
      hours, minutes, seconds = convert_timedelta(diff)
      total_hours = hours + minutes/60 + seconds/3600
      
      if total_hours > 25:
         consolidated_events.loc[j,'index'] = events['index'].loc[p] 
         consolidated_events.loc[j,'time_start'] = events['time_stamp'].loc[p]
         j = j+1
         #print(p,j,total_hours)
      else:
         consolidated_events.loc[j-1,'time_end'] = events['time_stamp'].loc[p]
         #print(p,j, total_hours)
   consolidated_events['time_start'] = consolidated_events['time_start'].astype('datetime64')
   consolidated_events['time_end'] = consolidated_events['time_end'].astype('datetime64')

   for i in consolidated_events.index:
      
   
      if  (consolidated_events['time_end'].loc[i] is pd.NaT):
         consolidated_events['time_end'].loc[i] = (consolidated_events['time_start'].loc[i]) + (timedelta(days=1)) 
      diff = consolidated_events['time_end'].loc[i] -  consolidated_events['time_start'].loc[i]  
      hours, minutes, seconds = convert_timedelta(diff)
      consolidated_events['days'].loc[i]= (hours + minutes/60 + seconds/3600)/24
  # my_dict['name']='Nick'
   display(df_resampled.head())
   dfs_leveltwo[str(name_label[k])+ '_leveltwo'] =  consolidated_events
   dfevent = dfs_leveltwo[str(name_label[k])+ '_leveltwo']
   display(dfevent.head())
   
   colors = ['red','purple','green','yellow','orange','brown','black','violet','slategrey','khaki'
             'gray','silver','whitesmoke','rosybrown','firebrick','darksalmon','sienna','sandybrown'
             'olivedrab','chartreuse','palegreen','darkgreen','seagreen','navy','peachpuff','darkorange'
             'navajowhite','darkgoldenrod','lemonchiffon','mediumseagreen','cadetblue','skyblue','dodgerblue','slategray']

   for i in consolidated_events.index:
      
      sz = len(consolidated_events.index)-1
      # Plot the data with Matplotlib Plt
      # plot start to end first 
      
      if i==0:
         #endindex = consolidated_events['index'].loc[i] + int((consolidated_events['days'].loc[i])*48)
         endindex = consolidated_events['index'].loc[sz] + int((consolidated_events['days'].loc[sz])*48) + 48 #Added one extra day at the end
         x = df_resampled['time_stamp'].loc[consolidated_events['index'].loc[i]:endindex]
         y = df_resampled['corrected'].loc[consolidated_events['index'].loc[i]:endindex]
         plt.plot(x,y,label='PM 2.5')
         x = [df_resampled['time_stamp'].loc[consolidated_events['index'].loc[i]], df_resampled['time_stamp'].loc[endindex]]
         y = [150.5,150.5]
         plt.plot(x,y,linewidth=1, linestyle='dashed',label = 'Level 2 Criteria Concentration')

      begindex = consolidated_events['index'].loc[i]
      endindex = begindex + int((consolidated_events['days'].loc[i])*48)
      
      x = [df_resampled['time_stamp'].loc[begindex],df_resampled['time_stamp'].loc[begindex]]
      y = [0,800]   
      plt.plot(x,y,linewidth=1,color = colors[i])
      #color = (0, i / 20.0, 0, 1)
      #color=plt.cm.RdYlBu(i)
      
      x= [df_resampled['time_stamp'].loc[begindex],df_resampled['time_stamp'].loc[endindex]]
      y= [800,800]
      plt.plot(x,y,linewidth=1,color =  colors[i])

      
      
      x = [df_resampled['time_stamp'].loc[endindex],df_resampled['time_stamp'].loc[endindex]]
      y = [0,800]
      plt.plot(x,y,linewidth=1,color =  colors[i],label=f'Event {i +1}')

   plt.legend(loc='upper right')
   plt.rc('legend', fontsize = 14)
   plt.show()
   
   
   
   for i,dataf in enumerate(dfevent['index']):
         dfevent['slope_first'] = -10.10
         dfevent['slope_second'] = -10.10
         dfevent['slope_third'] = -10.10
         dfevent['slope_fourth'] = -10.10
         
         if dfevent['index'].loc[i]<= 6:
            dfevent.loc[i] = np.nan
            continue
            
            #print(slope)
         if dfevent['index'].loc[i]<=12:
            dfevent.loc[i,'slope_first'] = (df_resampled['corrected'].loc[6] -df_resampled['corrected'].loc[0])/((6-0)*30)
            continue
         if dfevent['index'].loc[i]<=18:
            dfevent.loc[i,'slope_first'] = ((df_resampled['corrected'].loc[6] -df_resampled['corrected'].loc[0])/((6-0)*30))
            dfevent.loc[i,'slope_second'] = (df_resampled['corrected'].loc[12] -df_resampled['corrected'].loc[0])/((12-0)*30)
            continue
         if dfevent['index'].loc[i]<=24:
            dfevent.loc[i,'slope_third'] = (df_resampled['corrected'].loc[18] -df_resampled['corrected'].loc[0])/((18-0)*30)
            dfevent.loc[i,'slope_second'] = (df_resampled['corrected'].loc[12] -df_resampled['corrected'].loc[0])/((12-0)*30)
            dfevent.loc[i,'slope_first'] = (df_resampled['corrected'].loc[6] -df_resampled['corrected'].loc[0])/((6-0)*30)
            continue
         if dfevent['index'].loc[i]>24:
            ind = dfevent['index'].loc[i]
            dfevent.loc[i,'slope_fourth'] = (df_resampled['corrected'].loc[ind] -df_resampled['corrected'].loc[ind-24])/((24-0)*30)
            dfevent.loc[i,'slope_third'] = (df_resampled['corrected'].loc[ind] -df_resampled['corrected'].loc[ind-18])/((18-0)*30)
            dfevent.loc[i,'slope_second'] = (df_resampled['corrected'].loc[ind] -df_resampled['corrected'].loc[ind-12])/((12-0)*30)
            dfevent.loc[i,'slope_first'] = (df_resampled['corrected'].loc[ind] -df_resampled['corrected'].loc[ind-6])/((6-0)*30)
   print(name_label[k])
   display(dfevent)
   dfs_leveltwo[str(name_label[k])+ '_leveltwo'] = dfevent.copy('deep')
            
   #= dfs_leveltwo.append{str(name_label[k])+ '_leveltwo': consolidated_events}
#print(dfs_leveltwo['SAFE_Swillup_Creek_leveltwo'])
dfs_leveltwo.keys()
print(dfs_leveltwo['SAFE_Swillup_Creek_leveltwo'])
display(dfevent)

 #****************************************************************#
 #****************************************************************#
 #****************************************************************#
 #****************************************************************#  


   
print(r'Eents greater than 250.5 micro-grams/$m^3$ - 6, 30 minute intervals out of 48 30 minute intervals (1 day)')  
dfs_leveltwo = {}
#colors = mpl.colormaps.get_cmap('viridis').resampled(20).colors
for k,df in enumerate(sensor_list_gf):
   # Extreme Event Identification 

   
   plt.figure(figsize=(10,8))
   plt.title('Level 1 Events:  ' + sensor_location_names[k],fontsize = 24,weight = 'bold')
   plt.xlabel('Date',fontsize=18,weight = 'bold')
   plt.ylabel(r'PM2.5 micro-grams/$m^3$',fontsize=18,weight = 'bold')
   plt.xticks(rotation=45,fontsize=20) # This was important to limit the number of days displayed on the x axis
   plt.yticks(fontsize=20)
   df.head()
   df_resampled = df.resample('30T').mean()
   df_resampled['time_stamp'] = df_resampled.index
   df_resampled.reset_index(drop=True, inplace=True)
   df_resampled.head()
   
   

   # Counting 30 min concentrations > 250.5 if present 3 out of 48 hours. 
   events=pd.DataFrame(columns = ['index','time_stamp','third','fourth'])
   events.loc[0]=-50

   i =0 
   #for i,f in enumerate(df_nan['pm2.5_cf_1_a']):
   for i in df_resampled.index:
      j=i+47
      df_resampled.loc[i,'event_count'] = (df_resampled['pm25_epa'].loc[i:j]>250.5).sum()
      
   df_resampled.describe()
   df_resampled
   df_resampled['event_count'].describe()
   countlist = (df_resampled[df_resampled['event_count']>6].index.values)
   print(name_label[k])
   #print(countlist)
   # for i in countlist:
   #    print({df_resampled['time_stamp'].loc[i]})

   j = 0
   # Now separate events by 48 hours
   for i,f in enumerate(countlist):
      if i ==0:
         events.loc[j,'index'] = countlist[i] 
         events.loc[j,'time_stamp'] = df_resampled['time_stamp'].loc[countlist[i]]
         j = j+1
         
      if countlist[i]> (events['index'].loc[j-1] + 48):
         events.loc[j,'index'] = countlist[i] 
         events.loc[j,'time_stamp'] = df_resampled['time_stamp'].loc[countlist[i]]
         j = j+1
         
   # print(events['index'])
   # for i,f in enumerate(events):
      # print(i,f['index'].loc[i],f['time_stamp'].loc[i])
   display(events)
   
   # for i,f in enumerate(events):
   #    print({df_resampled['time_stamp'].loc[f]})
   consolidated_events=pd.DataFrame(columns = ['index','time_start','time_end','days'])
   
   # Identify and isolate consecutive events 


 
 
   j = 0 
   for p,f in enumerate(events['index']):
      #print(p)
      if p ==0:
         consolidated_events.loc[j,'index'] = events['index'].loc[p] 
         consolidated_events.loc[j,'time_start'] = events['time_stamp'].loc[p]
         j = j+1
         #print(p,j)
         continue
         
      diff = events['time_stamp'].loc[p] -  events['time_stamp'].loc[p-1]  
      hours, minutes, seconds = convert_timedelta(diff)
      total_hours = hours + minutes/60 + seconds/3600
      
      if total_hours > 25:
         consolidated_events.loc[j,'index'] = events['index'].loc[p] 
         consolidated_events.loc[j,'time_start'] = events['time_stamp'].loc[p]
         j = j+1
         #print(p,j,total_hours)
      else:
         consolidated_events.loc[j-1,'time_end'] = events['time_stamp'].loc[p]
         #print(p,j, total_hours)
   consolidated_events['time_start'] = consolidated_events['time_start'].astype('datetime64')
   consolidated_events['time_end'] = consolidated_events['time_end'].astype('datetime64')

   for i in consolidated_events.index:
      
   
      if  (consolidated_events['time_end'].loc[i] is pd.NaT):
         consolidated_events['time_end'].loc[i] = (consolidated_events['time_start'].loc[i]) + (timedelta(days=1)) 
      diff = consolidated_events['time_end'].loc[i] -  consolidated_events['time_start'].loc[i]  
      hours, minutes, seconds = convert_timedelta(diff)
      consolidated_events['days'].loc[i]= (hours + minutes/60 + seconds/3600)/24
  # my_dict['name']='Nick'
   display(df_resampled.head())
   dfs_leveltwo[str(name_label[k])+ '_leveltwo'] =  consolidated_events
   dfevent = dfs_leveltwo[str(name_label[k])+ '_leveltwo']
   display(dfevent.head())
   
   colors = ['red','purple','green','yellow','orange','brown','black','violet','slategrey','khaki'
             'gray','silver','whitesmoke','rosybrown','firebrick','darksalmon','sienna','sandybrown'
             'olivedrab','chartreuse','palegreen','darkgreen','seagreen','navy','peachpuff','darkorange'
             'navajowhite','darkgoldenrod','lemonchiffon','mediumseagreen','cadetblue','skyblue','dodgerblue','slategray'] 
   for i in consolidated_events.index:
      
      sz = len(consolidated_events.index)-1
      # Plot the data with Matplotlib Plt
      # plot start to end first 
      
      if i==0:
         if consolidated_events['index'].loc[i]==-50:
            continue 
         #endindex = consolidated_events['index'].loc[i] + int((consolidated_events['days'].loc[i])*48)
         endindex = consolidated_events['index'].loc[sz] + int((consolidated_events['days'].loc[sz])*48) + 48 #Added one extra day at the end
         x = df_resampled['time_stamp'].loc[consolidated_events['index'].loc[i]:endindex]
         y = df_resampled['corrected'].loc[consolidated_events['index'].loc[i]:endindex]
         plt.plot(x,y,label='PM 2.5')
         x = [df_resampled['time_stamp'].loc[consolidated_events['index'].loc[i]], df_resampled['time_stamp'].loc[endindex]]
         y = [250.5,250.5]
         plt.plot(x,y,linewidth=1, linestyle='dashed',label = 'Level 1 Criteria Concentration')

      begindex = consolidated_events['index'].loc[i]
      endindex = begindex + int((consolidated_events['days'].loc[i])*48)
      
      x = [df_resampled['time_stamp'].loc[begindex],df_resampled['time_stamp'].loc[begindex]]
      y = [0,800]   
      plt.plot(x,y,linewidth=1,color = colors[i])
      #color = (0, i / 20.0, 0, 1)
      #color=plt.cm.RdYlBu(i)
      
      x= [df_resampled['time_stamp'].loc[begindex],df_resampled['time_stamp'].loc[endindex]]
      y= [800,800]
      plt.plot(x,y,linewidth=1,color =  colors[i])

      
      
      x = [df_resampled['time_stamp'].loc[endindex],df_resampled['time_stamp'].loc[endindex]]
      y = [0,800]
      plt.plot(x,y,linewidth=1,color =  colors[i],label=f'Event {i +1}')

   plt.legend(loc='upper right')
   plt.rc('legend', fontsize = 14)
   plt.show()
   
   
   
   for i,dataf in enumerate(dfevent['index']):
         dfevent['slope_first'] = -10.10
         dfevent['slope_second'] = -10.10
         dfevent['slope_third'] = -10.10
         dfevent['slope_fourth'] = -10.10
         
         if dfevent['index'].loc[i]<= 6:
            dfevent.loc[i] = np.nan
            continue
            
            #print(slope)
         if dfevent['index'].loc[i]<=12:
            dfevent.loc[i,'slope_first'] = (df_resampled['corrected'].loc[6] -df_resampled['corrected'].loc[0])/((6-0)*30)
            continue
         if dfevent['index'].loc[i]<=18:
            dfevent.loc[i,'slope_first'] = ((df_resampled['corrected'].loc[6] -df_resampled['corrected'].loc[0])/((6-0)*30))
            dfevent.loc[i,'slope_second'] = (df_resampled['corrected'].loc[12] -df_resampled['corrected'].loc[0])/((12-0)*30)
            continue
         if dfevent['index'].loc[i]<=24:
            dfevent.loc[i,'slope_third'] = (df_resampled['corrected'].loc[18] -df_resampled['corrected'].loc[0])/((18-0)*30)
            dfevent.loc[i,'slope_second'] = (df_resampled['corrected'].loc[12] -df_resampled['corrected'].loc[0])/((12-0)*30)
            dfevent.loc[i,'slope_first'] = (df_resampled['corrected'].loc[6] -df_resampled['corrected'].loc[0])/((6-0)*30)
            continue
         if dfevent['index'].loc[i]>24:
            ind = dfevent['index'].loc[i]
            dfevent.loc[i,'slope_fourth'] = (df_resampled['corrected'].loc[ind] -df_resampled['corrected'].loc[ind-24])/((24-0)*30)
            dfevent.loc[i,'slope_third'] = (df_resampled['corrected'].loc[ind] -df_resampled['corrected'].loc[ind-18])/((18-0)*30)
            dfevent.loc[i,'slope_second'] = (df_resampled['corrected'].loc[ind] -df_resampled['corrected'].loc[ind-12])/((12-0)*30)
            dfevent.loc[i,'slope_first'] = (df_resampled['corrected'].loc[ind] -df_resampled['corrected'].loc[ind-6])/((6-0)*30)
   print(name_label[k])
   display(dfevent)
   dfs_leveltwo[str(name_label[k])+ '_leveltwo'] = dfevent.copy('deep')
            
   #= dfs_leveltwo.append{str(name_label[k])+ '_leveltwo': consolidated_events}
#print(dfs_leveltwo['SAFE_Swillup_Creek_leveltwo'])
dfs_leveltwo.keys()
print(dfs_leveltwo['SAFE_Swillup_Creek_leveltwo'])
display(dfevent)

'''

 #****************************************************************#
 #****************************************************************#
 #****************************************************************#
 #****************************************************************# 

#First versions of event identification (before the graphing)

 
 
print('Eents greater than 55.5 ug/m3 - 6, 30 minute intervals out of 48 30 minute intervals (1 day)')  
for k,df in enumerate(sensor_list_gf):
   # Extreme Event Identification 



   df_resampled = df.resample('30T').mean()
   df_resampled['time_stamp'] = df_resampled.index
   df_resampled.reset_index(drop=True, inplace=True)
   df_resampled.head()

   # Counting 30 min concentrations > 55.5 if present 3 out of 48 hours. 
   events=pd.DataFrame(columns = ['index','time_stamp','third','fourth'])
   events.loc[0]=-50

   i =0 
   #for i,f in enumerate(df_nan['pm2.5_cf_1_a']):
   for i in df_resampled.index:
      j=i+47
      df_resampled.loc[i,'event_count'] = (df_resampled['pm25_epa'].loc[i:j]>55.5).sum()
      
   df_resampled.describe()
   df_resampled
   df_resampled['event_count'].describe()
   countlist = (df_resampled[df_resampled['event_count']>6].index.values)
   print(name_label[k])
   #print(countlist)
   # for i in countlist:
   #    print({df_resampled['time_stamp'].loc[i]})

   j = 0
   # Now separate events by 48 hours
   for i,f in enumerate(countlist):
      if i ==0:
         events.loc[j,'index'] = countlist[i] 
         events.loc[j,'time_stamp'] = df_resampled['time_stamp'].loc[countlist[i]]
         j = j+1
         
      if countlist[i]> (events['index'].loc[j-1] + 48):
         events.loc[j,'index'] = countlist[i] 
         events.loc[j,'time_stamp'] = df_resampled['time_stamp'].loc[countlist[i]]
         j = j+1
         
   # print(events['index'])
   # for i,f in enumerate(events):
      # print(i,f['index'].loc[i],f['time_stamp'].loc[i])
   display(events)
   
   # for i,f in enumerate(events):
   #    print({df_resampled['time_stamp'].loc[f]})
   consolidated_events=pd.DataFrame(columns = ['index','time_start','time_end','days'])
   
   # Identify and isolate consecutive events 


 
 
   j = 0 
   for p,f in enumerate(events['index']):
     # print(p)
      if p ==0:
         consolidated_events.loc[j,'index'] = events['index'].loc[p] 
         consolidated_events.loc[j,'time_start'] = events['time_stamp'].loc[p]
         j = j+1
         #print(p,j)
         continue
         
      diff = events['time_stamp'].loc[p] -  events['time_stamp'].loc[p-1]  
      hours, minutes, seconds = convert_timedelta(diff)
      total_hours = hours + minutes/60 + seconds/3600
      
      if total_hours > 25:
         consolidated_events.loc[j,'index'] = events['index'].loc[p] 
         consolidated_events.loc[j,'time_start'] = events['time_stamp'].loc[p]
         j = j+1
         #print(p,j,total_hours)
      else:
         consolidated_events.loc[j-1,'time_end'] = events['time_stamp'].loc[p]
         #print(p,j, total_hours)
   consolidated_events['time_start'] = consolidated_events['time_start'].astype('datetime64')
   consolidated_events['time_end'] = consolidated_events['time_end'].astype('datetime64')

   for i in consolidated_events.index:
   
      if  (consolidated_events['time_end'].loc[i] is pd.NaT):
         consolidated_events['time_end'].loc[i] = (consolidated_events['time_start'].loc[i]) + (timedelta(days=1)) 
      diff = consolidated_events['time_end'].loc[i] -  consolidated_events['time_start'].loc[i]  
      hours, minutes, seconds = convert_timedelta(diff)
      consolidated_events['days'].loc[i]= (hours + minutes/60 + seconds/3600)/24
      
   display(consolidated_events)



#****************************************************************#
#****************************************************************#
#****************************************************************# 
#****************************************************************#
#****************************************************************#
#****************************************************************#
#****************************************************************# 
 
print('Events greater than 250.5 ug/m3 - 6, 30 minute intervals out of 48 30 minute intervals (1 day)')  





#time_stamp is now in local time 
for k,df in enumerate(sensor_list_gf):
   # Extreme Event Identification 
   df_resampled = df.resample('30T').mean()
   df_resampled['time_stamp'] = df_resampled.index
   df_resampled.reset_index(drop=True, inplace=True)
   df_resampled.head()
   # Counting 30 min concentrations > 250.5 if present 3 out of 48 hours. 
   events=pd.DataFrame(columns = ['index','time_stamp','third','fourth'])
   events.loc[0]=-50

   i =0 
   #for i,f in enumerate(df_nan['pm2.5_cf_1_a']):
   for i in df_resampled.index:
      j=i+47
      df_resampled.loc[i,'event_count'] = (df_resampled['pm25_epa'].loc[i:j]>250.5).sum()
      
   df_resampled.describe()
   df_resampled
   df_resampled['event_count'].describe()
   countlist = (df_resampled[df_resampled['event_count']>6].index.values)
   print(name_label[k])
   print(countlist)
   # for i in countlist:
   #    print({df_resampled['time_stamp'].loc[i]})


   j = 0
   # Now separate events by 48 hours
   for i,f in enumerate(countlist):
      if i ==0:
         events.loc[j,'index'] = countlist[i] 
         events.loc[j,'time_stamp'] = df_resampled['time_stamp'].loc[countlist[i]]
         j = j+1
         
      if countlist[i]> (events['index'].loc[j-1] + 48):
         events.loc[j,'index'] = countlist[i] 
         events.loc[j,'time_stamp'] = df_resampled['time_stamp'].loc[countlist[i]]
         j = j+1
         
   # print(events['index'])
   # for i,f in enumerate(events):
      # print(i,f['index'].loc[i],f['time_stamp'].loc[i])
   display(events)
   
   # for i,f in enumerate(events):
   #    print({df_resampled['time_stamp'].loc[f]})
   consolidated_events=pd.DataFrame(columns = ['index','time_start','time_end','days'])
   
   # Identify and isolate consecutive events 


 
 
   j = 0 
   for p,f in enumerate(events['index']):
      #print(p)
      if p ==0:
         consolidated_events.loc[j,'index'] = events['index'].loc[p] 
         consolidated_events.loc[j,'time_start'] = events['time_stamp'].loc[p]
         j = j+1
         #print(p,j)
         continue
         
      diff = events['time_stamp'].loc[p] -  events['time_stamp'].loc[p-1]  
      hours, minutes, seconds = convert_timedelta(diff)
      total_hours = hours + minutes/60 + seconds/3600
      
      if total_hours > 25:
         consolidated_events.loc[j,'index'] = events['index'].loc[p] 
         consolidated_events.loc[j,'time_start'] = events['time_stamp'].loc[p]
         j = j+1
         #print(p,j,total_hours)
      else:
         consolidated_events.loc[j-1,'time_end'] = events['time_stamp'].loc[p]
         #print(p,j, total_hours)
   consolidated_events['time_start'] = consolidated_events['time_start'].astype('datetime64')
   consolidated_events['time_end'] = consolidated_events['time_end'].astype('datetime64')

   for i in consolidated_events.index:
   
      if  (consolidated_events['time_end'].loc[i] is pd.NaT):
         consolidated_events['time_end'].loc[i] = (consolidated_events['time_start'].loc[i]) + (timedelta(days=1)) 
      diff = consolidated_events['time_end'].loc[i] -  consolidated_events['time_start'].loc[i]  
      hours, minutes, seconds = convert_timedelta(diff)
      consolidated_events['days'].loc[i]= (hours + minutes/60 + seconds/3600)/24

   display(consolidated_events)

##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################


# Animation for all locations 
#To make it easier to run downsample to 3 hour averages, and restrict to one month 

df_anim = pd.DataFrame()
df_anim.shape

for i, dg in enumerate(sensor_list_gf):
   print(i)
   display(dg.shape)
   
   dt = dg.loc['2021/09/01':'2021/09/30'].resample('3H').mean()
   display(dt.shape)
   df_anim = df_anim.append(dt)
   display(df_anim.shape)
df_anim['dummy_column_for_size'] = 10
      
      

 

      
df_anim.head()
#Check to make sur ethe maximum concentration is within the color range 
df_anim['corrected'].max()

fig = px.scatter_mapbox(df_anim, lon = df_anim['longitude'], lat = df_anim['latitude'], zoom = 8, color = df_anim['corrected'], size = df_anim['dummy_column_for_size'],
                        color_continuous_scale = [(0.00, "green"),   (0.1, "green"),
                                        (0.1, "yellow"), (0.2, "yellow"),
                                        (0.2, "orange"),  (0.3, "orange"),
                                        (0.3, "red"),  (0.4, "red"),
                                        (0.4, "purple"),  (0.6, "purple"),
                                        (0.6, "maroon"),  (1.00, "maroon"),
                                        ],
              range_color = (0, 500),
                        width = 900, height = 600, title = 'PM2.5 15 min Average Concentration - Aug 2021',animation_frame=df_anim.index)
print('working...')
fig.update_layout(mapbox_style="open-street-map")


#Make it faster
fig.update_geos(projection_type="equirectangular", visible=True, resolution=110)

   #fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 5

#A different base map layer
#fig.update_layout(mapbox_style="white-bg",
# mapbox_layers=[
#        {
#            "below": 'traces',
#            "sourcetype": "raster",
#            "sourceattribution": "United States Geological Survey",
#            "source": [
#                "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
#            ]
#        }
#      ])

fig.update_layout(margin={"r":0,"t":50,"l":0,"b":10})


print('plot complete.')
fig.show()

##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################



# Try plotting without animation 
# This will help with conversio to a video/gif value

      
#Setup plotly frame to figure conversion (source link in 'Resources' page in shared drive )
def plotly_fig2array(fig):
    #convert Plotly fig to  an array
    fig_bytes = fig.to_image(format="png")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return np.asarray(img)
 
 
date_index3[239]
date_index3 = pd.date_range('2021-09-01 00:00:00-0700', periods=240, freq='3H')
date_index3.day

def make_frame(t):
   #print(int(t*23))
   local = int(t*23)
   print(local)
   #local = int(local/2)
   df = df_anim.loc[date_index3[local]]
   fig = px.scatter_mapbox(df, lon = df['longitude'], lat = df['latitude'], zoom = 8, color = df['corrected'], size = df['dummy_column_for_size'],
                           color_continuous_scale = [(0.00, "green"),   (0.1, "green"),
                                          (0.1, "yellow"), (0.2, "yellow"),
                                          (0.2, "orange"),  (0.3, "orange"),
                                          (0.3, "red"),  (0.4, "red"),
                                          (0.4, "purple"),  (0.6, "purple"),
                                          (0.6, "maroon"),  (1.00, "maroon"),
                                          ],
               range_color = (0, 500),
                           width = 900, height = 600, title =date_index3[int(t*100)].day )
   print('working...')
   fig.update_layout(mapbox_style="open-street-map")


   #Make it faster
   fig.update_geos(projection_type="equirectangular", visible=True, resolution=110)

      #fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 5

   #A different base map layer
   #fig.update_layout(mapbox_style="white-bg",
   # mapbox_layers=[
   #        {
   #            "below": 'traces',
   #            "sourcetype": "raster",
   #            "sourceattribution": "United States Geological Survey",
   #            "source": [
   #                "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
   #            ]
   #        }
   #      ])

   fig.update_layout(margin={"r":0,"t":50,"l":0,"b":10})

   #Pass on plotly frame to conversion function (plotly_fig2array), and then to mpy.VideoClip
   # animation = mpy.VideoClip(plotly_fig2array(fig),duration = 30)
   #animation.write_gif("SepPMConc.gif", fps=20)

   #print('plot complete.')
   #fig.show()
   return plotly_fig2array(fig)
   
   


animation = mpy.VideoClip(make_frame, duration = 10)
animation.write_gif("pm_31_1.gif", fps=20)
   


   

      
     

df_anim['2021/09/01']
display(df_anim)
df_anim.shape
df_anim.index.min()
df_anim.index.max()
df_anim.index.sort

df_anim.loc[date_index3[0]]
