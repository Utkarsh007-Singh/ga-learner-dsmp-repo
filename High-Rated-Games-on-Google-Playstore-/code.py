# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns





#Code starts here
data = pd.read_csv(path)
data.hist('Rating')
data = data[data['Rating']<=5 ]
data.hist('Rating')


#Code ends here


# --------------
# code starts here
#heredata.isnull().sum()
total_null=data.isnull().sum()
print(total_null)
percent_null=(total_null/data.isnull().count())
print(percent_null)
missing_data=pd.concat([total_null,percent_null],axis=1,keys=['Total','Percent'])
print(missing_data)
data.dropna(inplace=True)
total_null_1=data.isnull().sum()
print(total_null_1)
percent_null_1=(total_null_1/data.isnull().count())
print(percent_null_1)
missing_data_1=pd.concat([total_null_1,percent_null_1],axis=1,keys=['Total','Percent'])

# code ends here


# --------------

#Code starts here
sns.catplot(x="Category",y="Rating",data=data, kind="box",height=10)
#lt.set_title('Rating vs Category [BoxPlot]')


#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here
data['Installs']=data['Installs'].str.replace(',', '')
data['Installs']=data['Installs'].str.replace('+', '')
data['Installs'] = data['Installs'].astype(int)
le = LabelEncoder()
data['Installs']=le.fit_transform(data['Installs'])
sns.regplot(x=data['Installs'], y=data['Rating'])
plt.title('Rating vs Installs [RegPlot]')



#Code ends here



# --------------
#Code starts here
print(data['Price'].value_counts())
data['Price']=data['Price'].str.replace('$', '')
data['Price']= data['Price'].astype('float')
sns.regplot(x="Price", y="Rating" ,data=data)
plt.title('Rating vs Price [RegPlot]')


#Code ends here


# --------------

#Code starts here
data1=data['Genres'].str.split(';',expand = True)
data['Genres']=data1[0]
print(data.head(5))
#gr_mean = data['Genres','Rating'].groupby('Genres',as_index=False).mean()
gr_mean=data[['Genres', 'Rating']].groupby(['Genres'], as_index=False).mean()
#print(gr_mean)
gr_mean=gr_mean.sort_values(by='Rating')
print(gr_mean)



#Code ends here


# --------------

#Code starts here
data['Last Updated']=pd.to_datetime(data['Last Updated'])
max_date = max(data['Last Updated'])
print(max_date)

data['Last Updated Days']=(max_date - data['Last Updated']).dt.days
print(data['Last Updated Days'])
sns.regplot(x="Last Updated Days", y="Rating", data=data)
plt.title('Rating vs Last Updated [RegPlot]')



#Code ends here


