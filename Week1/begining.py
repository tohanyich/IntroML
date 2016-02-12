# -*- coding: UTF-8 -*-

import pandas
data = pandas.read_csv('titan.csv',index_col='PassengerId')

pasNumber = data['Survived'].count()

#Ex.1
#sex_data = data.groupby('Sex')
#ds = sex_data.count()
#print(ds)

#Ex.2
#print("alive %d all %d perc %f") % (data['Survived'].sum(),data['Survived'].count(),float(data['Survived'].sum())/float(data['Survived'].count()*100))

#Ex.3
#class_data = data.groupby('Pclass')
#pasFC, pasSC, pasTC = class_data['Survived'].count()
#print("first class %d all %d perc %f") % (pasFC, pasNumber, float(pasFC)/float(pasNumber)*100)

#Ex.4
#print("sum %f count %f")%(float(data['Age'].sum()),float(data['Age'].count()))
print("avg %f median %f")%(data['Age'].mean(),data['Age'].median())

#Ex.5
#print(data['SibSp'].corr(data['Parch'],method = 'pearson'))

#Ex.6
def find_name(name):
    first_symbol = name.find('(')+1
    if first_symbol == 0: 
        first_symbol = name.find('. ')+2
        
    last_symbol = name.find(' ',first_symbol)
    return name[first_symbol:last_symbol]
dataF = data[data.Sex == 'female']
Names = dataF['Name'].apply(lambda x: find_name(str(x)))
dataF['FName'] = Names
print(dataF.groupby('FName').count().sort('Sex'))