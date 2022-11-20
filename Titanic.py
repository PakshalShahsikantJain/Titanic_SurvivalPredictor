###########################################################################################################################
##
##  Author : Pakshal Shashikant Jain
##  Date : 23/05/2021
##  Program : Machine Learning of Titanic DataSet using Logistic Regression Algorithm
##
################################################################################################################################
import numpy as np 
import pandas as pd 
import seaborn as sb
import matplotlib.pyplot as plt 
from matplotlib.pyplot import figure,show
from seaborn import countplot 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score

def TitanicLogistic() :
    print("Inside Logistic Function")
    #step 1 : Load Data 
    
    titanic_Data = pd.read_csv('MarvellousTitanicDataset.csv')
    print("First Five Records of Data set : ")
    
    print(titanic_Data.head())
    print("Total Number of Records are : ",len(titanic_Data))
    print(titanic_Data.info())

    #Step 2 : Analyze The Data
    
    print("----------------------Visualization of Survived And Non Survived Passenger-------------------")
    figure()
    countplot(data = titanic_Data,x = "Survived").set_title("Survived VS Non Survived")
    show()

    print("----------------------Visualization According To Sex-----------------")
    figure()
    countplot(data = titanic_Data,x = "Survived",hue = "Sex").set_title("Visualization According To Sex")
    show()

    print("----------------------Visualization According To Passenger Class-----------------")
    figure()
    countplot(data = titanic_Data,x = "Survived",hue = "Pclass").set_title("Visualization According To PClass")
    show()

    print("Survived VS Non - survived Based on Age")
    figure()
    titanic_Data["Age"].plot.hist().set_title("Visualization According To Age")
    show()

    print("----------------------Visualization According To Embarked-----------------")
    figure()
    countplot(data = titanic_Data,x = "Survived",hue = "Embarked").set_title("Visualization According To Embarked")
    show()

    # Step 3 : Data Cleaning 
    titanic_Data.drop("zero",axis = 1,inplace = True)
    print("----------------------Data After Column Removal--------------------------------")
    print(titanic_Data.head())

    print("----------------------Sex Column Before Updation-------------------------------")
    Sex = pd.get_dummies(titanic_Data["Sex"])
    print(Sex.head())

    print("----------------------Sex Column After Updation--------------------------------")
    Sex = pd.get_dummies(titanic_Data["Sex"],drop_first = True)
    print(Sex.head())

    print("----------------------Passeger_Class Columns Before Updation-------------------")
    PClass = pd.get_dummies(titanic_Data["Pclass"])
    print(PClass.head())

    print("----------------------Passeger Column After Updation---------------------------")
    PClass = pd.get_dummies(titanic_Data["Pclass"],drop_first = True)
    print(PClass.head())

    #if np.any(np.isnan(titanic_Data)) == True :
    #    print("There is Nan Value")
    #else :
    #    print("Data is Cleaned")
    
    titanic_Data["Embarked"].fillna(titanic_Data["Embarked"].mode()[0] , inplace = True)
    print("----------------------Data After Cleaning--------------------------------------")
    print(titanic_Data.head(64))
    
    #Step 4 : Training of Data
    
    obj = LogisticRegression(solver = 'lbfgs',max_iter = 1000)

    print("------------------------Data For Training and Testing Purpose--------------------------")
    data = titanic_Data.iloc[:,:8]
    print(data)

    print("------------------------Target For Training and Testing Purpose--------------------------")
    target = titanic_Data["Survived"]
    print(target)

    data_train,data_test,target_train,target_test = train_test_split(data,target,test_size = 0.5)

    obj.fit(data_train,target_train)

    print("-----------------------------Final Output of This Alogorithm-------------------------------")

    output = obj.predict(data_test)
    print(output)

    Accuracy = accuracy_score(target_test,output)
    print("\nAccuracy of Logistic Regression Alogorithm is : ",Accuracy*100,"%")    

def main() :
    print("Jay Ganesh.....")
    print("---------------Logistic Case Study----------------------")

    TitanicLogistic()
if __name__ == "__main__" :
    main()