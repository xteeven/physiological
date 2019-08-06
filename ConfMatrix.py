# %% imports
from os import listdir
from os.path import isfile, join, dirname, splitext, abspath
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD

=======
import csv
from babel.numbers import format_number, format_decimal, format_percent
>>>>>>> d9dd1dba44bf4f3715ef36b5739c682c8e220f16
=======
import csv
from babel.numbers import format_number, format_decimal, format_percent
>>>>>>> d9dd1dba44bf4f3715ef36b5739c682c8e220f16
=======
import csv
from babel.numbers import format_number, format_decimal, format_percent
>>>>>>> d9dd1dba44bf4f3715ef36b5739c682c8e220f16
=======
import csv
from babel.numbers import format_number, format_decimal, format_percent
>>>>>>> d9dd1dba44bf4f3715ef36b5739c682c8e220f16
import psignifit as ps
from matplotlib.ticker import ScalarFormatter 
from sys import getsizeof, argv

# %% open

def openDocuments(pathdir=None):
    retorno = []
    if pathdir == None:

        pathdir = abspath('')
    else:

        pathdir = abspath(pathdir)

    for f in listdir(pathdir):
        if isfile(abspath(pathdir)+"\\"+f):
            retorno.append(abspath(pathdir)+"\\"+f)
    return retorno

# %% Filter just csv files
def filterByFormat(extension, filepaths):
    return [fp for fp in filepaths if splitext(fp)[-1].lower() == extension]

# %% Plot
def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap,vmin=0, vmax=1.0) # imshow
    plt.title(title)
    
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)

    for i in range(len(df_confusion)):
        for j in range(len(df_confusion)):
            value = df_confusion.iloc[i,j]
            color = "black" if value<0.5 else "w"
            text = plt.text(j, i, str(value*100)+"%", ha="center", va="center", color=color, size=15, weight='bold') 
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
# %% Execute
filescsv = filterByFormat(".csv", openDocuments("ActualFriction"))

# %% read File
user = 0
print(filescsv[user])
readfile = pd.read_csv(filescsv[user], delimiter=";")

# %%separate conditions
# Condition Bump and Holes
BH = readfile[readfile.Condition != 3]
BH.rendered = BH.rendered.replace(1,"Hole")
BH.rendered = BH.rendered.replace(-1,"Bump")
# %% Condition YesNo
YN = readfile[readfile.Condition == 3]
YN.rendered = YN.rendered.replace(1,"Yes")
YN.rendered = YN.rendered.replace(-1,"No")

# %% 
confusion_matrixBH1 = pd.crosstab(BH.rendered[BH.Condition == 1],BH.Answer[BH.Condition == 1], rownames=['Actual'], colnames=['Perceived'])
confusion_matrixBH2 = pd.crosstab(BH.rendered[BH.Condition == 2],BH.Answer[BH.Condition == 2], rownames=['Actual'], colnames=['Perceived'])
confusion_matrixYN = pd.crosstab(YN.rendered,YN.Answer, rownames=['Actual'], colnames=['Perceived'])

# %% 
CMBH1 = confusion_matrixBH1/confusion_matrixBH1.sum(axis=1)
CMBH2 = confusion_matrixBH2/confusion_matrixBH2.sum(axis=1)
CMYN = confusion_matrixYN/confusion_matrixYN.sum(axis=1)
# %% 
plt.style.use(plt.style.available[20])
#plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['savefig.facecolor'] = 'w'
plot_confusion_matrix(CMYN,"Condition 3")
plt.savefig("User_"+str(user)+"C3")
plot_confusion_matrix(CMBH2,"Condition 2")
plt.savefig("User_"+str(user)+"C2")
plot_confusion_matrix(CMBH1,"Condition 1")
plt.savefig("User_"+str(user)+"C1")
=======
=======
>>>>>>> d9dd1dba44bf4f3715ef36b5739c682c8e220f16
=======
>>>>>>> d9dd1dba44bf4f3715ef36b5739c682c8e220f16
=======
>>>>>>> d9dd1dba44bf4f3715ef36b5739c682c8e220f16


#%%
def getAccuracy(ConfMat):
    return (ConfMat.iloc[0,0]+ConfMat.iloc[1,1])/ConfMat.to_numpy().sum()

def getErr(ConfMat):
    return (ConfMat.iloc[1,0]+ConfMat.iloc[0,1])/ConfMat.to_numpy().sum()

def getPrecision(ConfMat):
    return (ConfMat.iloc[1,1])/(ConfMat.iloc[0,1]+ConfMat.iloc[1,1])

def getRecall(ConfMat):
    return (ConfMat.iloc[1,1])/(ConfMat.iloc[1,0]+ConfMat.iloc[1,1])

def MCC(ConfMat):
    #Matheus Correlation Coefficient
    numerator = (ConfMat.iloc[0,0]*ConfMat.iloc[1,1])-(ConfMat.iloc[1,0]*ConfMat.iloc[0,1])
    denominator = (ConfMat.iloc[0,1]+ConfMat.iloc[1,1])*(ConfMat.iloc[1,1]+ConfMat.iloc[1,0])*(ConfMat.iloc[0,1]+ConfMat.iloc[0,0])*(ConfMat.iloc[1,0]+ConfMat.iloc[0,0])
    return numerator/np.sqrt(denominator)

    

# %% Execute
filescsv = filterByFormat(".csv", openDocuments("FinalFriction"))

# %% read File

with open('ConfMatrixInfso.csv', 'w') as writeFile:
    #writer = csv.writer(writeFile, delimiter =';', lineterminator = '\n')
    header = ['User',
    'Condition', 'accuracy', 'error rate', 'precision', 'recall', 'MCC',
    'Condition', 'accuracy', 'error rate', 'precision', 'recall', 'MCC',
    'Condition', 'accuracy', 'error rate', 'precision', 'recall', 'MCC']
    #writer.writerow(header)
    CMBH1total = []
    CMBH2total = []
    CMYNtotal = []
    for user in range(13):
        readfile = pd.read_csv(filescsv[user], delimiter=";")
        # %%separate conditions
        # Condition Bump and Holes
        BH = readfile[readfile.Condition != 3]
        BH.rendered = BH.rendered.replace(1,"Hole")
        BH.rendered = BH.rendered.replace(-1,"Bump")
        # %% Condition YesNo
        YN = readfile[readfile.Condition == 3]
        YN.rendered = YN.rendered.replace(1,"Yes")
        YN.rendered = YN.rendered.replace(-1,"No")
        
        # %% 
        confusion_matrixBH1 = pd.crosstab(BH.rendered[BH.Condition == 1],BH.Answer[BH.Condition == 1], rownames=['Actual'], colnames=['Perceived'])
        confusion_matrixBH2 = pd.crosstab(BH.rendered[BH.Condition == 2],BH.Answer[BH.Condition == 2], rownames=['Actual'], colnames=['Perceived'])
        confusion_matrixYN = pd.crosstab(YN.rendered,YN.Answer, rownames=['Actual'], colnames=['Perceived'])

        # %% 
        CMBH1 = confusion_matrixBH1 #/confusion_matrixBH1.sum(axis=1)
        CMBH2 = confusion_matrixBH2 #/confusion_matrixBH2.sum(axis=1)
        CMYN = confusion_matrixYN   #/confusion_matrixYN.sum(axis=1)
        CMBH1total.append(CMBH1)
        CMBH2total.append(CMBH2)
        CMYNtotal.append(CMYN)
        # %% 
        #plt.style.use(plt.style.available[20])
        ##plt.rcParams['axes.facecolor'] = 'w'
        #plt.rcParams['savefig.facecolor'] = 'w'
        #plot_confusion_matrix(CMYN,"Condition 3")
        #plt.savefig("User_"+str(user)+"C3")
        #plot_confusion_matrix(CMBH2,"Condition 2")
        #plt.savefig("User_"+str(user)+"C2")
        #plot_confusion_matrix(CMBH1,"Condition 1")
        #plt.savefig("User_"+str(user)+"C1")


        data = [user,
        'C1', format_decimal(getAccuracy(CMBH1), locale='sv_SE'), format_decimal(getErr(CMBH1), locale='sv_SE'), 
        format_decimal(getPrecision(CMBH1), locale='sv_SE'), 
        format_decimal(getRecall(CMBH1), locale='sv_SE'), 
        format_decimal(MCC(CMBH1), locale='sv_SE'),
        'C2', format_decimal(getAccuracy(CMBH2), locale='sv_SE'), format_decimal(getErr(CMBH2), locale='sv_SE'), 
        format_decimal(getPrecision(CMBH2), locale='sv_SE'), 
        format_decimal(getRecall(CMBH2), locale='sv_SE'), 
        format_decimal(MCC(CMBH2), locale='sv_SE'),
        'C3', format_decimal(getAccuracy(CMYN), locale='sv_SE'),  format_decimal(getErr(CMYN), locale='sv_SE'),  
        format_decimal(getPrecision(CMYN), locale='sv_SE'),  
        format_decimal(getRecall(CMYN), locale='sv_SE'),  
        format_decimal(MCC(CMYN), locale='sv_SE')]
        #writer.writerow(data)
        print ('User_'+str(user)+' Done')

#writeFile.close()
#%%
from functools import reduce
CMBH1total
CMBH2total
CMYNtotal

CMBH1totalSum = reduce(lambda x, y: x.add(y, fill_value=0), CMBH1total)
CMBH1totalSum/=CMBH1totalSum.sum(axis=1)
CMBH2totalSum = reduce(lambda x, y: x.add(y, fill_value=0), CMBH2total)
CMBH2totalSum/=CMBH2totalSum.sum(axis=1)
CMYNtotalSum = reduce(lambda x, y: x.add(y, fill_value=0), CMYNtotal)
CMYNtotalSum/=CMYNtotalSum.sum(axis=1)
#%%


plt.style.use(plt.style.available[20])
#plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['savefig.facecolor'] = 'w'
plot_confusion_matrix(CMYNtotalSum.round(2),"Condition 3")
plt.savefig("User_T C3")
plot_confusion_matrix(CMBH2totalSum.round(2),"Condition 2")
plt.savefig("User_T C2")
plot_confusion_matrix(CMBH1totalSum.round(2),"Condition 1")
plt.savefig("User_T C1")


#%%0.8076923076923077
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> d9dd1dba44bf4f3715ef36b5739c682c8e220f16
=======
>>>>>>> d9dd1dba44bf4f3715ef36b5739c682c8e220f16
=======
>>>>>>> d9dd1dba44bf4f3715ef36b5739c682c8e220f16
=======
>>>>>>> d9dd1dba44bf4f3715ef36b5739c682c8e220f16
