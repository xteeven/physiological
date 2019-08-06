# %%
import subprocess
import sys
import csv
from matplotlib import pyplot as plt
from klepto.archives import file_archive
import psignifit as ps
import numpy as np
from matplotlib.ticker import ScalarFormatter 
from babel.numbers import format_number, format_decimal, format_percent
#%%
def plotsModelfit(result,
              showImediate   = True):
    """
    Plots some standard plots, meant to help you judge whether there are
    systematic deviations from the model. We dropped the statistical tests
    here though.
    
    The left plot shows the psychometric function with the data. 
    
    The central plot shows the Deviance residuals against the stimulus level. 
    Systematic deviations from 0 here would indicate that the measured data
    shows a different shape than the fitted one.
    
    The right plot shows the Deviance residuals against "time", e.g. against
    the order of the passed blocks. A trend in this plot would indicate
    learning/ changes in performance over time. 
    
    These are the same plots as presented in psignifit 2 for this purpose.
    """
    
    fit = result['Fit']
    data = result['data']
    options = result['options']
    
    minStim = min(data[:,0])
    maxStim = max(data[:,0])
    stimRange = [1.1*minStim - .1*maxStim, 1.1*maxStim - .1*minStim]
    
    plt.figure(figsize=(15,5))

    ax = plt.subplot(1,3,1)    
    # the psychometric function
    x = np.linspace(stimRange[0], stimRange[1], 1000)
    y = fit[3] + (1-fit[2]-fit[3]) * options['sigmoidHandle'](x, fit[0], fit[1])
    
    plt.plot(x, y, 'k', clip_on=False)
    plt.plot(data[:,0], data[:,1]/data[:,2], '.k', ms=10, clip_on=False)
    
    plt.xlim(stimRange)
    if options['expType'] == 'nAFC':
        plt.ylim([min(1./options['expN'], min(data[:,1]/data[:,2])), 1])
    else:
        plt.ylim([0,1])
    plt.xlabel('Stimulus Level',  fontsize=14)
    plt.ylabel('Percent Correct', fontsize=14)
    plt.title('Psychometric Function', fontsize=20)
    plt.tick_params(right=False,top=False)
    for side in ['top','right']: ax.spines[side].set_visible(False)
    plt.ticklabel_format(style='sci',scilimits=(-2,4))   
    
    ax = plt.subplot(1,3,2)
    # stimulus level vs deviance
    stdModel = fit[3] + (1-fit[2]-fit[3]) * options['sigmoidHandle'](data[:,0],fit[0],fit[1])
    deviance = data[:,1]/data[:,2] - stdModel
    stdModel = np.sqrt(stdModel * (1-stdModel))
    deviance = deviance / stdModel
    xValues = np.linspace(minStim, maxStim, 1000)
    
    plt.plot(data[:,0], deviance, 'k.', ms=10, clip_on=False)
    linefit = np.polyfit(data[:,0],deviance,1)
    plt.plot(xValues, np.polyval(linefit,xValues),'k-', clip_on=False)
    linefit = np.polyfit(data[:,0],deviance,2)
    plt.plot(xValues, np.polyval(linefit,xValues),'k--', clip_on=False)
    linefit = np.polyfit(data[:,0],deviance,3)
    plt.plot(xValues, np.polyval(linefit,xValues),'k:', clip_on=False)

    plt.xlabel('Stimulus Level',  fontsize=14)
    plt.ylabel('Deviance', fontsize=14)
    plt.title('Shape Check', fontsize=20)
    plt.tick_params(right=False,top=False)
    for side in ['top','right']: ax.spines[side].set_visible(False)
    plt.ticklabel_format(style='sci',scilimits=(-2,4))
    
    ax = plt.subplot(1,3,3)
    # block number vs deviance
    blockN = range(len(deviance))
    xValues = np.linspace(min(blockN), max(blockN), 1000)
    plt.plot(blockN, deviance, 'k.', ms=10, clip_on=False)
    linefit = np.polyfit(blockN,deviance,1)
    plt.plot(xValues, np.polyval(linefit,xValues),'k-', clip_on=False)
    linefit = np.polyfit(blockN,deviance,2)
    plt.plot(xValues, np.polyval(linefit,xValues),'k--', clip_on=False)
    linefit = np.polyfit(blockN,deviance,3)
    plt.plot(xValues, np.polyval(linefit,xValues),'k:', clip_on=False)
    
    plt.xlabel('Block Number',  fontsize=14)
    plt.ylabel('Deviance', fontsize=14)
    plt.title('Time Dependence?', fontsize=20)
    plt.tick_params(right=False,top=False)
    for side in ['top','right']: ax.spines[side].set_visible(False)
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    plt.ticklabel_format(style='sci',scilimits=(-2,4))
    
    plt.tight_layout()
    if (showImediate):
        plt.show(0)
def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

def getValue(target, Condition, x0 = 0.65, accuracy = 0):  
    fit = Condition['Fit']
    options = Condition['options']
    y = 0
    target = target
    x = 0.65
    timeout = 0
    while(abs(y-target)>accuracy):
        y = fit[3] + (1-fit[2]-fit[3]) * options['sigmoidHandle'](x,fit[0], fit[1])   
        x+=(target-y)/4
        timeout+=1
        if timeout>1000:
            accuracy = 1E-3   
        elif timeout>5000:
            print('MinErr ' + str(abs(y-target)))
            break
    return x
#%%

#%%
for style in plt.style.available:
    plt.style.use(style)
plt.style.use(plt.style.available[0])
#plotsModelfit(Model['C1'])






#%%
with open('PsychometricInfo.csv', 'w') as writeFile:
    writer = csv.writer(writeFile, delimiter =';', lineterminator = '\n')
    header = ['User', '0.25 C2', '0.5 C2', '0.75 C2', 'Sensitivity C2', 'Alpha C2', 'Beta C2', '0.25 C3', '0.5 C3', '0.75 C3', 'Sensitivity C3', 'Alpha C3', 'Beta C3']
    writer.writerow(header)
    for user in range(13):
        Model = file_archive("User_"+str(user)+"_Model.mde")
        Model.load()
        UserInfo = [user, 
        format_decimal(getValue(0.25, Model['C1']),locale='sv_SE'), 
        format_decimal(getValue(0.5, Model['C1']), locale='sv_SE'),
        format_decimal(getValue(0.75, Model['C1']), locale='sv_SE'),
        format_decimal(getValue(0.75, Model['C1'])-getValue(0.25, Model['C1']), locale='sv_SE'),
        format_decimal(Model['C1']['Fit'][0],locale='sv_SE'), 
        format_decimal(Model['C1']['Fit'][1],locale='sv_SE'), 
        format_decimal(getValue(0.25, Model['C2']), locale='sv_SE'),
        format_decimal(getValue(0.5, Model['C2']), locale='sv_SE'),
        format_decimal(getValue(0.75, Model['C2']), locale='sv_SE'),
        format_decimal(getValue(0.75, Model['C2']) - getValue(0.25, Model['C2']), locale='sv_SE'),
        format_decimal(Model['C2']['Fit'][0],locale='sv_SE'), 
        format_decimal(Model['C2']['Fit'][1],locale='sv_SE') ]

        print(Model['C1']['Fit'][0], Model['C1']['Fit'][1])
        writer.writerow(UserInfo)
        print('User '+ str(user) + ' Done')



writeFile.close()

#%%
