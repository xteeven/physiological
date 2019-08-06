# %%
from os import listdir
from os.path import isfile, join, dirname, splitext, abspath
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import psignifit as ps
from matplotlib.ticker import ScalarFormatter 
from sys import getsizeof, argv
from klepto.archives import file_archive
# %% Find The Directory and Open the Files
class fit(object):
    def __init__(self, data, options):
        self.res = ps.psignifit(data,options)
    def __enter__(self):
        return self.res
    def __exit__(self, type, value, trace_back):
        del self.res

def plotPsych(result,
              dataColor      = [0, 105./255, 170./255],
              plotData       = True,
              lineColor      = [0, 0, 0],
              lineWidth      = 2,
              xLabel         = 'Stimulus Level',
              yLabel         = 'Proportion Correct',
              labelSize      = 15,
              fontSize       = 10,
              fontName       = 'Helvetica',
              tufteAxis      = False,
              plotAsymptote  = True,
              plotThresh     = True,
              aspectRatio    = False,
              extrapolLength = .2,
              CIthresh       = False,
              dataSize       = 0,
              axisHandle     = None,
              showImediate   = True):
    """
    This function produces a plot of the fitted psychometric function with 
    the data.
    """
    
    fit = result['Fit']
    data = result['data']
    options = result['options']
    
    if axisHandle == None: axisHandle = plt.gca()
    try:
        plt.sca(axisHandle)
    except TypeError:
        raise ValueError('Invalid axes handle provided to plot in.')
    
    if np.isnan(fit[3]): fit[3] = fit[2]
    if data.size == 0: return
    if dataSize == 0: dataSize = 10000. / np.sum(data[:,2])
    
    if 'nAFC' in options['expType']:
        ymin = 1. / options['expN']
        ymin = min([ymin, min(data[:,1] / data[:,2])])
    else:
        ymin = 0
    
    
    # PLOT DATA
    #holdState = plt.ishold()
    #if not holdState: 
    #    plt.cla()
    #    plt.hold(True)
    xData = data[:,0]
    if plotData:
        yData = data[:,1] / data[:,2]
        markerSize = np.sqrt(dataSize/2 * data[:,2])
        for i in range(len(xData)):
            plt.plot(xData[i], yData[i], '.', ms=markerSize[i], c=dataColor, clip_on=False)
    
    # PLOT FITTED FUNCTION
    if options['logspace']:
        xMin = np.log(min(xData))
        xMax = np.log(max(xData))
        xLength = xMax - xMin
        x       = np.exp(np.linspace(xMin, xMax, num=1000))
        xLow    = np.exp(np.linspace(xMin - extrapolLength*xLength, xMin, num=100))
        xHigh   = np.exp(np.linspace(xMax, xMax + extrapolLength*xLength, num=100))
        axisHandle.set_xscale('log')
    else:
        xMin = min(xData)
        xMax = max(xData)
        xLength = xMax - xMin
        x       = np.linspace(xMin, xMax, num=1000)
        xLow    = np.linspace(xMin - extrapolLength*xLength, xMin, num=100)
        xHigh   = np.linspace(xMax, xMax + extrapolLength*xLength, num=100)
    
    fitValuesLow  = (1 - fit[2] - fit[3]) * options['sigmoidHandle'](xLow,  fit[0], fit[1]) + fit[3]
    fitValuesHigh = (1 - fit[2] - fit[3]) * options['sigmoidHandle'](xHigh, fit[0], fit[1]) + fit[3]
    fitValues     = (1 - fit[2] - fit[3]) * options['sigmoidHandle'](x,     fit[0], fit[1]) + fit[3]
    
    plt.plot(x,     fitValues,           c=lineColor, lw=lineWidth, clip_on=False)
    plt.plot(xLow,  fitValuesLow,  '--', c=lineColor, lw=lineWidth, clip_on=False)
    plt.plot(xHigh, fitValuesHigh, '--', c=lineColor, lw=lineWidth, clip_on=False)
    
    # PLOT PARAMETER ILLUSTRATIONS
    # THRESHOLD
    if plotThresh:
        if options['logspace']:
            x = [np.exp(fit[0]), np.exp(fit[0])]
        else:
            x = [fit[0], fit[0]]
        y = [ymin, fit[3] + (1 - fit[2] - fit[3]) * options['threshPC']]
        plt.plot(x, y, '-', c=lineColor)
    # ASYMPTOTES
    if plotAsymptote:
        plt.plot([min(xLow), max(xHigh)], [1-fit[2], 1-fit[2]], ':', c=lineColor, clip_on=False)
        plt.plot([min(xLow), max(xHigh)], [fit[3], fit[3]],     ':', c=lineColor, clip_on=False)
    # CI-THRESHOLD
    if CIthresh:
        CIs = result['confIntervals']
        y = np.array([fit[3] + .5*(1 - fit[2] - fit[3]) for i in range(2)])
        plt.plot(CIs[0,:,0],               y,               c=lineColor)
        plt.plot([CIs[0,0,0], CIs[0,0,0]], y + [-.01, .01], c=lineColor)
        plt.plot([CIs[0,1,0], CIs[0,1,0]], y + [-.01, .01], c=lineColor)
    
    #AXIS SETTINGS
    plt.axis('tight')
    plt.tick_params(labelsize=fontSize)
    plt.xlabel(xLabel, fontname=fontName, fontsize=labelSize)
    plt.ylabel(yLabel, fontname=fontName, fontsize=labelSize)
    if aspectRatio: axisHandle.set_aspect(2/(1 + np.sqrt(5)))

    plt.ylim([ymin, 1])
    # tried to mimic box('off') in matlab, as box('off') in python works differently
    plt.tick_params(direction='out',right=False,top=False)
    for side in ['top','right']: axisHandle.spines[side].set_visible(False)
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    plt.ticklabel_format(style= 'sci',scilimits=(-2,4))
    
    #plt.hold(holdState)
    if (showImediate):
        plt.show(0)
    return axisHandle

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


def invertData(d, c="UserSelection"):
    d.loc[(d["VariablePiston"] == 0) & (d["VariableStiffnes"] != "0,65"), [
        c]] = ~d.loc[(d["VariablePiston"] == 0) & (d["VariableStiffnes"] != "0,65"), [c]]

# %% Get The times the user answered true
def GetAxisX(Condition):
    return list(Condition["VariableStiffnes"].head(7).sort_values())

def GetTrueRate(Condition):
    TrueRate = []
    _VariableValues = GetAxisX(Condition)
    for value in _VariableValues:
        _Stiffness = Condition[Condition.VariableStiffnes == value]
        TrueRate.append(_Stiffness.UserSelection.sum())
    return TrueRate

# %%
filescsv = filterByFormat(".csv", openDocuments("Actual"))

# %%
readfile = pd.read_csv(filescsv[int(argv[1])], delimiter=";")

# %%
C2 = readfile[readfile.HringState == "ON"]
C3 = readfile[readfile.HringState == "RELEASING"]
# %%
invertData(C2)
invertData(C3)
# %%
TrueVector = GetTrueRate(C2)
TrueVector3 = GetTrueRate(C3)
# %%
VariableValues = GetAxisX(C2)
VariableValuesf = [round(float(value.replace(',', '.')), 3)
                  for value in VariableValues]

# %%
Trials = np.ones(len(VariableValues))*5

# %%
if int(argv[1])==8:
    data = np.array([VariableValuesf, TrueVector[::], Trials]).T
    data2 = np.array([VariableValuesf, TrueVector3[::], Trials]).T
else:
    data = np.array([VariableValuesf, TrueVector[::-1], Trials]).T
    data2 = np.array([VariableValuesf, TrueVector3[::-1], Trials]).T
# %%
options = dict()
options['sigmoidName'] = 'Weibull'

for style in plt.style.available:
    plt.style.use(style)
plt.style.use(plt.style.available[0])

# %%
# with fit(data,options) as res:
#     plotPsych(res, 
#                 dataColor      = [255./255, 0, 0],
#                 lineColor      = [255./255, 0, 0])
# 
# with fit(data2,options) as res:
#     plotPsych(res, 
#                 dataColor      = [0, 0, 255./255],
#                 lineColor      = [0, 0, 255./255])
#     plt.savefig("Weibull"+argv[1])
# # %%
fittedmodels = []
with fit(data,options) as res:
    fittedmodels.append(res)
    #plotsModelfit(res)
    #plt.savefig("WeibullC2"+argv[1])          
with fit(data2,options) as res:
    fittedmodels.append(res)
    #plotsModelfit(res)
    #plt.savefig("WeibullC3"+argv[1])
db = file_archive("User_"+argv[1]+"_Model.mde")
db['C1'] = fittedmodels[0]
db['C2'] = fittedmodels[1]
db.dump()