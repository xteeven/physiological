# %%
from os import listdir
from os.path import isfile, join, dirname, splitext, abspath
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import psignifit as ps
from matplotlib.ticker import ScalarFormatter 
from sys import getsizeof
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
readfile = pd.read_csv(filescsv[0], delimiter=";")

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
data = np.array([VariableValuesf, TrueVector[::-1], Trials]).T
data2 = np.array([VariableValuesf, TrueVector3[::-1], Trials]).T
# %%
options = dict()
options['sigmoidName'] = 'Weibull'
# %%
with fit(data,options) as res:
    plotPsych(res)
# %%
res=ps.psignifit(data,options)
#res2=ps.psignifit(data2,options)
# %%
# %%
plotPsych(res)
# %%
for style in plt.style.available:
    plt.style.use(style)
plt.style.use(plt.style.available[0])

# %%
del res, res2

# %%

