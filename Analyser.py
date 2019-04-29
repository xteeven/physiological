# %% Imports

from os import listdir
from os.path import isfile, join, dirname, splitext, abspath
from matplotlib import pyplot as plt
from scipy.special import binom
import numpy as np
import pandas as pd
from scipy import interpolate
import psignifit as ps

# %% Set Variables to Search and plot
deltaValues = [-2*0.15, -1*0.15, -0.5*0.15, 0, 0.5*0.15, 1*0.15, 2*0.15]
# [round(float(value.replace(',', '.'))-0.65,3) for value in C2["VariableStiffnes"]
deltaValues2 = [-3*0.15, -2*0.15, -1*0.15, 0, 1*0.15, 2*0.15, 3*0.15]
variableValues = [str(round(0.65+value, 2)) for value in deltaValues]
# list(C2["VariableStiffnes"].head(7).sort_values())
variableValues2 = [str(round(0.65+value, 2)) for value in deltaValues]

xsmooth = np.linspace(min(deltaValues), max(deltaValues), 300)
xsmooth2 = np.linspace(min(deltaValues), max(deltaValues2), 300)

# %% Find The Directory and Open the Files


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

# %% Invert Based on Variable Piston
# First find Trials with VariablePiston 0 (Piston1) and then verify if both sfiffness are different
# i.e Not variable piston 0,65 and Reference 0,65. then invert te ansers.


def invertData(d, c="UserSelection"):
    d.loc[(d["VariablePiston"] == 0) & (d["VariableStiffnes"] != "0,65"), [
        c]] = ~d.loc[(d["VariablePiston"] == 0) & (d["VariableStiffnes"] != "0,65"), [c]]


# %% Get The times the user answered rigth

def GetSucess(Condition):
    SucessVector = []
    _VariableValues = list(Condition["VariableStiffnes"].head(7).sort_values())
    for value in _VariableValues:
        _Stiffness = Condition[Condition.VariableStiffnes == value]
        SucessVector.append((_Stiffness.UserSelection ==
                             _Stiffness.TrueStiff).sum())
    return pd.DataFrame(SucessVector)/5.0

# %% Get The times the user answered true


def GetTrueRate(Condition):
    TrueRate = []
    _VariableValues = list(Condition["VariableStiffnes"].head(7).sort_values())
    for value in _VariableValues:
        _Stiffness = Condition[Condition.VariableStiffnes == value]
        TrueRate.append(_Stiffness.UserSelection.sum())
    return pd.DataFrame(TrueRate)/5.0


def getSpline(input, baseline):
    spline = interpolate.splrep(baseline[0], input, s=0)
    ynew = interpolate.splev(baseline[1], spline, der=0)
    return ynew


def Bernstein(n, k):
    """Bernstein polynomial.
    """
    coeff = binom(n, k)

    def _bpoly(x):
        return coeff * x ** k * (1 - x) ** (n - k)

    return _bpoly


def Bezier(points, num=300):
    """Build Bézier curve from points.
    """
    points = np.array(points)
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for ii in range(N):
        curve += np.outer(Bernstein(N - 1, ii)(t), points[ii])
    return curve


# %%
filescsv = filterByFormat(".csv", openDocuments("Actual"))


# %% Filter By Condition and Save success vector of with corrected answers
axisx = []
plots = []
for file in filescsv:
    readfile = pd.read_csv(file, delimiter=";")
    C2 = readfile[readfile.HringState == "ON"]
    C3 = readfile[readfile.HringState == "RELEASING"]
    conditions = [C2, C3]
    plotcond = []

    for condition in conditions:
        invertData(condition)
        TrueVector = GetTrueRate(condition)
        plotcond.append(TrueVector)

    plots.append(plotcond)
    Normaldots = [round(float(value.replace(',', '.'))-0.65, 3)
                  for value in C2["VariableStiffnes"].head(7).sort_values()]
    xsmooth = np.linspace(min(Normaldots), max(Normaldots), 300)
    axisx.append([Normaldots, xsmooth])
# %% Plot


def setTicks(stepe=0.3):
    plt.yticks(np.arange(0, 1, step=stepe),
               np.round(np.arange(0, 1, step=stepe), 1))


for style in plt.style.available:
    plt.style.use(style)
plt.style.use(plt.style.available[0])

plt.subplot(111).set_aspect(0.5)
setTicks()
plt.title("% of yes answers")
plt.ylabel('% Accuracy')

total = plots[0][0]*0
for plot in plots:
    total+=plot[0]/len(plots)


total2 = Bezier(plots[0][0]*0)
for plot in plots:
    total2+=Bezier(plot[0]/len(plots))

totalc3 = plots[0][0]*0
for plot in plots:
    totalc3+=plot[1]/len(plots)


total2c3 = Bezier(plots[0][0]*0)
for plot in plots:
    total2c3+=Bezier(plot[1]/len(plots))

plt.plot(axisx[0][0],total, 'sc' )

plt.plot(axisx[0][0],totalc3, 'sr')
plot1,_ = plt.plot(axisx[0][1],total2, 'xkcd:blue' , label='Condition 2')
plot2,_ = plt.plot(axisx[0][1],total2c3, 'xkcd:brick' , label='Condition 3')

plt.ylim(-.1,1.1)
plt.legend(handles =[plot1, plot2])
plt.savefig("UsersBezier.png")
plt.xlabel('Stiffness')

# %%
plt.show()

# %% fdsf

YesNo = np.array(total.loc[::-1])
YesNo = YesNo.T[0]
Stimulus = np.array(axisx[0][0])+0.65
Target = np.ones(7)

Datas = np.array([Stimulus, YesNo*100, Target*100]).T

YesNo = np.array(totalc3.loc[::-1])
YesNo = YesNo.T[0]

Datas2 = np.array([Stimulus, YesNo*100, Target*100]).T


options             = dict()
options['sigmoidName'] = 'norm'
options['expType']     = 'YesNo'
result = ps.psignifit(Datas,options);
result2 = ps.psignifit(Datas2,options);
result['Fit']
result['conf_Intervals']
ps.psigniplot.plotPsych(result)
ps.psigniplot.plotPsych(result2)

if __name__ == "__main__":
    pass