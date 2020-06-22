import numpy as np
import pandas as pd
import scipy.stats.stats as stats
import pandas.core.algorithms as algos
from IPython.display import display, HTML, Markdown

# %matplotlib inline
import matplotlib.pyplot as plt


class BinVariable:
    """ BinVariable """
    def __init__(self, var, var_desc, tgt, df):
        self.target = tgt
        self.var_name = var
        self.var_desc = var_desc
        
        self.var_type = ""

        self.data = df.loc[:, [self.var_name, self.target]].copy()
        
        self.Total = self.data.shape[0]
        self.Bads  = self.data.loc[self.data[self.target] == True, ].shape[0]
        self.Goods = self.data.loc[self.data[self.target] == False, ].shape[0]

        self.MinValue = np.nan
        self.AvgValue = np.nan
        self.MedianValue = np.nan
        self.MaxValue = np.nan

        self.IV = np.nan
        self.Gini = np.nan
        self.KS = np.nan
        self.DiversityIndex = np.nan

        self.intervals = pd.DataFrame({}, index=[])

        self.char_rpt_cols = ['BadRate', 'WOE', 'Total', 'Bads', 'DistrBads', 'CumPctBads', 'Goods', 'DistrGoods', 'CumPctGoods']


    def __repr__(self):
        return "Binarized %s Variable %s: IV %f, Gini %f, K-S %f, Diversity %f" % (self.var_type, self.var_name, self.IV, self.Gini, self.KS, self.DiversityIndex)


    def autoBinarize(self):
        # Hare has to be data type specific code from relative class
        self.intervals.index = self.intervals['Interval']

        self.intervals['CumulativeTotal'] = self.intervals['Total'].cumsum()
        self.intervals['CumulativeBads'] = self.intervals['Bads'].cumsum()
        self.intervals['CumulativeGoods'] = self.intervals['Goods'].cumsum()
        
        self.intervals['BadRate'] = self.intervals['Bads']/self.intervals['Total']
        self.intervals['GBOdds'] = self.intervals['Goods']/self.intervals['Bads']
        self.intervals['DistrBads'] = self.intervals['Bads']/self.Bads
        self.intervals['DistrGoods'] = self.intervals['Goods']/self.Goods
        self.intervals['CumPctBads'] = self.intervals['DistrBads'].cumsum()
        self.intervals['CumPctGoods'] = self.intervals['DistrGoods'].cumsum()
        self.intervals['WOE'] = np.log(self.intervals['DistrGoods']/self.intervals['DistrBads'])
        
        self.intervals['tmp_IVPart'] = (self.intervals['DistrGoods'] - self.intervals['DistrBads'])*self.intervals['WOE']
        self.intervals = self.intervals.replace([np.inf, -np.inf, np.Inf, -np.Inf], 0)

        self.IV = self.intervals['tmp_IVPart'].sum()

        self.intervals['tmp_CumBadsPct'] = self.intervals['CumulativeBads']/self.intervals['Bads'].sum()
        self.intervals['tmp_CumGoodsPct'] = self.intervals['CumulativeGoods']/self.intervals['Goods'].sum()
        self.intervals['tmp_CumBadsPct_Shift1'] = self.intervals['tmp_CumBadsPct'].shift(1)
        self.intervals['tmp_CumGoodsPct_Shift1'] = self.intervals['tmp_CumGoodsPct'].shift(1)
        self.intervals.loc[np.isnan(self.intervals['tmp_CumBadsPct_Shift1']), 'tmp_CumBadsPct_Shift1'] = 0
        self.intervals.loc[np.isnan(self.intervals['tmp_CumGoodsPct_Shift1']), 'tmp_CumGoodsPct_Shift1'] = 0
        self.intervals['tmp_square'] = 0.5*(self.intervals['tmp_CumGoodsPct'] + self.intervals['tmp_CumGoodsPct_Shift1'])*(self.intervals['tmp_CumBadsPct'] - self.intervals['tmp_CumBadsPct_Shift1'])

        self.Gini = 100*np.abs((self.intervals['tmp_square'].sum() - 0.5))/0.5
        
        self.intervals['tmp_KS'] = (self.intervals['tmp_CumGoodsPct'] - self.intervals['tmp_CumBadsPct'])
        self.KS = self.intervals['tmp_KS'].abs().max()*100
        
        self.intervals['tmp_diversity'] = self.intervals['Bads']*self.intervals['Goods']/(self.intervals['Total'])
        self.DiversityIndex = 1 - self.intervals['tmp_diversity'].sum()*self.Total/(self.Bads*self.Goods)


    def charasteristicsAnalysisReport(self):
        display(Markdown("## %s Charasteristics Analysis Report" %(self.var_name))) 
        display(Markdown(" *%s* " %(self.var_desc)))
        display(Markdown("**Number of observations** %i, **Bads** %i, **Goods** %i" %(self.Total, self.Bads, self.Goods)))
        if not np.isnan(self.MinValue):
            display(Markdown("**Minimum** %f, **Average** %f, **Median** %f, **Maximum** %f" %(self.MinValue, self.AvgValue, self.MedianValue, self.MaxValue)))
        display(Markdown("**Information Value** is %f" %(self.IV)))
        display(Markdown("**Gini** is %f" %(self.Gini)))
        
        display(Markdown("### Bins: "))
        display(HTML(self.intervals.loc[:, self.char_rpt_cols].to_html()))
        display(self.intervals.loc[:, ['Interval', 'BadRate']].plot.barh())


    def getStatsTable(self):
        return self.intervals.copy()


class BinNumVariable(BinVariable):
    """ BinNumVariable """

    def __init__(self, var, var_desc, tgt, df):
        BinVariable.__init__(self, var, var_desc, tgt, df)
        self.var_type = "Numeric"


    def autoBinarize(self):
        self.MinValue = self.data[self.var_name].min()
        self.AvgValue = self.data[self.var_name].mean()
        self.MedianValue = self.data[self.var_name].median()
        self.MaxValue = self.data[self.var_name].max()
        
        justmiss = self.data.loc[self.data[self.var_name].isnull(), [self.var_name, self.target]]
        notmiss = self.data.loc[self.data[self.var_name].notnull(), [self.var_name, self.target]]
        
        r = 0
        n = 20
        best_r = 0
        best_n = 0
        
        if (notmiss.shape[0] < self.Total*0.005):
            # non-empty records less than 0.5% of Total
            d1 = pd.DataFrame({  "X": notmiss[self.var_name]
                               , "Y": notmiss[self.target]
                               , "Bucket": pd.qcut(notmiss[self.var_name], 1, duplicates='drop')
                              }
                             )
            d2 = d1.groupby('Bucket', as_index=True)
        else:
            while ((np.abs(r) < 0.99999) and (n > 0)):
                try:
                    d1 = pd.DataFrame({  "X": notmiss[self.var_name]
                                       , "Y": notmiss[self.target]
                                       , "Bucket": pd.qcut(notmiss[self.var_name], n, duplicates='drop')
                                      }
                                     )
                    d2 = d1.groupby('Bucket', as_index=True)
                    r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
                    if np.abs(r) > np.abs(best_r):
                        best_r = r
                        best_n = n
                    n = n - 1
                except Exception as e:
                    print("Exception for variable %s step n = %i: %s" %(self.var_name, n, e))
                    n = n - 1

            if len(d2) == 1:
                try:
                    n = best_n
                    bins = algos.quantile(notmiss[self.var_name], np.linspace(0, 1, n))
                    if len(np.unique(bins)) == 2:
                        bins = np.insert(bins, 0, 1)
                        bins[1] = bins[1] - (bins[1]/2)
                    d1 = pd.DataFrame({  "X": notmiss[self.var_name]
                                       , "Y": notmiss[self.target]
                                       , "Bucket": pd.cut(notmiss[self.var_name], np.unique(bins), include_lowest=True)
                                      }
                                     )
                    d2 = d1.groupby('Bucket', as_index=True)
                except Exception as e:
                    print("Exception for variable %s step n = %i: %s" %(self.var_name, n, e))
                    d1 = pd.DataFrame({  "X": notmiss[self.var_name]
                               , "Y": notmiss[self.target]
                               , "Bucket": pd.qcut(notmiss[self.var_name], 1, duplicates='drop')
                              }
                             )
                    d2 = d1.groupby('Bucket', as_index=True)
        
        self.intervals['Variable'] = self.var_name
        self.intervals['MinValue'] = d2.min().X
        self.intervals['MaxValue'] = d2.max().X

        self.intervals['Interval'] = [' - '.join(str(x) for x in y) for y in map(tuple, self.intervals[['MinValue', 'MaxValue']].values)]
        
        self.intervals['Total'] = d2.count().Y
        self.intervals['Bads'] = d2.sum().Y
        self.intervals.loc[np.isnan(self.intervals['Bads']), 'Bads'] = 0
        self.intervals['Goods'] = d2.count().Y - d2.sum().Y
        self.intervals.loc[np.isnan(self.intervals['Goods']), 'Goods'] = 0

        if len(justmiss.index) > 0:
            d4 = pd.DataFrame({'MinValue': np.nan}, index=[0])
            d4['MaxValue'] = np.nan
            d4['Interval'] = "Missing Value"
            d4['Total'] = justmiss.count()[self.target]
            d4['Bads'] = justmiss.sum()[self.target]
            d4['Goods'] = justmiss.count()[self.target] - justmiss.sum()[self.target]
            self.intervals = self.intervals.append(d4, ignore_index=True, sort=True)
            
        # Here has to be the common code from Base class
        BinVariable.autoBinarize(self)


class BinCharVariable(BinVariable):
    """ BinCharVariable """

    def __init__(self, var, var_desc, tgt, df):
        BinVariable.__init__(self, var, var_desc, tgt, df)
        self.var_type = "String"


    def autoBinarize(self):

        self.intervals['Variable'] = self.var_name
        self.intervals['MinValue'] = ""
        self.intervals['MaxValue'] = ""

        self.intervals['Interval'] = ""
        self.intervals['Total'] = self.data.loc[:, self.var_name].value_counts(dropna=False)
        
        self.intervals['Interval'] = self.intervals.index

        self.intervals = pd.merge(  self.intervals
                                  , pd.DataFrame({"Bads": self.data.loc[self.data[self.target] == True, self.var_name].value_counts(dropna=False)})
                                  , how="left"
                                  , left_index=True
                                  , right_index=True
                                 )
        self.intervals.loc[np.isnan(self.intervals['Bads']), 'Bads'] = 0
        
        self.intervals = pd.merge(  self.intervals
                                  , pd.DataFrame({"Goods": self.data.loc[self.data[self.target] == False, self.var_name].value_counts(dropna=False)})
                                  , how="left"
                                  , left_index=True
                                  , right_index=True
                                 )
        self.intervals.loc[np.isnan(self.intervals['Goods']), 'Goods'] = 0
        
        self.intervals.loc[((pd.isnull(self.intervals['Interval'])) | (self.intervals['Interval'].isin([np.nan, "NaN"]))), 'Interval'] = "Missing Value"
        
        # Here has to be the common code from Base class
        BinVariable.autoBinarize(self)
        

class DataSet:
    """ Dataset with data, Good/Bad definition """
    def __init__(self, tgt, df):
        self.target = tgt
        self.data = df.copy()
        self.Goods, self.Bads = self.data.loc[:, tgt].value_counts(dropna=False)
        self.Total = self.Goods + self.Bads
        self.BadRate = 100*self.Bads/self.Total
        
        self.binsList = []
        
        self.statsTable = pd.DataFrame(columns=['Variable', 'IV', 'Gini', 'KS', 'Diversity'])


    def __repr__(self):
        return "DataSet of %i observations. Tatget variable is %s, BadRate %f%% \n %s" % (self.Total, self.target, self.BadRate, self.statsTable)


    def initializeBins(self, objExclusions = [], numExclusions = [], dd = pd.DataFrame({})):
        # Char bins
        tmpList = list(self.data.select_dtypes(include=['object']).columns)
        for x in objExclusions:
            try: 
                tmpList.remove(x)
            except ValueError:
                print("Look like cannot remove column %s, might be not in this DataFrame" %(x))
        for x in tmpList:
            try:
                x_desc = ""
                if ((pd.isnull(dd.loc[x]['Options'])) or (dd.loc[x]['Options'] == "")):
                    x_desc = dd.loc[x]['Description']
                else:
                    x_desc = dd.loc[x]['Description'] + " (" + dd.loc[x]['Options'] + ")"
            except KeyError:
                print("Didn't find description in dictionary")
            
            self.binsList.append(BinCharVariable(x, x_desc, self.target, self.data))

        # Numeric bins
        tmpList = list(self.data.select_dtypes(include=['float', 'integer']).columns)
        for x in numExclusions:
            try:
                tmpList.remove(x)
            except ValueError:
                print("Look like cannot remove column %s, might be not in this DataFrame" %(x))
        for x in tmpList:
            try:
                x_desc = ""
                if ((pd.isnull(dd.loc[x]['Options'])) or (dd.loc[x]['Options'] == "")):
                    x_desc = dd.loc[x]['Description']
                else:
                    x_desc = dd.loc[x]['Description'] + " (" + dd.loc[x]['Options'] + ")"
            except KeyError:
                print("Didn't find description in dictionary")
            
            self.binsList.append(BinNumVariable(x, x_desc, self.target, self.data))
        
        del tmpList


    def printDataSet(self):
        print(self.data.head())


    def getVariablesStatistics(self):
        if self.statsTable.empty:
            tmp_rows_list = []
            for xx in self.binsList:
                if np.isnan(xx.IV):
                    xx.autoBinarize()
                tmp_row = {'Variable': xx.var_name, 'IV': xx.IV, 'Gini': xx.Gini, 'KS': xx.KS, 'Diversity': xx.DiversityIndex}
                tmp_rows_list.append(tmp_row)
                
            self.statsTable = pd.DataFrame(tmp_rows_list)
            del tmp_rows_list


    def characteristicsAnalysisReport(self):
        for xx in self.binsList:
            if np.isnan(xx.IV):
                xx.autoBinarize()
            xx.charasteristicsAnalysisReport()
            plt.show()


    def datasetStatisticsReport(self):
        display(Markdown("## DataSet Variables Statistics Table"))
        display(Markdown("**Number of observations** %i, **Bads** %i, **Goods** %i, **BadRate** %f%%" %(self.Total, self.Bads, self.Goods, self.BadRate)))
        display(Markdown("Tatget variable name is *%s*" %(self.target)))
        display(Markdown("### Predictors Stats: "))
        display(HTML(self.statsTable.to_html()))
