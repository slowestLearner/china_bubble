#!/usr/bin/env python
# coding: utf-8
from typing import Any, List, Sequence
from statsmodels.iolib.summary import SimpleTable, fmt_params, fmt_2cols
from statsmodels.iolib import summary
from statsmodels.iolib.table import default_txt_fmt
import pandas as pd
import numpy as np

latex_fmt = dict(
    fmt='ltx',
    # basic table formatting
    table_dec_above=r'\toprule',
    table_dec_below=r'\bottomrule',
    header_dec_below=r'\midrule',
    row_dec_below=None,
    strip_backslash=True,  # NotImplemented
    # row formatting
    row_post=r'  \\',
    data_aligns='c',
    colwidths=None,
    colsep=' & ',
    # data formats
    data_fmts=['%s'],
    data_fmt='%s',  # deprecated; use data_fmts
    # labeled alignments
    # stubs_align='l',   # deprecated; use data_fmts
    stub_align='l',
    header_align='c',
    empty_align='l',
    # labeled formats
    header_fmt=r'%s',  # deprecated; just use 'header'
    stub_fmt=r'%s',  # deprecated; just use 'stub'
    empty_cell='',  # deprecated; just use 'empty'
    header=r'%s',
    stub=r'%s',
    empty='',
    missing='--',
    #replacements will be processed in lexicographical order
    replacements={"#" : "\#", "$" : "$", "%" : "\%", "&" : "\&", ">" : "$>$", "_" : "\_", "|" : "$|$"} 
)

def _str(v: float) -> str:
    """Preferred basic formatter"""
    if np.isnan(v):
        return "        "
    av = abs(v)
    digits = 0
    if av != 0:
        digits = int(np.ceil(np.log10(av)))
    if digits > 3 or digits < -3:
        return "{0:8.3g}".format(v)
    
    if digits == -3:
        return "{0:3.2e}".format(v)

    if digits > 0:
        d = int(4 - digits)
    else:
        d = int(3)

    format_str = "{0:" + "0.{0}f".format(d) + "}"
    return format_str.format(v)


def _comma(v) -> str:
    """Split integer with commas"""
    if np.isnan(v):
        return "        "
    return "{0:,}".format(v)

def pval_format(v: float) -> str:
    """Preferred formatting for x in [0,1]"""
    if np.isnan(v):
        return "        "
    return "{0:4.3f}".format(v)



# TODO: typing for Any
def param_table(results: Any, title: str, pad_bottom: bool = False) -> SimpleTable:
    """Formatted standard parameter table"""
    param_data = np.c_[
        np.asarray(results.params)[:, None],
        np.asarray(results.std_errors)[:, None],
        np.asarray(results.tstats)[:, None],
        np.asarray(results.pvalues)[:, None],
        results.conf_int(),
    ]
    data = []
    for row in param_data:
        txt_row = []
        for i, v in enumerate(row):
            func = _str
            if i == 3:
                func = pval_format
            txt_row.append(func(v))
        data.append(txt_row)
    header = ["Parameter", "Std. Err.", "T-stat", "P-value", "Lower CI", "Upper CI"]
    table_stubs = list(results.params.index)
    if pad_bottom:
        # Append blank row for spacing
        data.append([""] * 6)
        table_stubs += [""]

    return SimpleTable(
        data, stubs=table_stubs, txt_fmt=fmt_params, headers=header, title=title
    )


def format_wide(s: Sequence[str], cols: int) -> List[List[str]]:
    """
    Format a list of strings.
    Parameters
    ----------
    s : List[str]
        List of strings to format
    cols : int
        Number of columns in output
    Returns
    -------
    List[List[str]]
        The joined list.
    """
    lines = []
    line = ""
    for i, val in enumerate(s):
        if line == "":
            line = val
            if i + 1 != len(s):
                line += ", "
        else:
            temp = line + val
            if i + 1 != len(s):
                temp += ", "
            if len(temp) > cols:
                lines.append([line])
                line = val
                if i + 1 != len(s):
                    line += ", "
            else:
                line = temp
    lines.append([line])
    return lines


# def add_star(value: str, pvalue: float, star: bool) -> str:
#     """
#     Add 1, 2 or 3 stars to a string base on the p-value
#     Adds 1 star if the pvalue is less than 10%, 2 if less than 5% and 3 is
#     less than 1%.
#     Parameters
#     ----------
#     value : str
#         The formatted parameter value as a string.
#     pvalue : float
#         The p-value of the parameter
#     star : bool
#         Flag indicating whether the star should be added
#     """
#     if not star:
#         return value
#     return value + "*" * sum([pvalue <= c for c in (0.01, 0.05, 0.1)])

def add_star(value: str, tvalue: float, star: bool) -> str:
    """
    Add 1, 2 or 3 stars to a string base on the p-value
    Adds 1 star if the pvalue is less than 10%, 2 if less than 5% and 3 is
    less than 1%.
    Parameters
    ----------
    value : str
        The formatted parameter value as a string.
    tvalue : float
        The cluster t-value of the parameter
    star : bool
        Flag indicating whether the star should be added
    """
    if not star:
        return value
    return value + "*" * sum([np.abs(tvalue) >= c for c in (1.65, 1.96, 2.58)])

def stub_concat(lists: Sequence[Sequence[str]], sep: str = "=") -> List[str]:
    col_size = max([max(map(len, stubs)) for stubs in lists])
    out: List[str] = []
    for stubs in lists:
        out.extend(stubs)
        out.append(sep * (col_size + 2))
    return out


def table_concat(lists: Sequence[List[List[str]]], sep: str = "=") -> List[List[str]]:
    col_sizes = []
    for table in lists:
        #size = list(map(lambda r: list(map(len, r)), table))
        size = list(map(lambda r: list(map(lambda x: len(str(x)), r)), table))
        col_sizes.append(list(np.array(size).max(0)))
    col_size = np.array(col_sizes).max(axis=0)
    sep_cols: List[str] = [sep * (cs + 2) for cs in col_size]
    out: List[List[str]] = []
    for table in lists:
        out.extend(table)
        out.append(sep_cols)
    return out

class Summary(summary.Summary):
    def as_html(self) -> str:
        """
        Return tables as string
        Returns
        -------
        str
            concatenated summary tables in HTML format
        """
        html = summary.summary_return(self.tables, return_fmt="html")
        if self.extra_txt is not None:
            html = html + "<br/><br/>" + self.extra_txt.replace("\n", "<br/>")
        return html
class _SummaryStr(object):
    """
    Mixin class for results classes to automatically show the summary.
    """

    @property
    def summary(self) -> Summary:
        return Summary()

    def __str__(self) -> str:
        return self.summary.as_text()

    def __repr__(self) -> str:
        return (
            self.__str__()
            + "\n"
            + self.__class__.__name__
            + ", id: {0}".format(hex(id(self)))
        )

    def _repr_html_(self) -> str:
        return self.summary.as_html() + "<br/>id: {0}".format(hex(id(self)))


# ## Model Comparison Functions

# In[ ]:

# ## Model Comparison Functions

# In[ ]:


class _ModelComparison(_SummaryStr):
    """
    Base class for model comparisons
    """

    _PRECISION_TYPES = {
        "tstats": "T-stats",
        "pvalues": "P-values",
        "std_errors": "Std. Errors",
    }

    # TODO: Replace Any with better list of types
    def __init__(
        self,
        results,
        params_list,
        model_title,
        investor_FE,
        time_FE,
        if_Interpect_List,
        *,
        summary_type = 'summary',
        precision = "std_errors",
        stars  = False,
       ):
        if not isinstance(results, dict):
            _results = {}
            for i, res in enumerate(results):
                _results["Model " + str(i)] = res
        else:
            _results = {}
            _results.update(results)
        self._results = _results
        
        self._model_title = model_title
        self._investor_FE = investor_FE
        self._time_FE = time_FE
        self._if_Interpect_List = if_Interpect_List            
        self._params_list = params_list
        self._summary_type = summary_type
        precision = precision.lower().replace("-", "_")
        self._precision = precision
        self._stars = stars

    def _get_params_property(self,name):
        out = [
            (k, list(getattr(v, name))) for k, v in self._results.items()
        ]
        cols = [v[0] for v in out]
        values = pd.concat([pd.Series(v[1][0:len(param_name)],index=param_name) for v, param_name in zip(out,self._params_list)], axis=1, sort=False)
        values.columns = cols
        
        if np.any(self._if_Interpect_List):
            values.loc['Intercept', :] = np.nan
            for i in range(len(self._if_Interpect_List)):
                if_intercept = self._if_Interpect_List[i]
                if if_intercept:
                    col_keys = list(self._results.keys())[i]
                    intercepts = getattr(self._results[col_keys], 'intercept') 
                    values.loc['Intercept', col_keys] = intercepts
        return values
    
    def _get_cluster_std_property(self, name = 'cluster_se'):
        # have cluster_se and cluster_std
        out = [(k, list(getattr(v, name))) for k, v in self._results.items()]
        cols = [v[0] for v in out]
        values = pd.Series()
        if np.any(self._if_Interpect_List):
            for i in range(len(self._if_Interpect_List)):
                if_intercept = self._if_Interpect_List[i]
                if if_intercept:
                    col_keys = out[i][0]
                    param_name = self._params_list[i]
                    v = out[i][1]
                    v_series = pd.Series(v[:-1],index=param_name)
                    v_series['Intercept'] = v[-1]
                    values =  pd.concat([values, v_series],axis=1, sort=False)
                else:
                    param_name = self._params_list[i]
                    v = out[i][1]
                    v_series = pd.Series(v,index=param_name)
                    v_series['Intercept'] = np.nan
                    values =  pd.concat([values, v_series],axis=1, sort=False)  
            new_cols = list(values.index)
            new_cols.remove('Intercept')
            new_cols.append('Intercept')
            values = values.loc[new_cols]
        else:
            values = pd.concat([pd.Series(v[1][0:len(param_name)],index=param_name) for v, param_name in zip(out,self._params_list)], axis=1, sort=False)

        if values.shape[1] > len(cols):
            values = values.iloc[:,1:]
        values.columns = cols

        return values
    
    def _get_series_summary_property(self, name):
        if self._summary_type == 'summary':
            out = [(k, getattr(v.summary, name)) for k, v in self._results.items()]
        else:
            out = [(k, v.summary_load[name]) for k, v in self._results.items()]
        cols = [v[0] for v in out]
        values = pd.Series()
        if np.any(self._if_Interpect_List):
            for i in range(len(self._if_Interpect_List)):
                if_intercept = self._if_Interpect_List[i]
                if if_intercept:
                    col_keys = out[i][0]
                    param_name = self._params_list[i]
                    v = out[i][1][0:len(param_name)]
                    v_series = pd.Series(v,index=param_name)
                    v_series['Intercept'] = out[i][1][-1]
                    values =  pd.concat([values, v_series],axis=1, sort=False)
                else:
                    param_name = self._params_list[i]
                    v = out[i][1][0:len(param_name)]
                    v_series = pd.Series(v,index=param_name)
                    v_series['Intercept'] = np.nan
                    values =  pd.concat([values, v_series],axis=1, sort=False)  
            new_cols = list(values.index)
            new_cols.remove('Intercept')
            new_cols.append('Intercept')
            values = values.loc[new_cols]
        else:
            values = pd.concat([pd.Series(v[1][0:len(param_name)],index=param_name) for v, param_name in zip(out,self._params_list)], axis=1, sort=False)

        if values.shape[1] > len(cols):
            values = values.iloc[:,1:]
        values.columns = cols
        return values
    
    
    def _get_summary_property(self, name):
        out = {}
        items = []
        if self._summary_type == 'summary':
            for k, v in self._results.items():
                items.append(k)
                out[k] = getattr(v.summary, name)
        else:
            for k, v in self._results.items():
                items.append(k)
                out[k] = v.summary_load[name]
            
        return pd.Series(out, name=name).loc[items]
    
    
    @property
    def individual_effect(self):
        """Effect for all models"""
        out = {}
        items = []
        
        for k, v in zip(self._results.keys(), self._investor_FE):
            items.append(k)
            out[k] = v
        return pd.Series(out, name='investor_FE').loc[items]
    
    @property
    def time_effect(self):
        """Effect for all models"""
        out = {}
        items = []
        
        for k, v in zip(self._results.keys(), self._time_FE):
            items.append(k)
            out[k] = v
        return pd.Series(out, name='time_FE').loc[items]

    @property
    def nobs(self):
        """Parameters for all models"""
        return self._get_summary_property("numInstances")

    @property
    def params(self):
        """Parameters for all models"""
        return self._get_params_property('coefficients')

    @property
    def tstats_no_adj(self):
        """Parameter t-stats for all models"""
        return self._get_series_summary_property("tValues")
    
    @property
    def tstats(self):
        """Parameter t-stats for all models"""
        return self._get_cluster_std_property('cluster_tstat')

    @property
    def std_errors(self) :
        """Parameter standard errors for all models"""
        return self._get_cluster_std_property('cluster_se')

    @property
    def pvalues(self):
        """Parameter p-vals for all models"""
        return self._get_series_summary_property("pValues")

    @property
    def rsquared(self):
        """Coefficients of determination (R**2)"""
        return self._get_summary_property("r2adj")


# In[ ]:


class PanelModelComparison(_ModelComparison):
    """
    Comparison of multiple models
    Parameters
    ----------
    results : {list, dict}
        Set of results to compare.  If a dict, the keys will be used as model
        names.
    precision : {'tstats','std_errors', 'std-errors', 'pvalues'}
        Estimator precision estimator to include in the comparison output.
        Default is 'tstats'.
    stars : bool
        Add stars based on the p-value of the coefficient where 1, 2 and
        3-stars correspond to p-values of 10%, 5% and 1%, respectively.
    """

    def __init__(
        self,
        results,
        params_list,
        model_title,
        investor_FE,
        time_FE,
        if_Interpect_List,
        individual_count,
        *,
        summary_type = 'summary',
        precision = "std_errors",
        stars = False,
    ) -> None:
        super().__init__(results,params_list,model_title,investor_FE,time_FE, if_Interpect_List,summary_type=summary_type, precision=precision, stars=stars)
        self._individual_count = "{0:,}".format(individual_count)

    @property
    def summary(self):
        """
        Model estimation summary.
        Returns
        -------
        Summary
            Summary table of model estimation results
        Supports export to csv, html and latex  using the methods ``summary.as_csv()``,
        ``summary.as_html()`` and ``summary.as_latex()``.
        """
        smry = Summary()
        models = list(self._results.keys())
        
        title = self._model_title
        
        params = self.params
        precision = getattr(self, self._precision)
        tvalues = np.asarray(self.tstats)
        params_fmt = []
        params_stub = []
        for i in range(len(params)):
            formatted_and_starred = []
            for v, pv in zip(params.values[i], tvalues[i]):
                formatted_and_starred.append(add_star(_str(v), pv, self._stars))
            params_fmt.append(formatted_and_starred)

            precision_fmt = []
            for v in precision.values[i]:
                v_str = _str(v)
                v_str = "({0})".format(v_str) if v_str.strip() else v_str
                precision_fmt.append(v_str)
            params_fmt.append(precision_fmt)
            params_stub.append(params.index[i])
            params_stub.append(" ")
                 
        stubs = [
            "Time Fixed Effects",
            "Investor Fixed Effects",
            "R-Squared",
            "Observations.",
        ]


        vals = pd.concat(
            [
                self.time_effect,
                self.individual_effect,
                self.rsquared,
                self.nobs,
            ],
            1,
        )
        vals = [[i for i in v] for v in vals.T.values]
        vals[2] = [_str(v) for v in vals[2]]
        vals[3] = [_comma(v) for v in vals[3]]
       


        vals = table_concat((params_fmt, vals))
        stubs = stub_concat((params_stub, stubs))
        
        txt_fmt = default_txt_fmt.copy()
        txt_fmt["data_aligns"] = "m"
        txt_fmt["header_align"] = "m"
        table = SimpleTable(
            vals, headers=models, title=title, stubs=stubs, txt_fmt=txt_fmt
        )
        smry.tables.append(table)

        smry.add_extra_txt(["Notes: Unit of observation is individual-by-month. Sample contains {} investor accounts from the Shanghai Stock  Exchange from Jan 2011 to Dec 2019.Standard errors (in parentheses) are double clustered at the individual level and date level. All regressions controlled Number Stocks and HHI up to 3rd polynomials. The ***, **, * denote significance at the 1%, 5%, 10% level correspondingly.".format(self._individual_count)])
        return smry


# In[ ]:


def compare(
    results,
    params_list,
    model_title,
    investor_FE,
    time_FE,
    if_Interpect_List,
    individual_count,
    *,
    summary_type = 'summary',
    precision = "tstats",
    stars = False,
):
    """
    Compare the results of multiple models
    Parameters
    ----------
    results : {list, dict}
        Set of results to compare.  If a dict, the keys will be used as model
        names.
    precision : {'tstats','std_errors', 'std-errors', 'pvalues'}
        Estimator precision estimator to include in the comparison output.
        Default is 'tstats'.
    stars : bool
        Add stars based on the p-value of the coefficient where 1, 2 and
        3-stars correspond to p-values of 10%, 5% and 1%, respectively.
    Returns
    -------
    PanelModelComparison
        The model comparison object.
    """
    return PanelModelComparison(results, params_list,model_title,investor_FE, time_FE,if_Interpect_List,individual_count,summary_type = summary_type , precision=precision, stars=stars)

class _ModelComparison_linear(_SummaryStr):
    """
    Base class for model comparisons
    """

    _PRECISION_TYPES = {
        "tstats": "T-stats",
        "pvalues": "P-values",
        "std_errors": "Std. Errors",
    }

    # TODO: Replace Any with better list of types
    def __init__(
        self,
        results, 
        investor_FE, 
        time_FE,
        *,
        precision = "std_errors",
        stars  = False,
       ):
        if not isinstance(results, dict):
            _results = {}
            for i, res in enumerate(results):
                _results["Model " + str(i)] = res
        else:
            _results = {}
            _results.update(results)
        self._results = _results
        self._investor_FE = investor_FE
        self._time_FE = time_FE
        precision = precision.lower().replace("-", "_")
        self._precision = precision
        self._stars = stars

    def _get_series_property(self, name):
        out = [
            (k, getattr(v, name)) for k, v in self._results.items()
        ]
        cols = [v[0] for v in out]
        values = pd.concat([v[1] for v in out], axis=1, sort=False)
        values.columns = cols
        return values
    
    def _get_property(self, name):
        out = {}
        items = []
        for k, v in self._results.items():
            items.append(k)
            out[k] = getattr(v, name)
        return pd.Series(out, name=name).loc[items]
    
    
    @property
    def nobs(self):
        """Parameters for all models"""
        return self._get_property("nobs")

    @property
    def params(self):
        """Parameters for all models"""
        return self._get_series_property("params")

    @property
    def tstats(self):
        """Parameter t-stats for all models"""
        return self._get_series_property("tstats")

    @property
    def std_errors(self):
        """Parameter standard errors for all models"""
        return self._get_series_property("std_errors")

    @property
    def pvalues(self):
        """Parameter p-vals for all models"""
        return self._get_series_property("pvalues")

    @property
    def rsquared(self):
        """Coefficients of determination (R**2)"""
        return self._get_property("rsquared")
    
    @property
    def individual_effect(self):
        """Effect for all models"""
        out = {}
        items = []
        
        for k, v in zip(self._results.keys(), self._investor_FE):
            items.append(k)
            out[k] = v
        return pd.Series(out, name='investor_FE').loc[items]
    
    @property
    def time_effect(self):
        """Effect for all models"""
        out = {}
        items = []
        
        for k, v in zip(self._results.keys(), self._time_FE):
            items.append(k)
            out[k] = v
        return pd.Series(out, name='time_FE').loc[items]


class PanelModelComparison_Linear(_ModelComparison_linear):
    """
    Comparison of multiple models
    Parameters
    ----------
    results : {list, dict}
        Set of results to compare.  If a dict, the keys will be used as model
        names.
    precision : {'tstats','std_errors', 'std-errors', 'pvalues'}
        Estimator precision estimator to include in the comparison output.
        Default is 'tstats'.
    stars : bool
        Add stars based on the p-value of the coefficient where 1, 2 and
        3-stars correspond to p-values of 10%, 5% and 1%, respectively.
    """

    def __init__(
        self,
        results,
        investor_FE,
        time_FE,
        *,
        precision = "std_errors",
        stars = False,
    ) -> None:
        super().__init__(results, investor_FE, time_FE, precision=precision, stars=stars)
        
    @property
    def rsquared_between(self):
        """Coefficients of determination (R**2)"""
        return self._get_property("rsquared_between")

    @property
    def rsquared_within(self):
        """Coefficients of determination (R**2)"""
        return self._get_property("rsquared_within")

    @property
    def rsquared_overall(self):
        """Coefficients of determination (R**2)"""
        return self._get_property("rsquared_overall")

    @property
    def estimator_method(self):
        """Estimation methods"""
        return self._get_property("name")

    @property
    def cov_estimator(self):
        """Covariance estimator descriptions"""
        return self._get_property("_cov_type")
    
    
    @property
    def summary(self):
        """
        Model estimation summary.
        Returns
        -------
        Summary
            Summary table of model estimation results
        Supports export to csv, html and latex  using the methods ``summary.as_csv()``,
        ``summary.as_html()`` and ``summary.as_latex()``.
        """
        smry = Summary()
        models = list(self._results.keys())
        title = "Model Comparison"
        
        dep_name = {}
        for key in self._results:
            dep_name[key] = self._results[key].model.dependent.vars[0]
        dep_name = pd.Series(dep_name)
        
        params = self.params
        precision = getattr(self, self._precision)
        tvalues = np.asarray(self.tstats)
        params_fmt = []
        params_stub = []
        for i in range(len(params)):
            formatted_and_starred = []
            for v, pv in zip(params.values[i], tvalues[i]):
                formatted_and_starred.append(add_star(_str(v), pv, self._stars))
            params_fmt.append(formatted_and_starred)

            precision_fmt = []
            for v in precision.values[i]:
                v_str = _str(v)
                v_str = "({0})".format(v_str) if v_str.strip() else v_str
                precision_fmt.append(v_str)
            params_fmt.append(precision_fmt)
            params_stub.append(params.index[i])
            params_stub.append(" ")
            
        stubs = [
            "Time Fixed Effects",
            "Stock Fixed Effects",
            "R-Squared",
            "Observations.",
        ]


        vals = pd.concat(
            [
                self.time_effect,
                self.individual_effect,
                self.rsquared,
                self.nobs,
            ],
            axis = 1,
        )
        vals = [[i for i in v] for v in vals.T.values]
        vals[2] = [_str(v) for v in vals[2]]
        vals[3] = [_comma(v) for v in vals[3]]



        vals = table_concat((params_fmt, vals))
        stubs = stub_concat((params_stub, stubs))
        
        txt_fmt = default_txt_fmt.copy()
        txt_fmt["data_aligns"] = "m"
        txt_fmt["header_align"] = "m"
        table = SimpleTable(
            vals, headers=models, title=title, stubs=stubs, txt_fmt=txt_fmt
        )
        smry.tables.append(table)

        return smry

def compare_panel(
    results,
    investor_FE,
    time_FE,
    *,
    precision = "std_errors",
    stars = False,
):
    """
    Compare the results of multiple models
    Parameters
    ----------
    results : {list, dict}
        Set of results to compare.  If a dict, the keys will be used as model
        names.
    precision : {'tstats','std_errors', 'std-errors', 'pvalues'}
        Estimator precision estimator to include in the comparison output.
        Default is 'tstats'.
    stars : bool
        Add stars based on the p-value of the coefficient where 1, 2 and
        3-stars correspond to p-values of 10%, 5% and 1%, respectively.
    Returns
    -------
    PanelModelComparison
        The model comparison object.
    """
    return PanelModelComparison_Linear(results,investor_FE, time_FE, precision=precision, stars=stars)


def as_latex_tabular(compare_results, dep_var_name, individual_count):
    '''Return string, the table as a LaTeX tabular environment.
    Note: will require the booktabs package.'''
    # fetch the text format, override with fmt_dict
    simple_table = compare_results.summary.tables[0]
    fmt = simple_table._get_fmt('latex', **latex_fmt)

    formatted_rows = []
    formatted_rows.append(r'\begin{table}[H]')
    formatted_rows.append( r'\centering' )
    formatted_rows.append( r'\caption{%s}\label{reg:predict}' % simple_table.title)

    table_dec_above = fmt['table_dec_above'] or ''
    table_dec_below = fmt['table_dec_below'] or ''

    total_n = len(simple_table)
    for i in range(total_n):
        row = simple_table[i]
        if i == 0: # first row
            repeat = len(row) - 1
            aligns = row.get_aligns('latex', **fmt)
            formatted_rows.append(r'\begin{tabular}{%s}' % aligns)
            formatted_rows.append(table_dec_above)
            formatted_rows.append(table_dec_above)
            # add dependency variable names
            var_name = ' & ' + dep_var_name
            dep_var = r'\multicolumn{1}{r}{Outcome Var:               }'+ var_name * repeat + r' \\'
            formatted_rows.append(dep_var)
            header_name = ' & '.join(row.data)
            header_row = r'\cmidrule{2-%d}' %(repeat+1) +   header_name  + r' \\'
            formatted_rows.append(header_row)
            formatted_rows.append('\cmidrule{2-%d}' %(repeat+1))
        elif i ==(total_n - 6):
            formatted_rows.append(r'\midrule')
        elif i == (total_n - 1):
            formatted_rows.append(table_dec_below)
            formatted_rows.append(r'\multicolumn{5}{p{35em}}{Notes: Unit of observation is individual-by-month. Sample contains %s investor accounts from the Shanghai Stock  Exchange from Jan 2011 to Dec 2019.Standard errors (in parentheses) are double clustered at the individual  level and date level. All regressions controlled Number Stocks and HHI up to 3rd polynomials. The ***, **, * denote significance at the 1%%, 5%%, 10%%  level correspondingly.}' % "{0:,}".format(individual_count))
            formatted_rows.append(r'\end{tabular}%')
            formatted_rows.append(r'\label{tab:addlabel}%')
            formatted_rows.append(r'\end{table}%')
        else:
            formatted_rows.append(row.as_string(output_format='latex', **fmt))
        
    return '\n'.join(formatted_rows)
