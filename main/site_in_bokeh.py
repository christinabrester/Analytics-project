#----------------------------------------------------------------------------------------------------
#
# A class used to postprocess and visualize the results in Bokeh for one site
#
#----------------------------------------------------------------------------------------------------

from bokeh.models.widgets import Panel, Tabs, Div
from bokeh.plotting import figure
from bokeh.models import FixedTicker, HoverTool, DataTable, BoxAnnotation
from bokeh.layouts import layout, gridplot, Column
from bokeh.models import ColumnDataSource, TableColumn, NumberFormatter, DateFormatter, BoxAnnotation, HTMLTemplateFormatter
import datetime
import pandas as pd
import numpy as np
from metrics import evaluate_forecast

lines_in_bokeh = ['Measured', 'Physical model', 'MLP Baseline', 'MLP Alternative', 'MLP Optimistic']
period = 'test_id'

class SiteInBokeh:
    def __init__(self):
        pass
    
    def name(self, Site):
        return Site.name
    
    def source_outputs(self, Site):
        return ColumnDataSource(data = Site.power_collection)
       
    def x_range(self, Site):
        #START_DATE = '2020-07-01 22:00'
        #END_DATE = '2020-07-03 02:00'
        START_DATE = '2020-05-05 00:00'
        END_DATE = '2020-11-05 00:00'
        
        #return (pd.to_datetime(Site.dates[0], format = '%Y-%m-%d'), 
               #pd.to_datetime(Site.dates[len(Site.dates)-1], format = '%Y-%m-%d'))
        return  (pd.to_datetime(START_DATE, format = '%Y-%m-%d'), 
                       pd.to_datetime(END_DATE, format = '%Y-%m-%d'))
    
    def test_id(self, Site):
        return Site.test_id

    def data_table(self, Site):
        
        data = pd.DataFrame()
        for model in Site.models:
            if data.shape[0] == 0:
                data = pd.DataFrame(evaluate_forecast(Site.power_collection, Site.true_outcome, model, period)['overall'], index = [model])
            else:
                data = pd.concat([data, 
                                  pd.DataFrame(evaluate_forecast(Site.power_collection, Site.true_outcome, model, period)['overall'], 
                                               index = [model])], 
                                 ignore_index = True)

        data = data.T.reset_index()
        data.columns = [''] + lines_in_bokeh[1:]
        
        
        #html_font_template = '<font color="red" face="Verdana, Geneva, sans-serif" size="+1"><%= value %></font>'
        #formatter=HTMLTemplateFormatter(template=html_font_template)


        source_table = ColumnDataSource(data = data)
        
        columns = [TableColumn(field='', title='', width = 70)]
        for i, model in enumerate(lines_in_bokeh[1:]):
            columns.append(TableColumn(field=model, title=model, width=100, formatter=NumberFormatter(format="0,0.000")))

        table = DataTable(source=source_table, columns = columns, fit_columns=True, width = 450, height = 300, row_height = 50, 
                         css_classes=["my_table"], index_position = None)

        style = Div(text="""
        <style>
        .my_table{
        font-size:100% !important; 
        }
        </style>
        """)

        return Column(table, style)
        
        
    def source_group(self, Site, group):
        df = pd.DataFrame(Site.power_collection)

        df = pd.melt(df, id_vars=['UTC'], value_vars=[Site.true_outcome] + Site.models, var_name='group', value_name='Power [W]')

        df_group = df.loc[df.group == group]
        return ColumnDataSource(data = df_group)
    
    def data_table_in_fold(self, Site, period, r):
        
        data = pd.DataFrame()
        
        for model in Site.models:
            if data.shape[0] == 0:
                data = pd.DataFrame(evaluate_forecast(Site.power_collection, Site.true_outcome, model, period)[period])
            else:
                data = pd.concat([data, 
                                  pd.DataFrame(evaluate_forecast(Site.power_collection, Site.true_outcome, model, period)[period])], 
                                  ignore_index = False)
        data = data.loc[r,:]
        data = data.T.reset_index()
        
        data.columns = [''] + lines_in_bokeh[1:]

        source_table = ColumnDataSource(data = data)
        
        columns = [TableColumn(field='', title='', width = 60)]
        for i, model in enumerate(lines_in_bokeh[1:]):
            columns.append(TableColumn(field=model, title=model, width=60, formatter=NumberFormatter(format="0,0.000")))

        return DataTable(source=source_table, columns = columns, fit_columns=True, width = 300, height = 150, index_position = None)



def build_dashboard(sites):

    tabs = []
    colors = ['brown', 'navy', 'green', 'purple', 'gray', 'pink', 'lightblue']
    
    for site in sites:

        site_in_bokeh = SiteInBokeh()
        source_outputs = site_in_bokeh.source_outputs(site)
        data_table = site_in_bokeh.data_table(site)
        x_range = site_in_bokeh.x_range(site)
        test_id = site_in_bokeh.test_id(site)

        p = figure(plot_width=1000, plot_height=300, x_axis_type='datetime', title = 'Power [W]',
                    x_range = x_range, tools = 'reset,wheel_zoom,box_zoom,pan,save', toolbar_location='above')
        
        for i, model in enumerate([site.true_outcome] + site.models[:]):
            source_group = site_in_bokeh.source_group(site, model)

            p.line(x='UTC', y='Power [W]', source=source_group, line_width=2, color=colors[i], legend_label=lines_in_bokeh[i], alpha=0.5)

        hover_tool = HoverTool(tooltips=[('Name','@group'), ('UTC', '@UTC{%Y-%m-%d %H:%M}'), ('Power [W]', '@{Power [W]}{0,0.00}')],
                               formatters={'@UTC': 'datetime'},)
        p.add_tools(hover_tool)

        n_folds = int(np.max(test_id[~np.isnan(test_id)]))

        tables_in_cv = []
        for r in range(n_folds+1):
            data_table_in_fold = site_in_bokeh.data_table_in_fold(site, period, r)
            tables_in_cv.append(data_table_in_fold)
            
            
        div = Div(text="Results in different folds of cross-validation", 
                  style={'font-size': '150%', 'color': 'gray', 'font-family': 'Arial', 'margin': '30px'},
                  width=1000, height=60)
        
        tab = Panel(child=layout([[p, data_table], [div], layout([tables_in_cv])]), title=site.name)
        tabs.append(tab)
        
    return Tabs(tabs=tabs)




