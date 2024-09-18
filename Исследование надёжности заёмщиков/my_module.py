import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from ipywidgets import widgets, Layout
from IPython.display import display, display_html, display_markdown
from tqdm.auto import tqdm
import re
import itertools
from pymystem3 import Mystem
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import plotly.io as pio
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

colorway_for_line = ['rgb(127, 60, 141)', 'rgb(17, 165, 121)', 'rgb(231, 63, 116)',
                     '#03A9F4', 'rgb(242, 183, 1)', '#8B9467', '#FFA07A', '#005A5B', '#66CCCC', '#B690C4'
                     , 'rgb(127, 60, 141)', 'rgb(17, 165, 121)', 'rgb(231, 63, 116)',
                     '#03A9F4', 'rgb(242, 183, 1)', '#8B9467', '#FFA07A', '#005A5B', '#66CCCC', '#B690C4']
colorway_for_bar = ['rgba(128, 60, 170, 0.9)', '#049CB3', '#84a9e9', '#B690C4',
                    '#5c6bc0', '#005A5B', '#63719C', '#03A9F4', '#66CCCC', '#a771f2'
                    , 'rgba(128, 60, 170, 0.9)', '#049CB3', '#84a9e9', '#B690C4',
                    '#5c6bc0', '#005A5B', '#63719C', '#03A9F4', '#66CCCC', '#a771f2'
                    , 'rgba(128, 60, 170, 0.9)', '#049CB3', '#84a9e9', '#B690C4',
                    '#5c6bc0', '#005A5B', '#63719C', '#03A9F4', '#66CCCC', '#a771f2'
                    , 'rgba(128, 60, 170, 0.9)', '#049CB3', '#84a9e9', '#B690C4',
                    '#5c6bc0', '#005A5B', '#63719C', '#03A9F4', '#66CCCC', '#a771f2']
colorway_for_treemap = [
    'rgba(148, 100, 170, 1)',
    'rgba(50, 156, 179, 1)',
    'rgba(99, 113, 156, 1)',
    'rgba(92, 107, 192, 1)',
    'rgba(0, 90, 91, 1)',
    'rgba(3, 169, 244, 1)',
    'rgba(217, 119, 136, 1)',
    'rgba(64, 134, 87, 1)',
    'rgba(134, 96, 147, 1)',
        'rgba(132, 169, 233, 1)']
# default setting for Plotly
# for line plot
pio.templates["custom_theme_for_line"] = go.layout.Template(
    layout=go.Layout(
        colorway=colorway_for_line
    )
)
# pio.templates.default = 'simple_white+custom_theme_for_line'
# for bar plot
pio.templates["custom_theme_for_bar"] = go.layout.Template(
    layout=go.Layout(
        colorway=colorway_for_bar
    )
)
pio.templates.default = 'simple_white+custom_theme_for_bar'

# default setting for Plotly express
px.defaults.template = "simple_white"
px.defaults.color_continuous_scale = color_continuous_scale = [
    [0, 'rgba(0.018, 0.79, 0.703, 1.0)'],
    [0.5, 'rgba(64, 120, 200, 0.9)'],
    [1, 'rgba(128, 60, 170, 0.9)']
]
# px.defaults.color_discrete_sequence = colorway_for_line
px.defaults.color_discrete_sequence = colorway_for_bar
# px.defaults.color_discrete_sequence =  px.colors.qualitative.Bold
# px.defaults.width = 500
# px.defaults.height = 300


def plotly_default_settings(fig):
    # Segoe UI Light
    fig.update_layout(
        title_font=dict(size=24, color="rgba(0, 0, 0, 0.6)"),
        title={'text': f'<b>{fig.layout.title.text}</b>'},
        # Для подписей и меток
        font=dict(size=14, family="Lora", color="rgba(0, 0, 0, 1)"),
        xaxis_title_font=dict(size=18, color="rgba(0, 0, 0, 0.5)"),
        yaxis_title_font=dict(size=18, color="rgba(0, 0, 0, 0.5)"),
        xaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.5)"),
        yaxis_tickfont=dict(size=14, color="rgba(0, 0, 0, 0.5)"),
        xaxis_linecolor="rgba(0, 0, 0, 0.5)",
        # xaxis_linewidth=2,
        yaxis_linecolor="rgba(0, 0, 0, 0.5)",
        # yaxis_linewidth=2
        margin=dict(l=50, r=50, b=50, t=70),
        hoverlabel=dict(bgcolor="white")
    )


def pretty_value(value):
    '''
    Функция делает удобное представление числа с пробелами после разрядов
    '''
    if value == 0:
        return 0
    if value > 0:
        part1 = int(value % 1e3) if value % 1e3 != 0 else ''
        part2 = f'{(value // 1e3) % 1e3:.0f} ' if value // 1e3 != 0 else ''
        part3 = f'{(value // 1e6) % 1e3:.0f} ' if int((value // 1e6) %
                                                      1e3) != 0 else ''
        part4 = f'{(value // 1e9) % 1e3:.0f} ' if int((value // 1e9) %
                                                      1e3) != 0 else ''
        part5 = f'{(value // 1e12) % 1e3:.0f} ' if int((value // 1e12) %
                                                       1e3) != 0 else ''
        return f'{part5}{part4}{part3}{part2}{part1}'
    else:
        value = abs(value)
        part1 = int(value % 1e3) if value % 1e3 != 0 else ''
        part2 = f'{(value // 1e3) % 1e3:.0f} ' if value // 1e3 != 0 else ''
        part3 = f'{(value // 1e6) % 1e3:.0f} ' if int((value // 1e6) %
                                                      1e3) != 0 else ''
        part4 = f'{(value // 1e9) % 1e3:.0f} ' if int((value // 1e9) %
                                                      1e3) != 0 else ''
        part5 = f'{(value // 1e12) % 1e3:.0f} ' if int((value // 1e12) %
                                                       1e3) != 0 else ''
        return f'-{part5}{part4}{part3}{part2}{part1}'


def make_widget_all_frame(df):
    dupl = df.duplicated().sum()
    duplicates = dupl
    if duplicates == 0:
        duplicates = '---'
    else:
        duplicates = pretty_value(duplicates)
        duplicates_pct = dupl * 100 / df.shape[0]
        if 0 < duplicates_pct < 1:
            duplicates_pct = '<1'
        elif duplicates_pct > 99 and duplicates_pct < 100:
            duplicates_pct = round(duplicates_pct, 1)
            if duplicates_pct == 100:
                duplicates_pct = 99.9
        else:
            duplicates_pct = round(duplicates_pct)
        duplicates = f'{duplicates} ({duplicates_pct}%)'
    regex = re.compile(r'\s+')
    dupl_keep_false = df.duplicated(keep=False).sum()
    dupl_sub = df.applymap(lambda x: regex.sub(' ', x.lower().strip()) if isinstance(
        x, str) else x).duplicated(keep=False).sum()
    duplicates_sub_minis_origin = pretty_value(dupl_sub - dupl_keep_false)
    duplicates_sub_minis_origin_pct = (
        dupl_sub - dupl_keep_false) * 100 / dupl
    if 0 < duplicates_sub_minis_origin_pct < 1:
        duplicates_sub_minis_origin_pct = '<1'
    elif (duplicates_sub_minis_origin_pct > 99 and duplicates_sub_minis_origin_pct < 100):
        duplicates_sub_minis_origin_pct = round(
            duplicates_sub_minis_origin_pct, 1)
    else:
        duplicates_sub_minis_origin_pct = round(
            duplicates_sub_minis_origin_pct)
    duplicates_sub_minis_origin = f'{duplicates_sub_minis_origin} ({duplicates_sub_minis_origin_pct}%)'
    all_rows = pd.DataFrame({
        'Rows': [pretty_value(df.shape[0])], 'Features': [df.shape[1]], 'RAM (Mb)': [round(df.__sizeof__() / 1_048_576)], 'Duplicates': [duplicates], 'Dupl (sub - origin)': [duplicates_sub_minis_origin]
    })
    # widget_DataFrame = widgets.Output()
    # with widget_DataFrame:
    #      display_markdown('**DataFrame**', raw=True)
    widget_all_frame = widgets.Output()
    with widget_all_frame:
        # display_html('<h4>DataFrame</h4>', raw=True, height=3)
        display(all_rows.style
                .set_caption('DataFrame')
                .set_table_styles([{'selector': 'caption',
                                    'props': [('font-size', '18px'), ("text-align", "left"), ("font-weight", "bold")]
                                    }])
                .set_properties(**{'text-align': 'left'})
                # .hide_columns()
                .hide_index()
                )
    # widget_DataFrame.layout.margin = '0px 0px 0px 0px'
    return widget_all_frame


def make_widget_range_date(column):
    column_name = column.name
    fist_date = column.min()
    last_date = column.max()
    ram = round(column.__sizeof__() / 1_048_576)
    if ram == 0:
        ram = '<1 Mb'
    column_summary = pd.DataFrame({
        'First date': [fist_date], 'Last date': [last_date], 'RAM (Mb)': [ram]
    })
    widget_summary = widgets.Output()
    with widget_summary:
        # display_html(f'<h4>{column_name}</h4>', raw=True)
        display(column_summary.T.reset_index().style
                .set_caption(f'{column_name}')
                .set_table_styles([{'selector': 'caption',
                                    'props': [('font-size', '16px'), ("text-align", "left"), ("font-weight", "bold")]
                                    }])
                .set_properties(**{'text-align': 'left'})
                .hide_columns()
                .hide_index()
                )
    return widget_summary


def make_widget_summary_date(column):
    column_name = column.name
    values = column.count()
    values = pretty_value(column.count())
    values_pct = column.count() * 100 / column.size
    if 0 < values_pct < 1:
        values_pct = '<1'
    elif values_pct > 99 and values_pct < 100:
        values_pct = round(values_pct, 1)
        if values_pct == 100:
            values_pct = 99.9
    else:
        values_pct = round(values_pct)
    values = f'{values} ({values_pct}%)'
    missing = column.isna().sum()
    if missing == 0:
        missing = '---'
    else:
        missing = pretty_value(column.isna().sum())
        missing_pct = round(column.isna().sum() * 100 / column.size)
        if missing_pct == 0:
            missing_pct = '<1'
        missing = f'{missing} ({missing_pct}%)'
    distinct = pretty_value(column.nunique())
    distinct_pct = column.nunique() * 100 / column.size
    if distinct_pct > 99 and distinct_pct < 100:
        distinct_pct = round(distinct_pct, 1)
        if distinct_pct == 100:
            distinct_pct = 99.9
    else:
        distinct_pct = round(distinct_pct)
    if distinct_pct == 0:
        distinct_pct = '<1'
    distinct = f'{distinct} ({distinct_pct}%)'
    duplicates = column.duplicated().sum()
    if duplicates == 0:
        duplicates = '---'
    else:
        duplicates = pretty_value(duplicates)
        duplicates_pct = column.duplicated().sum() * 100 / column.size
        if 0 < duplicates_pct < 1:
            duplicates_pct = '<1'
        elif duplicates_pct > 99 and duplicates_pct < 100:
            duplicates_pct = round(duplicates_pct, 1)
            if duplicates_pct == 100:
                duplicates_pct = 99.9
        else:
            duplicates_pct = round(duplicates_pct)
        duplicates = f'{duplicates} ({duplicates_pct}%)'
    column_summary = pd.DataFrame({
        'Values': [values], 'Missing': [missing], 'Distinct': [distinct], 'Duplicates': [duplicates]
    })
    widget_summary = widgets.Output()
    with widget_summary:
        # display_html(f'<h4>{column_name}</h4>', raw=True)
        display(column_summary.T.reset_index().style
                # .set_caption(f'{column_name}')
                .set_table_styles([{'selector': 'caption',
                                    'props': [('font-size', '16px'), ("text-align", "left"), ("font-weight", "bold")]
                                    }])
                .set_properties(**{'text-align': 'left'})
                .hide_columns()
                .hide_index()
                )
    return widget_summary


def make_widget_check_missing_date(column):
    column_name = column.name
    fist_date = column.min()
    last_date = column.max()
    date_range = pd.date_range(start=fist_date, end=last_date, freq='D')
    years = date_range.year.unique()
    years_missed_pct = (~years.isin(column.dt.year.unique())
                        ).sum() * 100 / years.size
    if 0 < years_missed_pct < 1:
        years_missed_pct = '<1'
    elif years_missed_pct > 99:
        years_missed_pct = round(years_missed_pct, 1)
    else:
        years_missed_pct = round(years_missed_pct)
    months = date_range.to_period("M").unique()
    months_missed_pct = (~months.isin(column.dt.to_period(
        "M").unique())).sum() * 100 / months.size
    if 0 < months_missed_pct < 1:
        months_missed_pct = '<1'
    elif months_missed_pct > 99:
        months_missed_pct = round(months_missed_pct, 1)
    else:
        months_missed_pct = round(months_missed_pct)
    weeks = date_range.to_period("W").unique()
    weeks_missed_pct = (~weeks.isin(column.dt.to_period(
        "W").unique())).sum() * 100 / weeks.size
    if 0 < weeks_missed_pct < 1:
        weeks_missed_pct = '<1'
    elif weeks_missed_pct > 99:
        weeks_missed_pct = round(weeks_missed_pct, 1)
    else:
        weeks_missed_pct = round(weeks_missed_pct)
    days = date_range.unique().to_period("D")
    days_missed_pct = (~days.isin(column.dt.to_period(
        "D").unique())).sum() * 100 / days.size
    if 0 < days_missed_pct < 1:
        days_missed_pct = '<1'
    elif days_missed_pct > 99:
        days_missed_pct = round(days_missed_pct, 1)
    else:
        days_missed_pct = round(days_missed_pct)

    column_summary = pd.DataFrame({
        'Years missing': [f'{years_missed_pct}%'], 'Months missing': [f'{months_missed_pct}%'], 'Weeks missing': [f'{weeks_missed_pct}%'], 'Days missing': [f'{days_missed_pct}%']
    })
    widget_summary = widgets.Output()
    with widget_summary:
        # display_html(f'<h4>{column_name}</h4>', raw=True)
        display(column_summary.T.reset_index().style
                # .set_caption(f'{column_name}')
                .set_table_styles([{'selector': 'caption',
                                    'props': [('font-size', '16px'), ("text-align", "left"), ("font-weight", "bold")]
                                    }])
                .set_properties(**{'text-align': 'left'})
                .hide_columns()
                .hide_index()
                # .format('{:.0f}', subset=0)
                )
    return widget_summary


def make_widget_summary(column):
    column_name = column.name
    values = column.count()
    values = pretty_value(column.count())
    values_pct = column.count() * 100 / column.size
    if 0 < values_pct < 1:
        values_pct = '<1'
    elif values_pct > 99 and values_pct < 100:
        values_pct = round(values_pct, 1)
        if values_pct == 100:
            values_pct = 99.9
    else:
        values_pct = round(values_pct)
    values = f'{values} ({values_pct}%)'
    missing = column.isna().sum()
    if missing == 0:
        missing = '---'
    else:
        missing = pretty_value(column.isna().sum())
        missing_pct = round(column.isna().sum() * 100 / column.size)
        if missing_pct == 0:
            missing_pct = '<1'
        missing = f'{missing} ({missing_pct}%)'
    distinct = pretty_value(column.nunique())
    distinct_pct = column.nunique() * 100 / column.size
    if distinct_pct > 99 and distinct_pct < 100:
        distinct_pct = round(distinct_pct, 1)
        if distinct_pct == 100:
            distinct_pct = 99.9
    else:
        distinct_pct = round(distinct_pct)
    if distinct_pct == 0:
        distinct_pct = '<1'
    distinct = f'{distinct} ({distinct_pct}%)'
    zeros = ((column == 0) | (column == '')).sum()
    if zeros == 0:
        zeros = '---'
    else:
        zeros = pretty_value(((column == 0) | (column == '')).sum())
        zeros_pct = round(((column == 0) | (column == '')
                           ).sum() * 100 / column.size)
        if zeros_pct == 0:
            zeros_pct = '<1'
        zeros = f'{zeros} ({zeros_pct}%)'
    negative = (column < 0).sum()
    if negative == 0:
        negative = '---'
    else:
        negative = pretty_value(negative)
        negative_pct = round((column < 0).sum() * 100 / column.size)
        if negative_pct == 0:
            negative_pct = '<1'
        negative = f'{negative} ({negative_pct}%)'
    duplicates = column.duplicated().sum()
    if duplicates == 0:
        duplicates = '---'
    else:
        duplicates = pretty_value(duplicates)
        duplicates_pct = column.duplicated().sum() * 100 / column.size
        if 0 < duplicates_pct < 1:
            duplicates_pct = '<1'
        elif duplicates_pct > 99 and duplicates_pct < 100:
            duplicates_pct = round(duplicates_pct, 1)
            if duplicates_pct == 100:
                duplicates_pct = 99.9
        else:
            duplicates_pct = round(duplicates_pct)
        duplicates = f'{duplicates} ({duplicates_pct}%)'
    ram = round(column.__sizeof__() / 1_048_576)
    if ram == 0:
        ram = '<1 Mb'
    column_summary = pd.DataFrame({
        'Values': [values], 'Missing': [missing], 'Distinct': [distinct], 'Duplicates': [duplicates], 'Zeros': [zeros], 'Negative': [negative], 'RAM (Mb)': [ram]
    })
    widget_summary = widgets.Output()
    with widget_summary:
        # display_html(f'<h4>{column_name}</h4>', raw=True)
        display(column_summary.T.reset_index().style
                .set_caption(f'{column_name}')
                .set_table_styles([{'selector': 'caption',
                                    'props': [('font-size', '16px'), ("text-align", "left"), ("font-weight", "bold")]
                                    }])
                .set_properties(**{'text-align': 'left'})
                .hide_columns()
                .hide_index()
                )
    return widget_summary


def make_widget_pct(column):
    max_ = pretty_value(column.max())
    q_95 = pretty_value(column.quantile(0.95))
    q_75 = pretty_value(column.quantile(0.75))
    median_ = pretty_value(column.median())
    q_25 = pretty_value(column.quantile(0.25))
    q_5 = pretty_value(column.quantile(0.5))
    min_ = pretty_value(column.min())
    column_summary = pd.DataFrame({
        'Max': [max_], '95%': [q_95], '75%': [q_75], 'Median': [median_], '25%': [q_25], '5%': [q_5], 'Min': [min_]
    })
    widget_pct = widgets.Output()
    with widget_pct:
        display(column_summary.T.reset_index().style
                .set_caption(f'')
                .set_table_styles([{'selector': 'caption',
                                    'props': [('font-size', '15px')]
                                    }])
                .set_properties(**{'text-align': 'left'})
                .hide_columns()
                .hide_index()
                )
    return widget_pct


def make_widget_std(column):
    avg_ = pretty_value(column.mean())
    mode_ = column.mode()
    if mode_.size > 1:
        mode_ = '---'
    else:
        mode_ = pretty_value(mode_.iloc[0])
    range_ = pretty_value(column.max() - column.min())
    iQR = pretty_value(column.quantile(0.75) - column.quantile(0.25))
    std = pretty_value(column.std())
    kurt = f'{column.kurtosis():.2f}'
    skew = f'{column.skew():.2f}'
    column_summary = pd.DataFrame({
        'Avg': [avg_], 'Mode': [mode_], 'Range': [range_], 'iQR': [iQR], 'std': [std], 'kurt': [kurt], 'skew': [skew]
    })
    widget_std = widgets.Output()
    with widget_std:
        display(column_summary.T.reset_index().style
                .set_caption(f'')
                .set_table_styles([{'selector': 'caption',
                                    'props': [('font-size', '15px')]
                                    }])
                .set_properties(**{'text-align': 'left'})
                .hide_columns()
                .hide_index()
                )
    return widget_std


def make_widget_value_counts(column):
    column_name = column.name
    val_cnt = column.value_counts().iloc[:7]
    val_cnt_norm = column.value_counts(normalize=True).iloc[:7]
    column_name_pct = column_name + '_pct'
    val_cnt_norm.name = column_name_pct

    def make_value_counts_row(x):
        if x[column_name_pct] < 0.01:
            pct_str = '<1%'
        else:
            pct_str = f'({x[column_name_pct]:.0%})'
        return f'{x[column_name]:.0f} {pct_str}'
    top_5 = pd.concat([val_cnt, val_cnt_norm], axis=1).apply(
        make_value_counts_row, axis=1).reset_index()
    widget_value_counts = widgets.Output()
    with widget_value_counts:
        display(top_5.style
                # .set_caption(f'Value counts top')
                .set_table_styles([{'selector': 'caption',
                                    'props': [('font-size', '16px'), ("text-align", "left")]
                                    }])
                .set_properties(**{'text-align': 'left'})
                .hide_columns()
                .hide_index()
                )
    # widget_value_counts.
    return widget_value_counts


def num_trunc(x):
    '''
    Функция обрезает порядок числа и доабвляет К или М
    '''
    if abs(x) < 1e3:
        if round(abs(x), 2) > 0:
            return f'{x:.2f}'
        else:
            return '0'
    if abs(x) < 1e6:
        return f'{x / 1e3:.0f}K'
    if abs(x) < 1e9:
        return f'{x / 1e6:.0f}M'


def make_widget_hist(column):
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    sns.histplot(column, bins=20,  stat='percent', ax=ax, color='#9370db')

    ax.set(ylabel='', xlabel='')
    ax.locator_params(axis='x', nbins=5)
    bins = ax.get_xticks()
    vect = np.vectorize(num_trunc)
    bins = bins[(bins >= column.min()) & (bins <= column.max())]
    ax.set_xticks(ticks=bins, labels=vect(bins))
    plt.xticks(alpha=.9)
    plt.yticks(alpha=.9)
    plt.gca().spines['top'].set_alpha(0.3)
    plt.gca().spines['left'].set_alpha(0.3)
    plt.gca().spines['right'].set_alpha(0.3)
    plt.gca().spines['bottom'].set_alpha(0.3)
    plt.close()
    widget_hist = widgets.Output()
    # fig.tight_layout()
    with widget_hist:
        display(fig)
    return widget_hist


def make_widget_hist_plotly(column):
    fig = px.histogram(column, nbins=20, histnorm='percent',
                       template="simple_white", height=250, width=370)
    fig.update_traces(marker_color='MediumPurple', text=f'*',
                      textfont=dict(color='MediumPurple'))
    fig.update_layout(
        margin=dict(l=0, r=10, b=0, t=10), showlegend=False, hoverlabel=dict(
            bgcolor="white",
        ), xaxis_title="", yaxis_title=""
    )
    # fig.layout.yaxis.visible = False
    widget_hist = widgets.Output()
    with widget_hist:
        fig.show(config=dict(displayModeBar=False), renderer="png")
    return widget_hist


def make_widget_violin(column):
    fig, ax = plt.subplots(figsize=(2, 2.44))
    sns.violinplot(column, ax=ax, color='#9370db')
    ax.set(ylabel='', xlabel='')
    # ax.tick_params(right= False,top= False,left= False, bottom= False)
    # plt.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.gca().spines['top'].set_alpha(0.3)
    plt.gca().spines['left'].set_alpha(0.3)
    plt.gca().spines['right'].set_alpha(0.3)
    plt.gca().spines['bottom'].set_alpha(0.3)
    plt.close()
    widget_violin = widgets.Output()
    with widget_violin:
        display(fig)
    return widget_violin


def make_widget_violine_plotly(column):
    fig = px.violin(column, template="simple_white", height=250, width=300)
    fig.update_traces(marker_color='MediumPurple')
    fig.update_layout(
        margin=dict(l=20, r=20, b=0, t=10), showlegend=False, hoverlabel=dict(
            bgcolor="white",
        ), xaxis_title="", yaxis_title="", xaxis=dict(ticks='', showticklabels=False)
    )
    # fig.layout.yaxis.visible = False
    widget_hist = widgets.Output()
    with widget_hist:
        fig.show(config=dict(displayModeBar=False), renderer="png")
    return widget_hist


def make_widget_summary_obj(column):
    column_name = column.name
    values = column.count()
    values = pretty_value(column.count())
    values_pct = column.count() * 100 / column.size
    if 0 < values_pct < 1:
        values_pct = '<1'
    elif values_pct > 99 and values_pct < 100:
        values_pct = round(values_pct, 1)
        if values_pct == 100:
            values_pct = 99.9
    else:
        values_pct = round(values_pct)
    values = f'{values} ({values_pct}%)'
    missing = column.isna().sum()
    if missing == 0:
        missing = '---'
    else:
        missing = pretty_value(column.isna().sum())
        missing_pct = round(column.isna().sum() * 100 / column.size)
        if missing_pct == 0:
            missing_pct = '<1'
        missing = f'{missing} ({missing_pct}%)'
    distinct = pretty_value(column.nunique())
    distinct_pct = column.nunique() * 100 / column.size
    if distinct_pct > 99 and distinct_pct < 100:
        distinct_pct = round(distinct_pct, 1)
        if distinct_pct == 100:
            distinct_pct = 99.9
    else:
        distinct_pct = round(distinct_pct)
    if distinct_pct == 0:
        distinct_pct = '<1'
    distinct = f'{distinct} ({distinct_pct}%)'
    zeros = ((column == 0) | (column == '')).sum()
    if zeros == 0:
        zeros = '---'
    else:
        zeros = pretty_value(((column == 0) | (column == '')).sum())
        zeros_pct = round(((column == 0) | (column == '')
                           ).sum() * 100 / column.size)
        if zeros_pct == 0:
            zeros_pct = '<1'
        zeros = f'{zeros} ({zeros_pct}%)'
    duplicates = column.duplicated().sum()
    if duplicates == 0:
        duplicates = '---'
        duplicates_sub_minis_origin = '---'
    else:
        duplicates = pretty_value(duplicates)
        duplicates_pct = column.duplicated().sum() * 100 / column.size
        if 0 < duplicates_pct < 1:
            duplicates_pct = '<1'
        elif duplicates_pct > 99 and duplicates_pct < 100:
            duplicates_pct = round(duplicates_pct, 1)
            if duplicates_pct == 100:
                duplicates_pct = 99.9
        else:
            duplicates_pct = round(duplicates_pct)
        duplicates = f'{duplicates} ({duplicates_pct}%)'
        regex = re.compile(r'\s+')
        duplicates_keep_false = column.duplicated(keep=False).sum()
        duplicates_sub = column.apply(lambda x: regex.sub(' ', x.lower(
        ).strip()) if isinstance(x, str) else x).duplicated(keep=False).sum()
        duplicates_sub_minis_origin = duplicates_sub - duplicates_keep_false
        if duplicates_sub_minis_origin == 0:
            duplicates_sub_minis_origin = '---'
        else:
            duplicates_sub_minis_origin = pretty_value(
                duplicates_sub_minis_origin)
            duplicates_sub_minis_origin_pct = (
                duplicates_sub - duplicates_keep_false) * 100 / duplicates_sub
            if 0 < duplicates_sub_minis_origin_pct < 1:
                duplicates_sub_minis_origin_pct = '<1'
            elif (duplicates_sub_minis_origin_pct > 99 and duplicates_sub_minis_origin_pct < 100) \
                    or duplicates_sub_minis_origin_pct < 1:
                duplicates_sub_minis_origin_pct = round(
                    duplicates_sub_minis_origin_pct, 1)
            else:
                duplicates_sub_minis_origin_pct = round(
                    duplicates_sub_minis_origin_pct)
            duplicates_sub_minis_origin = f'{duplicates_sub_minis_origin} ({duplicates_sub_minis_origin_pct}%)'

    ram = round(column.__sizeof__() / 1_048_576)
    if ram == 0:
        ram = '<1 Mb'
    column_summary = pd.DataFrame({
        'Values': [values], 'Missing': [missing], 'Distinct': [distinct], 'Duplicated origin': [duplicates], 'Dupl (modify - origin)': [duplicates_sub_minis_origin], 'Empty': [zeros], 'RAM (Mb)': [ram]
    })
    widget_summary_obj = widgets.Output()
    with widget_summary_obj:
        # display_html(f'<h4>{column_name}</h4>', raw=True)
        display(column_summary.T.reset_index().style
                .set_caption(f'{column_name}')
                .set_table_styles([{'selector': 'caption',
                                    'props': [('font-size', '16px'), ("text-align", "left"), ("font-weight", "bold")]
                                    }])
                .set_properties(**{'text-align': 'left'})
                .hide_columns()
                .hide_index()
                )
    return widget_summary_obj


def make_widget_value_counts_obj(column):
    column_name = column.name
    val_cnt = column.value_counts().iloc[:7]
    val_cnt_norm = column.value_counts(normalize=True).iloc[:7]
    column_name_pct = column_name + '_pct'
    val_cnt_norm.name = column_name_pct

    def make_value_counts_row(x):
        if x[column_name_pct] < 0.01:
            pct_str = '<1%'
        else:
            pct_str = f'({x[column_name_pct]:.0%})'
        return f'{x[column_name]:.0f} {pct_str}'
    top_5 = pd.concat([val_cnt, val_cnt_norm], axis=1).apply(
        make_value_counts_row, axis=1).reset_index()
    widget_value_counts_obj = widgets.Output()
    with widget_value_counts_obj:
        display(top_5.style
                # .set_caption(f'Value counts top')
                .set_table_styles([{'selector': 'caption',
                                    'props': [('font-size', '16px'), ("text-align", "left")]
                                    }])
                .set_properties(**{'text-align': 'left'})
                .hide_columns()
                .hide_index()
                )
    # widget_value_counts.
    return widget_value_counts_obj


def make_widget_bar_obj(column):
    df_fig = column.value_counts(ascending=True).iloc[-10:]
    text_labels = [label[:30] for label in df_fig.index.to_list()]
    fig = px.bar(df_fig, orientation='h',
                 template="simple_white", height=220, width=500)
    fig.update_traces(marker_color='MediumPurple')
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=5), showlegend=False, hoverlabel=dict(
            bgcolor="white",
        ), xaxis_title="", yaxis_title="")
    fig.update_traces(y=text_labels)
    widget_bar_obj = widgets.Output()
    # fig.tight_layout()
    with widget_bar_obj:
        fig.show(config=dict(displayModeBar=False), renderer="png")
    return widget_bar_obj


def make_hbox(widgets_: list):
    # add some CSS styles to distribute free space
    hbox_layout = Layout(display='flex',
                         flex_flow='row',
                         # justify_content='space-around',
                         width='auto',
                         grid_gap='20px',
                         align_items='flex-end'
                         )
    # create Horisontal Box container
    hbox = widgets.HBox(widgets_, layout=hbox_layout)
    return hbox


def my_info(df, graphs=True, num=True, obj=True, date=True):
    '''
    Функция выводить информацию о датафрейме
    Четвертый столбцев (перед графиками) Value counts
    Parameters
    df: pandas.DataFrame
    Датафрейм с данными
    graphs: bool, default True
    Если True, то выводятся графики. 
    '''
    if not num and not obj and not date:
        return
    vbox_layout = Layout(display='flex',
                         # flex_flow='column',
                         justify_content='space-around',
                         # width='auto',
                         # grid_gap = '20px',
                         # align_items = 'flex-end'
                         )
    funcs_num = [make_widget_summary, make_widget_pct,
                 make_widget_std, make_widget_value_counts]
    func_obj = [make_widget_summary_obj, make_widget_value_counts_obj]
    func_date = [make_widget_range_date,
                 make_widget_summary_date, make_widget_check_missing_date]
    if graphs:
        funcs_num += [make_widget_hist_plotly, make_widget_violine_plotly]
        func_obj += [make_widget_bar_obj]
    boxes = []
    if date:
        date_columns = filter(
            lambda x: pd.api.types.is_datetime64_any_dtype(df[x]), df.columns)
        for column in tqdm(date_columns):
            widgets_ = [func(df[column]) for func in func_date]
            boxes.extend((widgets_))
        layout = widgets.Layout(
            grid_template_columns='1fr 1fr 1fr 1fr 1fr')
        date_grid = widgets.GridBox(boxes, layout=layout)
    boxes = []
    if num:
        num_columns = filter(
            lambda x: pd.api.types.is_numeric_dtype(df[x]), df.columns)
        for column in tqdm(num_columns):
            widgets_ = [func(df[column]) for func in funcs_num]
            boxes.extend((widgets_))
        if graphs:
            layout = widgets.Layout(
                grid_template_columns='auto auto auto auto auto auto')
        else:
            layout = widgets.Layout(
                grid_template_columns='repeat(4, 0.2fr)')
        num_grid = widgets.GridBox(boxes, layout=layout)
    boxes = []
    if obj:
        obj_columns = filter(
            lambda x: not pd.api.types.is_numeric_dtype(df[x]) and not pd.api.types.is_datetime64_any_dtype(df[x]), df.columns)
        for column in tqdm(obj_columns):
            widgets_ = [func(df[column]) for func in func_obj]
            boxes.extend((widgets_))
        if graphs:
            layout = widgets.Layout(
                grid_template_columns='auto auto auto')
        else:
            layout = widgets.Layout(
                grid_template_columns='repeat(2, 0.3fr)')
        obj_grid = widgets.GridBox(boxes, layout=layout)

    # widgets.Layout(grid_template_columns="200px 200px 200px 200px 200px 200px")))
    display(make_widget_all_frame(df))
    if date:
        display(date_grid)
    if num:
        display(num_grid)
    if obj:
        display(obj_grid)


def my_info_gen(df, graphs=True, num=True, obj=True, date=True):
    '''
    Генератор выводить информацию о датафрейме
    Четвертый столбцев (перед графиками) Value counts
    Parameters
    df: pandas.DataFrame
    Датафрейм с данными
    graphs: bool, default True
    Если True, то выводятся графики. 
    '''
    if not num and not obj and not date:
        return
    vbox_layout = Layout(display='flex',
                         # flex_flow='column',
                         justify_content='space-around',
                         # width='auto',
                         # grid_gap = '20px',
                         # align_items = 'flex-end'
                         )
    display(make_widget_all_frame(df))
    yield

    funcs_num = [make_widget_summary, make_widget_pct,
                 make_widget_std, make_widget_value_counts]
    func_obj = [make_widget_summary_obj, make_widget_value_counts_obj]
    func_date = [make_widget_range_date,
                 make_widget_summary_date, make_widget_check_missing_date]
    if graphs:
        funcs_num += [make_widget_hist_plotly, make_widget_violine_plotly]
        func_obj += [make_widget_bar_obj]
    if date:
        date_columns = filter(
            lambda x: pd.api.types.is_datetime64_any_dtype(df[x]), df.columns)
        layout = widgets.Layout(
            grid_template_columns='auto auto')
        for column in tqdm(date_columns):
            widgets_ = [func(df[column]) for func in func_date]
            display(widgets.GridBox(widgets_, layout=layout))
            yield
    if num:
        num_columns = filter(
            lambda x: pd.api.types.is_numeric_dtype(df[x]), df.columns)
        if graphs:
            layout = widgets.Layout(
                grid_template_columns='auto auto auto auto auto auto')
        else:
            layout = widgets.Layout(
                grid_template_columns='repeat(4, 0.2fr)')
        for column in tqdm(num_columns):
            widgets_ = [func(df[column]) for func in funcs_num]
            display(widgets.GridBox(widgets_, layout=layout))
            yield
    if obj:
        obj_columns = filter(
            lambda x: not pd.api.types.is_numeric_dtype(df[x]) and not pd.api.types.is_datetime64_any_dtype(df[x]), df.columns)
        if graphs:
            layout = widgets.Layout(
                grid_template_columns='auto auto auto')
        else:
            layout = widgets.Layout(
                grid_template_columns='repeat(2, 0.3fr)')
        for column in tqdm(obj_columns):
            widgets_ = [func(df[column]) for func in func_obj]
            display(widgets.GridBox(widgets_, layout=layout))
            yield


def check_duplicated(df):
    '''
    Функция проверяет датафрейм на дубли.  
    Если дубли есть, то выводит дубли.
    '''
    dupl = df.duplicated().sum()
    if dupl == 0:
        return 'no duplicates'
    print(f'Duplicated is {dupl} rows')
    # приводим строки к нижнему регистру, удаляем пробелы
    regex = re.compile(r'\s+')
    display(df.applymap(lambda x: regex.sub(' ', x.lower().strip()) if isinstance(x, str) else x)
            .value_counts(dropna=False)
            .to_frame()
            .sort_values(0, ascending=False)
            .rename(columns={0: 'Count'})
            .head(10))


def check_duplicated_combinations(df, n):
    '''
    Функция считает дубликаты между всеми возможными комбинациями между столбцами.
    Сначала для проверки на дубли берутся пары столбцов.  
    Затем по 3 столбца. И так все возможные комибнации.  
    Можно выбрать до какого количества комбинаций двигаться.
    n - максимальное возможное количество столбцов в комбинациях
    '''
    if n < 2:
        return
    regex = re.compile(r'\s+')
    df_copy = df.applymap(lambda x: regex.sub(
        ' ', x.lower().strip()) if isinstance(x, str) else x)
    c2 = itertools.combinations(df.columns, 2)
    dupl_df_c2 = pd.DataFrame([], index=df.columns, columns=df.columns)
    print(f'Group by 2 columns')
    for c in c2:
        duplicates = df_copy[list(c)].duplicated().sum()
        dupl_df_c2.loc[c[1], c[0]] = duplicates
    display(dupl_df_c2.fillna('').style.set_caption('Duplicates').set_table_styles([{'selector': 'caption',
                                                                                     'props': [('font-size', '18px'), ("text-align", "left"), ("font-weight", "bold")]
                                                                                     }]))
    if n < 3:
        return
    c3 = itertools.combinations(df.columns, 3)
    dupl_c3_list = []
    print(f'Group by 3 columns')
    for c in c3:
        duplicates = df_copy[list(c)].duplicated().sum()
        if duplicates:
            dupl_c3_list.append([' | '.join(c), duplicates])
    dupl_df_c3 = pd.DataFrame(dupl_c3_list)
    # разобьем таблицу на 3 части, чтобы удобнее читать
    display(pd.concat([part_df.reset_index(drop=True) for part_df in np.array_split(dupl_df_c3, 3)], axis=1)
            .style.format({1: '{:.0f}'}, na_rep='').hide_index().hide_columns())
    if n < 4:
        return
    for col_n in range(4, df.columns.size + 1):
        print(f'Group by {col_n} columns')
        cn = itertools.combinations(df.columns, col_n)
        dupl_cn_list = []
        for c in cn:
            duplicates = df_copy[list(c)].duplicated().sum()
            if duplicates:
                dupl_cn_list.append([' | '.join(c), duplicates])
        dupl_df_cn = pd.DataFrame(dupl_cn_list)
        # разобьем таблицу на 3 части, чтобы удобнее читать
        display(pd.concat([part_df.reset_index(drop=True) for part_df in np.array_split(dupl_df_cn, 2)], axis=1)
                .style.format({1: '{:.0f}'}, na_rep='').hide_index().hide_columns())
        if n < col_n+1:
            return


def find_columns_with_missing_values(df) -> pd.Series:
    '''
    Фукнция проверяет каждый столбец в таблице,  
    если есть пропуски, то помещает строки исходного 
    дата фрейма с этими пропусками в Series. 
    Индекс - название колонки. 
    Если нужно соеденить фреймы в один, то используем 
    pd.concat(res.to_list())
    '''
    dfs_na = pd.Series(dtype=int)
    for col in df.columns:
        is_na = df[col].isna()
        if is_na.any():
            dfs_na[col] = df[is_na]
    return dfs_na


def check_na_in_both_columns(df, cols: list) -> pd.DataFrame:
    '''
    Фукнция проверяет есть ли пропуски одновременно во всех указанных столбцах
    и возвращает датафрейм только со строками, в которых пропуски одновременно во всех столбцах
    '''
    mask = df[cols].isna().all(axis=1)
    return df[mask]


def get_missing_value_proportion_by_category(df: pd.DataFrame, column_with_missing_values: str, category_column: str) -> pd.DataFrame:
    """
    Return a DataFrame with the proportion of missing values for each category.

    Parameters:
    df (pd.DataFrame): Input DataFrame
    column_with_missing_values (str): Column with missing values
    category_column (str): Category column

    Returns:
    pd.DataFrame: DataFrame with the proportion of missing values for each category
    """
    # Create a mask to select rows with missing values in the specified column
    mask = df[column_with_missing_values].isna()

    # Group by category and count the number of rows with missing values
    missing_value_counts = df[mask].groupby(
        category_column).size().reset_index(name='missing_count')
    summ_missing_counts = missing_value_counts['missing_count'].sum()
    # Get the total count for each category
    total_counts = df.groupby(
        category_column).size().reset_index(name='total_count')

    # Merge the two DataFrames to calculate the proportion of missing values
    result_df = pd.merge(missing_value_counts,
                         total_counts, on=category_column)
    result_df['missing_value_in_category_pct'] = (
        result_df['missing_count'] / result_df['total_count']).apply(lambda x: f'{x:.1%}')
    result_df['missing_value_in_column_pct'] = (
        result_df['missing_count'] / summ_missing_counts).apply(lambda x: f'{x:.1%}')
    # Return the result DataFrame
    return result_df[[category_column, 'total_count', 'missing_count', 'missing_value_in_category_pct', 'missing_value_in_column_pct']]


def get_duplicates_value_proportion_by_category(df: pd.DataFrame, column_with_dublicated_values: str, category_column: str) -> pd.DataFrame:
    """
    Return a DataFrame with the proportion of duplicated values for each category.

    Parameters:
    df (pd.DataFrame): Input DataFrame
    column_with_dublicated_values (str): Column with dublicated values
    category_column (str): Category column

    Returns:
    pd.DataFrame: DataFrame with the proportion of missing values for each category
    """
    # Create a mask to select rows with dublicated values in the specified column
    mask = df[column_with_dublicated_values].duplicated()

    # Group by category and count the number of rows with dublicated values
    dublicated_value_counts = df[mask].groupby(
        category_column).size().reset_index(name='dublicated_count')
    summ_dublicated_value_counts = dublicated_value_counts['dublicated_count'].sum(
    )
    # Get the total count for each category
    total_counts = df.groupby(
        category_column).size().reset_index(name='total_count')

    # Merge the two DataFrames to calculate the proportion of dublicated values
    result_df = pd.merge(dublicated_value_counts,
                         total_counts, on=category_column)
    result_df['dublicated_value_in_category_pct'] = (
        result_df['dublicated_count'] / result_df['total_count']).apply(lambda x: f'{x:.1%}')
    result_df['dublicated_value_in_column_pct'] = (
        result_df['dublicated_count'] / summ_dublicated_value_counts).apply(lambda x: f'{x:.1%}')
    # Return the result DataFrame
    return result_df[[category_column, 'total_count', 'dublicated_count', 'dublicated_value_in_category_pct', 'dublicated_value_in_column_pct']]


def check_or_fill_missing_values(df, target_column, identifier_columns, check=True):
    """
    Fill missing values in the target column by finding matching rows without missing values
    in the identifier columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    target_column (str): The column with missing values to be filled.
    identifier_columns (list of str): The columns that uniquely identify the rows.
    check: Is check or fill, default True

    Returns:
    pd.DataFrame: The input DataFrame with missing values filled in the target column.
    """
    # Identify rows with missing values in the target column
    missing_rows = df[df[target_column].isna()]

    # Extract unique combinations of identifying columns from the rows with missing values
    unique_identifiers = missing_rows[identifier_columns].drop_duplicates()

    # Find matching rows without missing values in the target column
    df_unique_identifiers_for_compare = df[identifier_columns].set_index(
        identifier_columns).index
    unique_identifiers_for_comapre = unique_identifiers.set_index(
        identifier_columns).index
    matching_rows = df[df_unique_identifiers_for_compare.isin(unique_identifiers_for_comapre) &
                       (~df['total_income'].isna())]
    # Check if there are matching rows without missing values
    if not matching_rows.empty:
        if check:
            print(
                f'Found {matching_rows.shape[0]} matching rows without missing values')
            return
        # Replace missing values with values from matching rows
        df.loc[missing_rows.index,
               target_column] = matching_rows[target_column].values
        print(
            f'Fiiled {matching_rows.shape[0]} matching rows without missing values')
    else:
        print("No matching rows without missing values found.")


def get_non_matching_rows(df, col1, col2):
    """
    Возвращает строки DataFrame, для которых значения в col1 имеют разные значения в col2.

    Parameters:
    df (pd.DataFrame): DataFrame с данными
    col1 (str): Название колонки с значениями, для которых нужно проверить уникальность
    col2 (str): Название колонки с значениями, которые нужно проверить на совпадение

    Returns:
    pd.DataFrame: Строки DataFrame, для которых значения в col1 имеют разные значения в col2
    """
    non_unique_values = df.groupby(col1)[col2].nunique()[lambda x: x > 1].index
    non_matching_rows = df[df[col1].isin(non_unique_values)]
    if non_matching_rows.empty:
        print('Нет строк для которых значения в col1 имеют разные значения в col2')
    else:
        return non_matching_rows


def fill_na_with_function_by_categories(df, category_columns, value_column, func='median'):
    """
    Fills missing values in the value_column with the result of the func function, 
    grouping by the category_columns.

    Parameters:
    df (pandas.DataFrame): DataFrame to fill missing values
    category_columns (list): list of column names to group by
    value_column (str): name of the column to fill missing values
    func (callable or str): function to use for filling missing values 
    (can be a string, e.g. "mean", or a callable function that returns a single number)

    Returns:
    pandas.DataFrame: modified DataFrame with filled missing values
    """
    available_funcs = {'median', 'mean', 'max', 'min'}

    if isinstance(func, str):
        if func not in available_funcs:
            raise ValueError(f"Unknown function: {func}")
        # If func is a string, use the corresponding pandas method
        df[value_column] = df.groupby(category_columns)[value_column].transform(
            lambda x: x.fillna(x.apply(func)))
    else:
        # If func is a callable, apply it to each group of values
        df[value_column] = df.groupby(category_columns)[
            value_column].transform(lambda x: x.fillna(func(x)))
    return df


def detect_outliers_Zscore(df: pd.DataFrame, z_level: float = 3.5) -> pd.Series:
    """
    Detect outliers in a DataFrame using the Modified Z-score method.

    Parameters:
    df (pd.DataFrame): DataFrame to detect outliers in.
    z_level (float, optional): Modified Z-score threshold for outlier detection. Defaults to 3.5.

    Returns:
    pd.Series: Series with column names as indices and outlier DataFrames as values.
    """
    outliers = pd.Series(dtype=object)
    cnt_outliers = pd.Series(dtype=int)
    for col in filter(lambda x: pd.api.types.is_numeric_dtype(df[x]), df.columns):
        median = df[col].median()
        mad = np.median(np.abs(df[col] - median))
        modified_z_scores = 0.6745 * (df[col] - median) / mad
        outliers[col] = df[np.abs(modified_z_scores) > z_level]
        cnt_outliers[col] = outliers[col].shape[0]
    display(cnt_outliers.to_frame().T.style
            .set_caption('Outliers')
            .set_table_styles([{'selector': 'caption',
                                'props': [('font-size', '18px'), ("text-align", "left"), ("font-weight", "bold")]}])
            .hide_index())
    return outliers


def detect_outliers_quantile(df: pd.DataFrame, lower_quantile: float = 0.05, upper_quantile: float = 0.95) -> pd.Series:
    """
    Detect outliers in a DataFrame using quantile-based method.

    Parameters:
    df (pd.DataFrame): DataFrame to detect outliers in.
    lower_quantile (float, optional): Lower quantile threshold for outlier detection. Defaults to 0.25.
    upper_quantile (float, optional): Upper quantile threshold for outlier detection. Defaults to 0.75.

    Returns:
    pd.Series: Series with column names as indices and outlier DataFrames as values.
    """
    outliers = pd.Series(dtype=object)
    cnt_outliers = pd.Series(dtype=int)
    for col in filter(lambda x: pd.api.types.is_numeric_dtype(df[x]), df.columns):
        lower_bound = df[col].quantile(lower_quantile)
        upper_bound = df[col].quantile(upper_quantile)
        outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        cnt_outliers[col] = outliers[col].shape[0]
    display(cnt_outliers.to_frame().T.style
            .set_caption('Outliers')
            .set_table_styles([{'selector': 'caption',
                                'props': [('font-size', '18px'), ("text-align", "left"), ("font-weight", "bold")]}])
            .hide_index())
    return outliers


def fill_missing_values_using_helper_column(df, categorical_column, helper_column):
    """
    Заполнить пропуски в категориальной переменной на основе значений другой переменной.

    Parameters:
    df (pd.DataFrame): Исходная таблица.
    categorical_column (str): Имя категориальной переменной с пропусками.
    helper_column (str): Имя переменной без пропусков, используемой для заполнения пропусков.

    Returns:
    pd.DataFrame: Таблица с заполненными пропусками.
    """
    # Создать таблицу справочника с уникальными значениями helper_column
    helper_df = df[[helper_column, categorical_column]
                   ].drop_duplicates(helper_column)

    # Удалить строки с пропусками в categorical_column
    helper_df = helper_df.dropna(subset=[categorical_column])

    # Создать новую таблицу с заполненными пропусками
    filled_df = df.drop(categorical_column, axis=1)
    filled_df = filled_df.merge(helper_df, on=helper_column, how='left')

    return filled_df


def get_outlier_quantile_proportion_by_category(df: pd.DataFrame, column_with_outliers: str, category_column: str, lower_quantile: float = 0.05, upper_quantile: float = 0.95) -> None:
    """
    Return a DataFrame with the proportion of outliers for each category.

    Parameters:
    df (pd.DataFrame): Input DataFrame
    column_with_outliers (str): Column with outliers
    category_column (str): Category column
    lower_quantile (float): Lower quantile (e.g., 0.25 for 25th percentile)
    upper_quantile (float): Upper quantile (e.g., 0.75 for 75th percentile)

    Returns:
    None
    """
    # Calculate the lower and upper bounds for outliers
    lower_bound = df[column_with_outliers].quantile(lower_quantile)
    upper_bound = df[column_with_outliers].quantile(upper_quantile)

    # Create a mask to select rows with outliers in the specified column
    mask = (df[column_with_outliers] < lower_bound) | (
        df[column_with_outliers] > upper_bound)

    # Group by category and count the number of rows with outliers
    outlier_counts = df[mask].groupby(
        category_column).size().reset_index(name='outlier_count')
    summ_outlier_counts = outlier_counts['outlier_count'].sum()
    # Get the total count for each category
    total_counts = df.groupby(
        category_column).size().reset_index(name='total_count')

    # Merge the two DataFrames to calculate the proportion of outliers
    result_df = pd.merge(outlier_counts,
                         total_counts, on=category_column)
    result_df['outlier_in_category_pct'] = (
        result_df['outlier_count'] / result_df['total_count']).apply(lambda x: f'{x:.1%}')
    result_df['outlier_in_column_pct'] = (
        result_df['outlier_count'] / summ_outlier_counts).apply(lambda x: f'{x:.1%}')
    display(result_df[[category_column, 'total_count', 'outlier_count', 'outlier_in_category_pct', 'outlier_in_column_pct']].style
            .set_caption(f'Outliers in "{column_with_outliers}" by category "{category_column}"')
            .set_table_styles([{'selector': 'caption',
                                'props': [('font-size', '18px'), ("text-align", "left"), ("font-weight", "bold")]}])
            .hide_index())


def get_outlier_proportion_by_category_modified_z_score(df: pd.DataFrame, column_with_outliers: str, category_column: str, threshold: float = 3.5) -> None:
    """
    Return a DataFrame with the proportion of outliers for each category.

    Parameters:
    df (pd.DataFrame): Input DataFrame
    column_with_outliers (str): Column with outliers
    category_column (str): Category column
    threshold (float): Threshold for modified z-score

    Returns:
    None
    """
    # Calculate the median and median absolute deviation (MAD) for the specified column
    median = df[column_with_outliers].median()
    mad = np.median(np.abs(df[column_with_outliers] - median))

    # Create a mask to select rows with outliers in the specified column
    mask = np.abs(
        0.6745 * (df[column_with_outliers] - median) / mad) > threshold

    # Group by category and count the number of rows with outliers
    outlier_counts = df[mask].groupby(
        category_column).size().reset_index(name='outlier_count')
    summ_outlier_counts = outlier_counts['outlier_count'].sum()

    # Get the total count for each category
    total_counts = df.groupby(
        category_column).size().reset_index(name='total_count')

    # Merge the two DataFrames to calculate the proportion of outliers
    result_df = pd.merge(outlier_counts,
                         total_counts, on=category_column)
    result_df['outlier_in_category_pct'] = (
        result_df['outlier_count'] / result_df['total_count']).apply(lambda x: f'{x:.1%}')
    result_df['outlier_in_column_pct'] = (
        result_df['outlier_count'] / summ_outlier_counts).apply(lambda x: f'{x:.1%}')

    display(result_df[[category_column, 'total_count', 'outlier_count', 'outlier_in_category_pct', 'outlier_in_column_pct']].style
            .set_caption(f'Outliers in "{column_with_outliers}" by category "{category_column}"')
            .set_table_styles([{'selector': 'caption',
                                'props': [('font-size', '18px'), ("text-align", "left"), ("font-weight", "bold")]}])
            .hide_index())


def merge_duplicates(df, duplicate_column, merge_functions):
    """
    Объединяет дубли в датафрейме по указанной колонке с помощью функций из словаря.

    Parameters:
    df (pd.DataFrame): датафрейм для объединения дублей
    duplicate_column (str): название колонки с дублями
    merge_functions (dict): словарь с функциями для объединения, где ключ - название колонки, а значение - функция для объединения

    Returns:
    pd.DataFrame: датафрейм с объединенными дублями
    """
    return df.groupby(duplicate_column, as_index=False).agg(merge_functions)


def create_category_column(column, method='custom_intervals', labels=None, n_intervals=None, bins=None, right=True):
    """
    Create a new category column based on the chosen method.

    Parameters:
    - column (pandas Series): input column
    - method (str, optional): either 'custom_intervals' or 'quantiles' (default is 'custom_intervals')
    - labels (list, optional): list of labels for future categories (default is None)
    - n_intervals (int, optional): number of intervals for 'custom_intervals' or 'quantiles' method (default is len(labels) + 1)
    - bins (list, optional): list of bins for pd.cut function (default is None). The length of `bins` should be `len(labels) + 1`.
    - right (bool, optional): Whether to include the rightmost edge or not. Default is True.

    Returns:
    - pandas Series: new category column

    Example:
    ```
    # Create a sample dataframe
    df = pd.DataFrame({'values': np.random.rand(100)})

    # Create a category column using custom intervals
    category_column = create_category_column(df['values'], method='custom_intervals', labels=['low', 'medium', 'high'], n_intervals=3)
    df['category'] = category_column

    # Create a category column using quantiles
    category_column = create_category_column(df['values'], method='quantiles', labels=['Q1', 'Q2', 'Q3', 'Q4'], n_intervals=4)
    df['category_quantile'] = category_column
    ```
    """
    if method == 'custom_intervals':
        if bins is None:
            if n_intervals is None:
                # default number of intervals
                n_intervals = len(labels) + 1 if labels is not None else 10
            # Calculate равные интервалы
            intervals = np.linspace(column.min(), column.max(), n_intervals)
            if labels is None:
                category_column = pd.cut(column, bins=intervals, right=right)
            else:
                category_column = pd.cut(
                    column, bins=intervals, labels=labels, right=right)
        else:
            if labels is None:
                category_column = pd.cut(column, bins=bins, right=right)
            else:
                category_column = pd.cut(
                    column, bins=bins, labels=labels, right=right)
    elif method == 'quantiles':
        if n_intervals is None:
            # default number of intervals
            n_intervals = len(labels) if labels is not None else 10
        if labels is None:
            category_column = pd.qcut(
                column.rank(method='first'), q=n_intervals)
        else:
            category_column = pd.qcut(column.rank(
                method='first'), q=n_intervals, labels=labels)
    else:
        raise ValueError(
            "Invalid method. Choose either 'custom_intervals' or 'quantiles'.")

    return category_column


def lemmatize_column(column):
    """
    Лемматизация столбца с текстовыми сообщениями.

    Parameters:
    column (pd.Series): Колонка для лемматизации.

    Returns:
    pd.Series: Лемматизированная колонка в виде строки.
    """
    m = Mystem()  # Создаем экземпляр Mystem внутри функции

    def lemmatize_text(text):
        """Приведение текста к леммам с помощью библиотеки Mystem."""
        if not text:
            return ''

        try:
            lemmas = m.lemmatize(text)
            return ' '.join(lemmas)
        except Exception as e:
            print(f"Ошибка при лемматизации текста: {e}")
            return ''

    return column.map(lemmatize_text)


def categorize_column_by_lemmatize(column, categorization_dict):
    """
    Категоризация столбца с помощью лемматизации.

    Parameters:
    column (pd.Series): Столбец для категоризации.
    categorization_dict (dict): Словарь для категоризации, где ключи - категории, а значения - списки лемм.

    Returns:
    pd.Series: Категоризированный столбец.

    Пример использования:
    ```
    # Создайте образец dataframe
    data = {'text': ['This is a sample text', 'Another example text', 'This is a test']}
    df = pd.DataFrame(data)

    # Определите словарь категоризации
    categorization_dict = {
        'Sample': ['sample', 'example'],
        'Test': ['test']
    }

    # Вызовите функцию
    categorized_column = categorize_column_by_lemmatize(df['text'], categorization_dict)

    print(categorized_column)
    ```
    """
    if column.empty:
        return pd.Series([])

    m = Mystem()

    def lemmatize_text(text):
        try:
            return m.lemmatize(text)
        except Exception as e:
            print(f"Ошибка при лемматизации текста: {e}")
            return []

    def categorize_text(lemmas):
        for category, category_lemmas in categorization_dict.items():
            if set(lemmas) & set(category_lemmas):
                return category
        return 'Unknown'

    lemmatized_column = column.map(lemmatize_text)
    return lemmatized_column.map(categorize_text)


def target_encoding_linear(df, category_col, value_col, func='mean', alpha=0.1):
    """
    Функция для target encoding.

    Parameters:
    df (pd.DataFrame): Датафрейм с данными.
    category_col (str): Название колонки с категориями.
    value_col (str): Название колонки со значениями.
    func (callable or str): Функция для target encoding (может быть строкой, например "mean", или вызываемой функцией, которая возвращает одно число).
    alpha (float, optional): Параметр регуляризации. Defaults to 0.1.

    Returns:
    pd.Series: Колонка с target encoding.

    Используется линейная регуляризация, x * (1 - alpha) + alpha * np.mean(x)
    Она основана на идее о том, что среднее значение по группе нужно сгладить, добавляя к нему часть среднего значения по всей таблице.
    """
    available_funcs = {'median', 'mean', 'max', 'min', 'std', 'count'}

    if isinstance(func, str):
        if func not in available_funcs:
            raise ValueError(f"Unknown function: {func}")
        # Если func является строкой, используйте соответствующий метод pandas
        encoding = df.groupby(category_col)[value_col].agg(func)
    else:
        # Если func является вызываемым, примените его к каждой группе значений
        encoding = df.groupby(category_col)[value_col].apply(func)

    # Добавляем линейную регуляризацию
    def regularize(x, alpha=alpha):
        return x * (1 - alpha) + alpha * np.mean(x)

    encoding_reg = encoding.apply(regularize)

    # Заменяем категории на средние значения
    encoded_col = df[category_col].map(encoding_reg.to_dict())

    return encoded_col


def target_encoding_bayes(df, category_col, value_col, func='mean', reg_group_size=100):
    """
    Функция для target encoding с использованием байесовского метода регуляризации.

    Parameters:
    df (pd.DataFrame): Датафрейм с данными.
    category_col (str): Название колонки с категориями.
    value_col (str): Название колонки со значениями.
    func (callable or str): Функция для target encoding (может быть строкой, например "mean", или вызываемой функцией, которая возвращает одно число).
    reg_group_size (int, optional): Размер группы регуляризации. Defaults to 10.

    Returns:
    pd.Series: Колонка с target encoding.

    Эта функция использует байесовский метод регуляризации, который основан на идее о том,   
    что среднее значение по группе нужно сгладить, добавляя к нему часть среднего значения по всей таблице,   
    а также учитывая дисперсию значений в группе.
    """
    if reg_group_size <= 0:
        raise ValueError("reg_group_size must be a positive integer")

    available_funcs = {'median', 'mean', 'max', 'min', 'std', 'count'}

    if isinstance(func, str):
        if func not in available_funcs:
            raise ValueError(f"Unknown function: {func}")
        # Если func является строкой, используйте соответствующий метод pandas
        encoding = df.groupby(category_col)[value_col].agg(
            func_val=(func), count=('count'))
    else:
        # Если func является вызываемым, примените его к каждой группе значений
        encoding = df.groupby(category_col)[value_col].agg(
            func_val=(func), count=('count'))

    global_mean = df[value_col].mean()
    # Добавляем байесовскую регуляризацию
    encoding_reg = (encoding['func_val'] * encoding['count'] +
                    global_mean * reg_group_size) / (encoding['count'] + reg_group_size)

    # Заменяем категории на средние значения
    encoded_col = df[category_col].map(encoding_reg.to_dict())

    return encoded_col


def heatmap(df, title='', xtick_text=None, ytick_text=None, xaxis_label=None, yaxis_label=None, width=None, height=None, decimal_places=2, font_size=14):
    """
    Creates a heatmap from a Pandas DataFrame using Plotly.

    Parameters:
    - `df`: The Pandas DataFrame to create the heatmap from.
    - `title`: The title of the heatmap (default is an empty string).
    - `xtick_text`: The custom tick labels for the x-axis (default is None).
    - `ytick_text`: The custom tick labels for the y-axis (default is None).
    - `xaxis_label`: The label for the x-axis (default is None).
    - `yaxis_label`: The label for the y-axis (default is None).
    - `width`: The width of the heatmap (default is None).
    - `height`: The height of the heatmap (default is None).
    - `decimal_places`: The number of decimal places to display in the annotations (default is 2).
    - `font_size`: The font size for the text in the annotations (default is 14).

    Returns:
    - A Plotly figure object representing the heatmap.

    Notes:
    - If `xtick_text` or `ytick_text` is provided, it must have the same length as the number of columns or rows in the DataFrame, respectively.
    - The heatmap is created with a custom colorscale and hover labels.
    - The function returns a Plotly figure object, which can be displayed using `fig.show()`.
    """
    # Create figure
    fig = go.Figure(data=go.Heatmap(
        z=df.values,
        x=df.columns,
        y=df.index,
        xgap=3,
        ygap=3,
        colorscale=[[0, 'rgba(204, 153, 255, 0.1)'], [1, 'rgb(127, 60, 141)']],
        hoverongaps=False,
        hoverinfo="x+y+z",
        hoverlabel=dict(
            bgcolor="white",
            # Increase font size to font_size
            font=dict(color="black", size=font_size)
        )
    ))

    # Create annotations
    center_color_bar = (df.max().max() + df.min().min()) * 0.7
    annotations = [
        dict(
            text=f"{df.values[row, col]:.{decimal_places}f}",
            x=col,
            y=row,
            showarrow=False,
            font=dict(
                color="black" if df.values[row, col] <
                    center_color_bar else "white",
                size=font_size
            )
        )
        for row, col in np.ndindex(df.values.shape)
    ]

    # Update layout
    fig.update_layout(
        title=title,
        annotations=annotations,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )

    # Update axis labels if custom labels are provided
    if xtick_text is not None:
        if len(xtick_text) != len(df.columns):
            raise ValueError(
                "xtick_text must have the same length as the number of columns in the DataFrame")
        fig.update_layout(xaxis=dict(tickvals=range(
            len(xtick_text)), ticktext=xtick_text))

    if ytick_text is not None:
        if len(ytick_text) != len(df.index):
            raise ValueError(
                "ytick_text must have the same length as the number of rows in the DataFrame")
        fig.update_layout(yaxis=dict(tickvals=range(
            len(ytick_text)), ticktext=ytick_text))

    # Update axis labels if custom labels are provided
    if xaxis_label is not None:
        fig.update_layout(xaxis=dict(title=xaxis_label))

    if yaxis_label is not None:
        fig.update_layout(yaxis=dict(title=yaxis_label))

    # Update figure size if custom size is provided
    if width is not None:
        fig.update_layout(width=width)
    if height is not None:
        fig.update_layout(height=height)

    return fig


def plot_feature_importances_classifier(df: pd.DataFrame, target: str):
    """
    Plot the feature importances of a random forest classifier using Plotly Express.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the features and target variable.
    target (str): The name of the target variable column in the DataFrame.

    Returns:
    fig (plotly.graph_objs.Figure): The feature importance plot.

    Notes:
    This function trains a random forest classifier on the input DataFrame, extracts the feature importances,
    and plots them using Plotly Express. 
    """

    # Select numeric columns and the target variable
    num_columns = [
        col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    df_tmp = df[num_columns + [target]].dropna()
    df_features = df_tmp[num_columns]
    target = df_tmp[target]
    # Get the feature names
    features = df_features.columns
    # Normalize the data using Standard Scaler
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_features)

    # Train a random forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(df_scaled, target)

    # Get the feature importances
    importances = clf.feature_importances_
    feature_importances = pd.DataFrame(
        {'Feature': features, 'Importance': importances})

    # Sort the feature importances in descending order
    feature_importances = feature_importances.sort_values(
        'Importance', ascending=False)

    # Create the bar chart
    fig = px.bar(feature_importances, x='Importance', y='Feature',
                 title=f'Feature Importances for {target}')
    fig.update_layout(
        yaxis=dict(categoryorder='total ascending'),
        width=700,  # Set the width of the graph
        height=500,  # Set the height of the graph
        hoverlabel=dict(bgcolor='white'),
        template='simple_white'  # Set the template to simple_white
    )
    # Set the bar color to mediumpurple
    fig.update_traces(marker_color='rgba(128, 60, 170, 0.9)')

    return fig


def plot_feature_importances_regression(df: pd.DataFrame, target: str):
    """
    Plot the feature importances of a random forest regressor using Plotly Express.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the features and target variable.
    target (str): The name of the target variable column in the DataFrame.

    Returns:
    fig (plotly.graph_objs.Figure): The feature importance plot.

    Notes:
    This function trains a random forest regressor on the input DataFrame, extracts the feature importances,
    and plots them using Plotly Express.
    """
    # Select numeric columns and the target variable
    num_columns = [
        col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    df_tmp = df[num_columns].dropna()
    df_features = df_tmp[num_columns].drop(columns=target)
    target_series = df_tmp[target]
    # Get the feature names
    feature_names = df_features.columns

    # Normalize the data using Standard Scaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_features)

    # Train a random forest regressor
    clf = RandomForestRegressor(n_estimators=100, random_state=0)
    clf.fit(scaled_data, target_series)

    # Get the feature importances
    importances = clf.feature_importances_
    feature_importances = pd.DataFrame(
        {'Feature': feature_names, 'Importance': importances})

    # Sort the feature importances in descending order
    feature_importances = feature_importances.sort_values(
        'Importance', ascending=False)

    # Create the bar chart
    fig = px.bar(feature_importances, x='Importance', y='Feature',
                 title=f'Feature Importances for {target}')
    fig.update_layout(
        yaxis=dict(categoryorder='total ascending'),
        width=700,  # Set the width of the graph
        height=500,  # Set the height of the graph
        hoverlabel=dict(bgcolor='white'),
        template='simple_white'  # Set the template to simple_white
    )
    # Set the bar color to mediumpurple
    fig.update_traces(marker_color='rgba(128, 60, 170, 0.9)')

    return fig


def linear_regression_with_vif(df: pd.DataFrame, target_column: str) -> None:
    """
    Perform linear regression with variance inflation factor (VIF) analysis.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing the data.
    target_column (str): Name of the target column.

    Returns:
    None

    Description:
    This function performs linear regression on the input DataFrame with the specified target column.
    It first selects only the numeric columns, drops any rows with missing values, and then splits the data into features (X) and target (y).
    A constant term is added to the independent variables (X) using `sm.add_constant(X)`.
    The function then fits an ordinary least squares (OLS) model with heteroscedasticity-consistent standard errors (HC1).
    The variance inflation factor (VIF) is calculated for each feature, and the results are displayed along with the model summary.
    """
    # Select numeric columns and drop rows with missing values
    num_columns = [
        col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    df_tmp = df[num_columns].dropna()
    if target_column not in df_tmp.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in the DataFrame.")
    # Split data into features (X) and target (y)
    X = df_tmp.drop(columns=target_column)
    y = df_tmp[target_column]

    # Add a constant term to the independent variables
    X = sm.add_constant(X)

    # Fit OLS model with HC1 standard errors
    model = sm.OLS(y, X).fit(cov_type='HC1')

    # Calculate VIF for each feature
    vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    # Create a DataFrame with coefficients and VIF
    res = pd.DataFrame({'Coef': model.params, 'VIF': vif})

    # Display results
    display(res.iloc[1:])  # exclude the constant term
    display(model.summary())


def categorical_heatmap_matrix_gen(df):
    """
    Generate a heatmap matrix for all possible combinations of categorical variables in a dataframe.

    This function takes a pandas DataFrame as input and generates a heatmap matrix for each pair of categorical variables.
    The heatmap matrix is a visual representation of the cross-tabulation of two categorical variables, which can help identify patterns and relationships between them.

    Parameters:
    df (pandas DataFrame): Input DataFrame containing categorical variables.

    Returns:
    None
    """
    # Получаем список категориальных переменных
    categorical_cols = df.select_dtypes(include=['category']).columns

    # Перебираем все возможные комбинации категориальных переменных
    for col1, col2 in itertools.combinations(categorical_cols, 2):
        # Создаем матрицу тепловой карты
        heatmap_matrix = pd.crosstab(df[col1], df[col2])

        # Визуализируем матрицу тепловой карты
        fig = heatmap(
            heatmap_matrix, title=f'Матрица тепловой карты для {col1} и {col2}')
        plotly_default_settings(fig)
        fig.show()
        yield

def treemap_dash(df, columns):
    """
    Создает интерактивный treemap с помощью Dash и Plotly.

    Параметры:
    df (pandas.DataFrame): датафрейм с данными для treemap.
    columns (list): список столбцов, которые будут использоваться для создания treemap.

    Возвращает:
    app (dash.Dash): прилоожение Dash с интерактивным treemap.
    """
    date_columns = filter(
            lambda x: pd.api.types.is_datetime64_any_dtype(df[x]), df.columns)
    app = dash.Dash(__name__)

    app.layout = html.Div([
        dcc.Dropdown(
            id='reorder-dropdown',
            options=[{'label': col, 'value': col} for col in columns],
            value=columns,
            multi=True
        ),
        dcc.Graph(id='treemap-graph')
    ])

    @app.callback(
        Output('treemap-graph', 'figure'),
        [Input('reorder-dropdown', 'value')]
    )
    def update_treemap(value):
        fig = px.treemap(df, path=[px.Constant('All')] + value,
                         color_discrete_sequence=[
                             'rgba(148, 100, 170, 1)',
                             'rgba(50, 156, 179, 1)',
                             'rgba(99, 113, 156, 1)',
                             'rgba(92, 107, 192, 1)',
                             'rgba(0, 90, 91, 1)',
                             'rgba(3, 169, 244, 1)',
                             'rgba(217, 119, 136, 1)',
                             'rgba(64, 134, 87, 1)',
                             'rgba(134, 96, 147, 1)',
                             'rgba(132, 169, 233, 1)'
                         ])
        fig.update_traces(root_color="lightgrey", hovertemplate="<b>%{label}<br>%{value}</b>")
        fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
        fig.update_traces(hoverlabel=dict(bgcolor="white"))
        return fig

    return app

def treemap(df, columns, values=None):
    """
    Creates an interactive treemap using Plotly.

    Parameters:
    df (pandas.DataFrame): dataframe with data for the treemap.
    columns (list): list of columns to use for the treemap.
    values (str): column for values, if None - values  will be calculated as count.
    Returns:
    fig (plotly.graph_objs.Figure): interactive treemap figure.
    """
    fig = px.treemap(df, path=[px.Constant('All')] + columns
                     , values = values
                     , color_discrete_sequence=[
                         'rgba(148, 100, 170, 1)',
                         'rgba(50, 156, 179, 1)',
                         'rgba(99, 113, 156, 1)',
                         'rgba(92, 107, 192, 1)',
                         'rgba(0, 90, 91, 1)',
                         'rgba(3, 169, 244, 1)',
                         'rgba(217, 119, 136, 1)',
                         'rgba(64, 134, 87, 1)',
                         'rgba(134, 96, 147, 1)',
                         'rgba(132, 169, 233, 1)'
                     ])
    fig.update_traces(root_color="silver", hovertemplate="<b>%{label}<br>%{value:.2f}</b>")
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    fig.update_traces(hoverlabel=dict(bgcolor="white"))
    return fig

def treemap_dash(df):
    """
    Создает интерактивный treemap с помощью Dash и Plotly.

    Параметры:
    df (pandas.DataFrame): датафрейм с данными для treemap.
    columns (list): список столбцов, которые будут использоваться для создания treemap.

    Возвращает:
    app (dash.Dash): прилоожение Dash с интерактивным treemap.
    
    ```
    app = treemap_dash(df)
    if __name__ == '__main__':
        app.run_server(debug=True)
    ```
    """
    categroy_columns = [
        col for col in df.columns if pd.api.types.is_categorical_dtype(df[col])]
    app = dash.Dash(__name__)

    app.layout = html.Div([
        dcc.Dropdown(
            id='reorder-dropdown',
            options=[{'label': col, 'value': col} for col in categroy_columns],
            value=categroy_columns[:2],
            multi=True
        ),
        dcc.Graph(id='treemap-graph')
    ])

    @app.callback(
        Output('treemap-graph', 'figure'),
        [Input('reorder-dropdown', 'value')]
    )
    def update_treemap(value):
        fig = px.treemap(df, path=[px.Constant('All')] + value,
                         color_discrete_sequence=[
                             'rgba(148, 100, 170, 1)',
                             'rgba(50, 156, 179, 1)',
                             'rgba(99, 113, 156, 1)',
                             'rgba(92, 107, 192, 1)',
                             'rgba(0, 90, 91, 1)',
                             'rgba(3, 169, 244, 1)',
                             'rgba(217, 119, 136, 1)',
                             'rgba(64, 134, 87, 1)',
                             'rgba(134, 96, 147, 1)',
                             'rgba(132, 169, 233, 1)'
                         ])
        fig.update_traces(root_color="lightgrey", hovertemplate="<b>%{label}<br>%{value}</b>")
        fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
        fig.update_traces(hoverlabel=dict(bgcolor="white"))
        return fig

    return app

def parallel_categories(df, columns):
    """
    Creates an interactive parallel_categories using Plotly.

    Parameters:
    df (pandas.DataFrame): dataframe with data for the parallel_categories.
    columns (list): list of columns to use for the parallel_categories.

    Returns:
    fig (plotly.graph_objs.Figure): interactive parallel_categories figure.
    """
    # Создание значений цвета
    color_values = [1 for _ in range(df.shape[0])]

    # Создание параллельных категорий
    fig = px.parallel_categories(df, dimensions=columns, color=color_values, 
                                 color_continuous_scale = [
                                [0, 'rgba(128, 60, 170, 0.9)'],
                                [1, 'rgba(128, 60, 170, 0.9)']]
    )

    # Скрытие цветовой шкалы
    if fig.layout.coloraxis:
        fig.update_layout(coloraxis_showscale=False)
    else:
        print("Цветовая шкала не существует")

    # Обновление макета
    fig.update_layout(margin=dict(t=50, l=150, r=150, b=25))

    return fig

def parallel_categories_dash(df):
    """
    Creates a Dash application with an interactive parallel_categories using Plotly.

    Parameters:
    df (pandas.DataFrame): dataframe with data for the parallel_categories.

    Returns:
    app (dash.Dash): Dash application with interactive parallel_categories figure.
        
    ```
    app = treemap_dash(df)
    if __name__ == '__main__':
        app.run_server(debug=True)
    ```
    """
    categroy_columns = [
        col for col in df.columns if pd.api.types.is_categorical_dtype(df[col])]
    # Создание Dash-приложения
    app = dash.Dash(__name__)


    app.layout = html.Div([
        # html.H1('Parallel Categories'),
        dcc.Dropdown(
            id='columns-dropdown',
            options=[{'label': col, 'value': col} for col in categroy_columns],
            value=categroy_columns[:2],  # Значение по умолчанию
            multi=True
        ),
        dcc.Graph(id='parallel-categories-graph')
    ])

    # Обновление графика при изменении выбора столбцов
    @app.callback(
        Output('parallel-categories-graph', 'figure'),
        [Input('columns-dropdown', 'value')]
    )
    def update_graph(selected_columns):
        # Создание параллельных категорий
        color_values = [1 for _ in range(df.shape[0])]
        fig = px.parallel_categories(df, dimensions=selected_columns, color=color_values, 
                                    color_continuous_scale = [
                                    [0, 'rgba(128, 60, 170, 0.7)'],
                                    [1, 'rgba(128, 60, 170, 0.7)']]
        )

        # Скрытие цветовой шкалы
        if fig.layout.coloraxis:
            fig.update_layout(coloraxis_showscale=False)
        else:
            print("Цветовая шкала не существует")

        # Обновление макета
        fig.update_layout(margin=dict(t=50, l=150, r=150, b=25))

        return fig

    return app

def sankey(df, columns, values_column=None, func='sum', mode='fig'):
    """
    Создает Sankey-диаграмму

    Parameters:
    df (pandas.DataFrame): входной DataFrame
    columns (list): список столбцов для Sankey-диаграммы

    Returns:
    fig (plotly.graph_objects.Figure): Sankey-диаграмма
    """
    def prepare_data(df, columns, values_column, func):
        """
        Подготавливает данные для Sankey-диаграммы.

        Parameters:
        df (pandas.DataFrame): входной DataFrame
        columns (list): список столбцов для Sankey-диаграммы

        Returns:
        sankey_df (pandas.DataFrame): подготовленный DataFrame для Sankey-диаграммы
        """
        df_in = df.fillna(value={values_column: 0}).copy()
        columns_len = len(columns)
        temp_df = pd.DataFrame()
        if func == 'mode':
            func = lambda x: x.mode().iloc[0] 
        if func == 'range':
            func = lambda x: x.max() - x.min()
        for i in range(columns_len - 1):
            current_columns = columns[i:i+2]
            if values_column:
                df_grouped = df_in[current_columns+[values_column]].groupby(current_columns)[[values_column]].agg(value = (values_column, func)).reset_index()
            else:
                df_grouped = df_in[current_columns].groupby(current_columns).size().reset_index().rename(columns={0: 'value'})
            temp_df = pd.concat([temp_df, df_grouped
                                        .rename(columns={columns[i]: 'source_name', columns[i+1]: 'target_name'})], axis=0)
        sankey_df = temp_df.reset_index(drop=True)
        return sankey_df

    def create_sankey_nodes(sankey_df):
        """
        Создает узлы для Sankey-диаграммы.

        Parameters:
        sankey_df (pandas.DataFrame): подготовленный DataFrame для Sankey-диаграммы
        colors (list): список цветов для узлов

        Returns:
        nodes_with_indexes (dict): словарь узлов с индексами
        node_colors (list): список цветов узлов
        """
        nodes = pd.concat([sankey_df['source_name'], sankey_df['target_name']], axis=0).unique().tolist()
        nodes_with_indexes = {key: [val] for val, key in enumerate(nodes)}
        colors = [
            'rgba(148, 100, 170, 1)',
            'rgba(50, 156, 179, 1)',
            'rgba(99, 113, 156, 1)',
            'rgba(92, 107, 192, 1)',
            'rgba(0, 90, 91, 1)',
            'rgba(3, 169, 244, 1)',
            'rgba(217, 119, 136, 1)',
            'rgba(64, 134, 87, 1)',
            'rgba(134, 96, 147, 1)',
                'rgba(132, 169, 233, 1)']
        node_colors = []
        colors = itertools.cycle(colors)
        for node in nodes_with_indexes.keys():
            color = next(colors)
            nodes_with_indexes[node].append(color)
            node_colors.append(color)
        return nodes_with_indexes, node_colors

    def create_sankey_links(sankey_df, nodes_with_indexes):
        """
        Создает связи для Sankey-диаграммы.

        Parameters:
        sankey_df (pandas.DataFrame): подготовленный DataFrame для Sankey-диаграммы
        nodes_with_indexes (dict): словарь узлов с индексами

        Returns:
        link_color (list): список цветов связей
        """
        link_color = [nodes_with_indexes[source][1].replace(', 1)', ', 0.2)') for source in sankey_df['source_name']]
        return link_color
    sankey_df = prepare_data(df, columns, values_column, func)
    nodes_with_indexes, node_colors = create_sankey_nodes(sankey_df)
    link_color = create_sankey_links(sankey_df, nodes_with_indexes)
    sankey_df['source'] = sankey_df['source_name'].apply(lambda x: nodes_with_indexes[x][0])
    sankey_df['target'] = sankey_df['target_name'].apply(lambda x: nodes_with_indexes[x][0])
    sankey_df['sum_value'] = sankey_df.groupby('source_name')['value'].transform('sum')
    sankey_df['value_percent'] = round(sankey_df['value'] * 100 / sankey_df['sum_value'], 2)
    sankey_df['value_percent'] = sankey_df['value_percent'].apply(lambda x: f"{x}%")
    if mode == 'fig':
        fig = go.Figure(data=[go.Sankey(
            domain = dict(
            x =  [0,1],
            y =  [0,1]
            ),
            orientation = "h",
            valueformat = ".0f",
            node = dict(
            pad = 10,
            thickness = 15,
            line = dict(color = "black", width = 0.1),
            label =  list(nodes_with_indexes.keys()),
            color = node_colors
            ),
            link = dict(
            source = sankey_df['source'],
            target = sankey_df['target'],
            value  = sankey_df['value'],
            label = sankey_df['value_percent'],
            color = link_color
        )
        )])

        layout = dict(
                title = f"Sankey Diagram for {', '.join(columns+[values_column])}" if values_column else
                f"Sankey Diagram for {', '.join(columns)}",
                height = 772,
                font = dict(
                size = 10),)

        fig.update_layout(layout)  
        return fig
    if mode == 'data':
        sankey_dict = {}
        sankey_dict['sankey_df'] = sankey_df
        sankey_dict['nodes_with_indexes'] = nodes_with_indexes
        sankey_dict['node_colors'] = node_colors
        sankey_dict['link_color'] = link_color
        return sankey_dict

def sankey_dash(df):
    """
    Создает Sankey-диаграмму

    Parameters:
    df (pandas.DataFrame): входной DataFrame
    columns (list): список столбцов для Sankey-диаграммы

    Returns:
    app (dash.Dash): Dash application with interactive parallel_categories figure.
        
    ```
    app = sankey_dash(df)
    if __name__ == '__main__':
        app.run_server(debug=True)
    ```
    """
    def prepare_data(df, columns):
        """
        Подготавливает данные для Sankey-диаграммы.

        Parameters:
        df (pandas.DataFrame): входной DataFrame
        columns (list): список столбцов для Sankey-диаграммы

        Returns:
        sankey_df (pandas.DataFrame): подготовленный DataFrame для Sankey-диаграммы
        """
        df_in = df.dropna().copy()
        columns_len = len(columns)
        temp_df = pd.DataFrame()
        for i in range(columns_len - 1):
            current_columns = columns[i:i+2]
            df_grouped = df_in[current_columns].groupby(current_columns).size().reset_index()
            temp_df = pd.concat([temp_df, df_grouped
                                        .rename(columns={columns[i]: 'source_name', columns[i+1]: 'target_name'})], axis=0)
        sankey_df = temp_df.reset_index(drop=True).rename(columns={0: 'value'})
        return sankey_df

    def create_sankey_nodes(sankey_df):
        """
        Создает узлы для Sankey-диаграммы.

        Parameters:
        sankey_df (pandas.DataFrame): подготовленный DataFrame для Sankey-диаграммы
        colors (list): список цветов для узлов

        Returns:
        nodes_with_indexes (dict): словарь узлов с индексами
        node_colors (list): список цветов узлов
        """
        nodes = pd.concat([sankey_df['source_name'], sankey_df['target_name']], axis=0).unique().tolist()
        nodes_with_indexes = {key: [val] for val, key in enumerate(nodes)}
        colors = [
            'rgba(148, 100, 170, 1)',
            'rgba(50, 156, 179, 1)',
            'rgba(99, 113, 156, 1)',
            'rgba(92, 107, 192, 1)',
            'rgba(0, 90, 91, 1)',
            'rgba(3, 169, 244, 1)',
            'rgba(217, 119, 136, 1)',
            'rgba(64, 134, 87, 1)',
            'rgba(134, 96, 147, 1)',
                'rgba(132, 169, 233, 1)']
        node_colors = []
        colors = itertools.cycle(colors)
        for node in nodes_with_indexes.keys():
            color = next(colors)
            nodes_with_indexes[node].append(color)
            node_colors.append(color)
        return nodes_with_indexes, node_colors

    def create_sankey_links(sankey_df, nodes_with_indexes):
        """
        Создает связи для Sankey-диаграммы.

        Parameters:
        sankey_df (pandas.DataFrame): подготовленный DataFrame для Sankey-диаграммы
        nodes_with_indexes (dict): словарь узлов с индексами

        Returns:
        link_color (list): список цветов связей
        """
        link_color = [nodes_with_indexes[source][1].replace(', 1)', ', 0.2)') for source in sankey_df['source_name']]
        return link_color

    categroy_columns = [
        col for col in df.columns if pd.api.types.is_categorical_dtype(df[col])]
    # Создание Dash-приложения
    app = dash.Dash(__name__)
    app.layout = html.Div([
        # html.H1('Parallel Categories'),
        dcc.Dropdown(
            id='columns-dropdown',
            options=[{'label': col, 'value': col} for col in categroy_columns],
            value=categroy_columns[:2],  # Значение по умолчанию
            multi=True
        ),
        dcc.Graph(id='sankey-graph')
    ])

    # Обновление графика при изменении выбора столбцов
    @app.callback(
        Output('sankey-graph', 'figure'),
        [Input('columns-dropdown', 'value')]
    )
    def update_graph(selected_columns):
        # Создание sankey
        if len(selected_columns) < 2:
            selected_columns = categroy_columns[:2]
        sankey_df = prepare_data(df, selected_columns)
        nodes_with_indexes, node_colors = create_sankey_nodes(sankey_df)
        link_color = create_sankey_links(sankey_df, nodes_with_indexes)
        sankey_df['source'] = sankey_df['source_name'].apply(lambda x: nodes_with_indexes[x][0])
        sankey_df['target'] = sankey_df['target_name'].apply(lambda x: nodes_with_indexes[x][0])
        sankey_df['sum_value'] = sankey_df.groupby('source_name')['value'].transform('sum')
        sankey_df['value_percent'] = round(sankey_df['value'] * 100 / sankey_df['sum_value'], 2)
        sankey_df['value_percent'] = sankey_df['value_percent'].apply(lambda x: f"{x}%")
        fig = go.Figure(data=[go.Sankey(
            domain = dict(
            x =  [0,1],
            y =  [0,1]
            ),
            orientation = "h",
            valueformat = ".0f",
            node = dict(
            pad = 10,
            thickness = 15,
            line = dict(color = "black", width = 0.1),
            label =  list(nodes_with_indexes.keys()),
            color = node_colors
            ),
            link = dict(
            source = sankey_df['source'],
            target = sankey_df['target'],
            value  = sankey_df['value'],
            label = sankey_df['value_percent'],
            color = link_color
        )
        )])

        layout = dict(
                title = f"Sankey Diagram for {', '.join(selected_columns)}",
                height = 772,
                font = dict(
                size = 10),)

        fig.update_layout(layout)  

        return fig

    return app




# =========================================================================================
# =========================================================================================
# =========================================================================================
# def prepare_df(config):
#     df = config['df']
#     cat_column = config['cat_column']
#     num_column = config['num_column']
#     func = config.get('func', 'sum')  # default to 'sum' if not provided
#     if func == 'mode':
#         # Обработка случая для моды
#         func = lambda x: x.mode().iloc[0] 
#         return df[[cat_column, num_column]]\
#             .groupby(cat_column) \
#             .agg(num = (num_column, func), modes = (num_column, lambda x: x.mode().to_list())) \
#             .reset_index() \
#             .sort_values('num', ascending=False).rename(columns={'num': num_column})
#     else:
#         return df[[cat_column, num_column]] \
#             .groupby(cat_column) \
#             .agg(num = (num_column, func), modes = (num_column, lambda x: '')) \
#             .reset_index() \
#             .sort_values('num', ascending=False).rename(columns={'num': num_column})

# def create_figure(config):
#     df_for_fig = prepare_df(config)
#     x = df_for_fig[config['cat_column']]
#     y = df_for_fig[config['num_column']]

#     bar_fig = go.Bar(x=x, y=y, name='')
#     line_fig = go.Scatter(x=x, y=y, mode='lines', visible=False, name='')
#     area_fig = go.Scatter(x=x, y=y, mode='lines', fill='tozeroy', visible=False, name='')
#     return go.Figure(data=[bar_fig, line_fig, area_fig])

# def create_buttons(config):
#     buttons = []
#     # print(len(fig.data))
#     # print(fig['layout']['yaxis']['orientation'])
#         # orientation = fig['layout']['yaxis']['orientation'] if fig['layout']['yaxis']['orientation'] in ['v', 'h'] else fig['layout']['xaxis']['orientation']
#     df, cat_column, num_column = config['df'], config['cat_column'], config['num_column']
#     buttons.append(dict(label='Ver', method='update', args=[{'orientation': 'v'}]))
#     buttons.append(dict(label='Hor', method='update', args=[{'orientation': 'h'}]))
#     buttons.append(dict(label='Bar', method='update', args=[{'visible': [True, False, False]}]))
#     buttons.append(dict(label='Line', method='update', args=[{'visible': [False, True, False]}]))
#     buttons.append(dict(label='Area', method='update', args=[{'visible': [False, False, True]}]))
#     for func in ['sum', 'mean', 'median', 'count', 'mode', 'std', 'min', 'max']:
#         config['func'] = func
#         if func == 'mode':
#             buttons.append(dict(label=f'Agg {func.capitalize()}'
#                                 , method='update'
#                                 , args=[{
#                                     'orientation': 'v'
#                                     , 'x': [prepare_df(config)[cat_column].to_list()] 
#                                     , 'y': [prepare_df(config)[num_column].to_list()] 
#                                 # , args=[{'x': [prepare_df(config)[cat_column].to_list()] 
#                                 #             if any([fig.data[i]['orientation'] == 'v' for i in range(len(fig.data))]) 
#                                 #             else [prepare_df(config)[num_column].to_list()]
#                                 #         , 'y': [prepare_df(config)[num_column].to_list()] 
#                                 #             if any([fig.data[i]['orientation'] == 'v' for i in range(len(fig.data))]) 
#                                 #             else [prepare_df(config)[cat_column].to_list()]
#                                         # , 'text': [[', '.join(map(str,x)) for x in prepare_df(config)['modes'].to_list()]]
#                                         # , 'hovertemplate': '<br>Value: %{y:.2f}<br>Modes: %{text}'
#                                         # , 'marker': {'color': ['#049CB3' if len(x) > 1 else 'rgba(128, 60, 170, 0.9)' for x in prepare_df(config)['modes'].to_list()]}
#                                         # , 'textposition': 'none'
#                                         # , 'textposition': 'auto'
#                                         # , 'textfont': {'color': 'black'}
#                                         }]))
#         else:
#             buttons.append(dict(label=f'Agg {func.capitalize()}'
#                                 , method='update'
#                                 , args=[{
#                                     'orientation': 'v'
#                                     , 'x': [prepare_df(config)[cat_column].to_list()] 
#                                     , 'y': [prepare_df(config)[num_column].to_list()] 
#                                         #  'x': [prepare_df(config)[cat_column].to_list()] 
#                                         # , 'y': [prepare_df(config)[num_column].to_list()]    
#                                         # , 'text': ['']
#                                         # , 'hovertemplate': '<br>Value: %{y:.2f}'
#                                         # , 'marker': {'color': ['rgba(128, 60, 170, 0.9)' for x in prepare_df(config)['modes'].to_list()]}
#                                         # , 'textposition': 'none'
#                                         # , 'textposition': 'auto'
#                                         # , 'textfont': {'color': 'black'}
#                                         }]))
#     return buttons

# def update_layout(fig, buttons):
#     fig.update_layout(
#         updatemenus=[
#             dict(
#                 type="buttons",
#                 direction="left",
#                 buttons=buttons[:2],  # first 3 buttons (Bar, Line, Area)
#                 pad={"r": 10, "t": 70},
#                 showactive=True,
#                 x=0,
#                 xanchor="left",
#                 y=1.1,
#                 yanchor="bottom"
#             ),            
#             dict(
#                 type="buttons",
#                 direction="left",
#                 buttons=buttons[2:5],  # first 3 buttons (Bar, Line, Area)
#                 pad={"l": 120, "r": 10, "t": 70},
#                 showactive=True,
#                 x=0,
#                 xanchor="left",
#                 y=1.1,
#                 yanchor="bottom"
#             ),
#             dict(
#                 type="buttons",
#                 direction="left",
#                 buttons=buttons[5:],  # remaining buttons (Agg functions)
#                 pad={"l": 300, "r": 10, "t": 70},  # add left padding to separate from previous group
#                 showactive=True,
#                 x=0,
#                 xanchor="left",pad
#                 y=1.1,
#                 yanchor="bottom"
#             ),
#         ]
#     )

# config = {
#     'df': df,
#     'cat_column': 'education',
#     'num_column': 'dob_years',
#     'func': 'sum'
# }

# fig = create_figure(config)
# buttons = create_buttons(config)
# update_layout(fig, buttons)
# fig.update_layout(height = 500
#                 #  , title={'text': f"{config['num_column']}<br>{'cat_column'}", 'y': 0.9}
#                   , xaxis={'title': config['cat_column']}
#                   , yaxis={'title': config['num_column']})
# fig.show(config=dict(displayModeBar=True))