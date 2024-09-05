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
