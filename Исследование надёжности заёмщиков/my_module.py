import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from ipywidgets import widgets, Layout
from IPython.display import display, display_html, display_markdown
from tqdm.auto import tqdm


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
    all_rows = pd.DataFrame({
        'Rows': [pretty_value(df.shape[0])], 'Features': [df.shape[1]], 'RAM (Mb)': [round(df.__sizeof__() / 1_048_576)], 'Duplicates': [df.duplicated().sum()]
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


def make_widget_summary(column):
    column_name = column.name
    values = column.count()
    values = pretty_value(column.count())
    values_pct = column.count() * 100 / column.size
    if values_pct > 99 and values_pct < 100:
        values_pct = round(values_pct, 1)
    else:
        values_pct = round(values_pct)
    values = f'{values} ({values_pct}%)'
    missing = column.isna().sum()
    if missing == 0:
        missing = '---'
    else:
        missing = pretty_value(column.isna().sum())
        missing_pct = round(column.isna().sum() * 100 / column.size)
        missing = f'{missing} ({missing_pct}%)'
    distinct = pretty_value(column.nunique())
    distinct_pct = column.nunique() * 100 / column.size
    if distinct_pct > 99 and distinct_pct < 100:
        valudistinct_pctes_pct = round(distinct_pct, 1)
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
        zeros = f'{zeros} ({zeros_pct}%)'
    negative = (column < 0).sum()
    if negative == 0:
        negative = '---'
    else:
        negative = pretty_value(negative)
        negative_pct = round((column < 0).sum() * 100 / column.size)
        negative = f'{negative} ({negative_pct}%)'
    duplicates = column.duplicated().sum()
    if duplicates == 0:
        duplicates = '---'
    else:
        duplicates = pretty_value(duplicates)
        duplicates_pct = column.duplicated().sum() * 100 / column.size
        if duplicates_pct > 99 and duplicates_pct < 100:
            duplicates_pct = round(duplicates_pct, 1)
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
    if values_pct > 99 and values_pct < 100:
        values_pct = round(values_pct, 1)
    else:
        values_pct = round(values_pct)
    values = f'{values} ({values_pct}%)'
    missing = column.isna().sum()
    if missing == 0:
        missing = '---'
    else:
        missing = pretty_value(column.isna().sum())
        missing_pct = round(column.isna().sum() * 100 / column.size)
        missing = f'{missing} ({missing_pct}%)'
    distinct = pretty_value(column.nunique())
    distinct_pct = column.nunique() * 100 / column.size
    if distinct_pct > 99 and distinct_pct < 100:
        valudistinct_pctes_pct = round(distinct_pct, 1)
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
        zeros = f'{zeros} ({zeros_pct}%)'
    duplicates = column.duplicated().sum()
    if duplicates == 0:
        duplicates = '---'
    else:
        duplicates = pretty_value(duplicates)
        duplicates_pct = column.duplicated().sum() * 100 / column.size
        if duplicates_pct > 99 and duplicates_pct < 100:
            duplicates_pct = round(duplicates_pct, 1)
        else:
            duplicates_pct = round(duplicates_pct)
        duplicates = f'{duplicates} ({duplicates_pct}%)'
    ram = round(column.__sizeof__() / 1_048_576)
    if ram == 0:
        ram = '<1 Mb'
    column_summary = pd.DataFrame({
        'Values': [values], 'Missing': [missing], 'Distinct': [distinct], 'Duplicates': [duplicates], 'Empty': [zeros], 'RAM (Mb)': [ram]
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


def my_info(df, graphs=True, num=True, obj=True):
    '''
    Функция выводить информацию о датафрейме
    Четвертый столбцев (перед графиками) Value counts
    Parameters
    df: pandas.DataFrame
    Датафрейм с данными
    graphs: bool, default True
    Если True, то выводятся графики. 
    '''
    if not num and not obj:
        return
    vbox_layout = Layout(display='flex',
                         # flex_flow='column',
                         justify_content='space-around',
                         # width='auto',
                         # grid_gap = '20px',
                         # align_items = 'flex-end'
                         )
    boxes = []
    funcs_num = [make_widget_summary, make_widget_pct,
                 make_widget_std, make_widget_value_counts]
    func_obj = [make_widget_summary_obj, make_widget_value_counts_obj]
    if graphs:
        funcs_num += [make_widget_hist_plotly, make_widget_violine_plotly]
        func_obj += [make_widget_bar_obj]
    if num:
        num_columns = filter(
            lambda x: pd.api.types.is_numeric_dtype(df[x]), df.columns)
        for column in tqdm(num_columns):
            widgets_ = [func(df[column]) for func in funcs_num]
            boxes.extend((widgets_))
        layout = widgets.Layout(
            grid_template_columns='auto auto auto auto auto auto')
        num_grid = widgets.GridBox(boxes, layout=layout)
    if obj:
        obj_columns = filter(
            lambda x: not pd.api.types.is_numeric_dtype(df[x]), df.columns)
        for column in tqdm(obj_columns):
            widgets_ = [func(df[column]) for func in func_obj]
            boxes.extend((widgets_))
        layout = widgets.Layout(
            grid_template_columns='auto auto auto')
        obj_grid = widgets.GridBox(boxes, layout=layout)

    # widgets.Layout(grid_template_columns="200px 200px 200px 200px 200px 200px")))
    display(make_widget_all_frame(df))
    if num:
        display(num_grid)
    if obj:
        display(obj_grid)

def my_info_gen(df, graphs=True, num=True, obj=True):
    '''
    Генератор выводить информацию о датафрейме
    Четвертый столбцев (перед графиками) Value counts
    Parameters
    df: pandas.DataFrame
    Датафрейм с данными
    graphs: bool, default True
    Если True, то выводятся графики. 
    '''
    if not num and not obj:
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
    if graphs:
        funcs_num += [make_widget_hist_plotly, make_widget_violine_plotly]
        func_obj += [make_widget_bar_obj]
    if num:
        num_columns = filter(
            lambda x: pd.api.types.is_numeric_dtype(df[x]), df.columns)
        layout = widgets.Layout(
            grid_template_columns='auto auto auto auto auto auto')
        for column in tqdm(num_columns):
            widgets_ = [func(df[column]) for func in funcs_num]
            display(widgets.GridBox(boxes, layout=layout))
            yield
    if obj:
        obj_columns = filter(
            lambda x: not pd.api.types.is_numeric_dtype(df[x]), df.columns)
        for column in tqdm(obj_columns):
            widgets_ = [func(df[column]) for func in func_obj]
            boxes.extend((widgets_))
        layout = widgets.Layout(
            grid_template_columns='auto auto auto')
        obj_grid = widgets.GridBox(boxes, layout=layout)

