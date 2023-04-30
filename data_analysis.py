"""
Marcus Jundt & Mitchell Stapelman
UW, CSE 163
Takes age and race income and health spending datasets, predicts future data,
and visualizes the results in various plots.
"""

import pandas as pd
import numpy as np
import plotly.express as py
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from data_processing import get_full_data, col_display

# Filenames
HEALTH_FILE_NAME = 'data/health_data_2002-2016.CSV'
CENSUS_AGE_FILE_NAME = 'data/p10ar.csv'
CENSUS_RACE_FILE_NAMES = ['data/p01wnh.csv', 'data/p01a.csv',
                          'data/p01b.csv', 'data/p01h.csv']
# Sets default values for size of plot elements
T_SIZE = 36
AX_SIZE = 18
LBL_SIZE = 24
LEG_SIZE = 24


def percentage_income(df):
    """
    Creates new column showing percentage of income spent on health spending.
    :param df: merged age or race dataframe
    :return: dataframe w/ new column showing percent of income spent on health.
    """
    df['% Income Health Spending'] = (df['Health spending per capita ($)'] /
                                      df['Income per capita ($)']) * 100
    return df


def future_results(data_r, data_a, acc=False):
    """
    Returns tuple of dataframes showing future age and race percent income.
    :param data_r: merged race dataframe
    :param data_a: merged age dataframe
    :param acc: test for accuracy, True or False
    :return: tuple of dataframes with predicted future percent income data.
    """
    return predict_future(data_r, 'Race/Ethnicity', acc=acc), \
        predict_future(data_a, 'Age group', acc=acc)


def predict_future(data, factor, acc=False):
    """
    Runs linear regression model to predict future percent income spent.
    :param data: merged race or age dataframe
    :param factor: either 'Race/Ethnicity' or 'Age group'
    :param acc: test for accuracy, True or False
    :return: dataframe with predicted percent income spent data
    """
    predictions_list = []
    data = data[['Year', factor, '% Income Health Spending']]
    groups = sorted(list(set(data[factor].tolist())))
    errors = None
    n = 0
    if acc:
        errors = {"Train error": np.empty(len(groups)),
                  "Test error": np.empty(len(groups))}
    for group in groups:
        s_data = data.loc[data[factor] == group, data.columns != factor]
        features = np.array(s_data['Year']).reshape(-1, 1)
        labels = s_data['% Income Health Spending']
        model = LinearRegression()
        features_train, features_test, labels_train, labels_test = \
            train_test_split(features, labels, test_size=0.1)
        model.fit(features_train, labels_train)
        if acc:
            train_predictions = model.predict(features_train)
            test_predictions = model.predict(features_test)
            errors['Train error'][n] = mean_squared_error(labels_train,
                                                          train_predictions)
            errors['Test error'][n] = mean_squared_error(labels_test,
                                                         test_predictions)
            n += 1
        else:
            years = np.arange(2017, 2029)
            prediction = model.predict(years.reshape(-1, 1))
            df_dict = {'Year': years, factor: group,
                       '% Income Health Spending': prediction}
            future_df = pd.DataFrame(df_dict)
            predictions_list.append(future_df)
    if acc:
        errors['Train error'] = np.mean(errors['Train error'])
        errors['Test error'] = np.mean(errors['Test error'])
        return errors
    else:
        predictions_list.append(data)
        combined = pd.concat(predictions_list, ignore_index=True).sort_values(
            ['Year', factor])
        return combined.reset_index(drop=True)


def plot_charts(data_r, data_a, race_future, age_future):
    """
    Plots the results of the analysis of percent income spent on health and the
    future predictions.
    :param data_r: merged race dataframe
    :param data_a: merged age dataframe
    :param race_future: future race income spent dataframe
    :param age_future: future age income spent dataframe
    """
    plot_set(data_r, 'Race/Ethnicity')
    plot_set(data_a, 'Age group')
    plot_future(race_future, 'Race/Ethnicity')
    plot_future(age_future, 'Age group')


def plot_set(data, factor):
    """
    Plots health spending, income, and percent income spent on health from
    2002-2016, by race or age.
    :param data: merged age or race dataframe
    :param factor: either 'Race/Ethnicity' or 'Age group'
    """
    # Bar chart - Health spending per capita for all years
    fig1_title = 'Per Capita Health Spending by ' + factor + ' Over Time'
    fig1 = py.bar(data, x=factor, y='Health spending per capita ($)',
                  color='Year', title=fig1_title)
    plot_size_helper(fig1, T_SIZE, AX_SIZE, LBL_SIZE, LEG_SIZE)
    fig1.show()

    # Line plot - Health spending per capita for all years
    fig2_title = 'Per Capita Health Spending by ' + factor + ' Over Time'
    fig2 = py.line(data, x='Year', y='Health spending per capita ($)',
                   color=factor, title=fig2_title, markers=True)
    plot_size_helper(fig2, T_SIZE, AX_SIZE, LBL_SIZE, LEG_SIZE)
    fig2.update_layout(xaxis_range=[2002, 2016], xaxis=dict(dtick=1))
    fig2.show()

    # Line plot - Income per capita for all years
    fig3_title = 'Per Capita Income by ' + factor + ' Over Time'
    fig3 = py.line(data, x='Year', y='Income per capita ($)',
                   color=factor, title=fig3_title, markers=True)
    plot_size_helper(fig3, T_SIZE, AX_SIZE, LBL_SIZE, LEG_SIZE)
    fig3.update_layout(xaxis_range=[2002, 2016], xaxis=dict(dtick=1))
    fig3.show()

    # Line plot - Percentage of income spent on health for all years
    fig4_title = 'Percentage of Income Spent on Health by ' \
        + factor + ' Over Time'
    fig4 = py.line(data, x='Year', y='% Income Health Spending',
                   color=factor, title=fig4_title, markers=True)
    plot_size_helper(fig4, T_SIZE, AX_SIZE, LBL_SIZE, LEG_SIZE)
    fig4.update_layout(xaxis_range=[2002, 2016], xaxis=dict(dtick=1))
    fig4.show()

    # Scatter plot - Income vs Health spending for all years
    fig5_title = 'Percentage of Income Spent on Health by ' \
        + factor + ' Over Time'
    fig5 = py.scatter(data, x='Income per capita ($)',
                      y='Health spending per capita ($)', color=factor,
                      title=fig5_title, hover_name='Year', trendline='ols')
    plot_size_helper(fig5, T_SIZE, AX_SIZE, LBL_SIZE, LEG_SIZE)
    fig5.update_traces(line_width=3, selector=dict(type="scatter"))
    fig5.show()


def plot_future(data, factor):
    """
    Plots the future prediction for percent income spent on health until 2028.
    :param data: future age or race percent spent dataframe
    :param factor: either 'Race/Ethnicity' or 'Age group'
    """
    fig_title = 'Future Percentage of Income Spent on Health by ' \
        + factor + ' Over Time'
    fig = py.line(data[data['Year'] >= 2016], x='Year',
                  y='% Income Health Spending', color=factor, title=fig_title,
                  markers=True)
    fig2 = py.line(data[data['Year'] <= 2016], x='Year',
                   y='% Income Health Spending', color=factor)
    fig2.update_traces(opacity=0.4)
    for line in fig2.data:
        line.update(showlegend=False)
        fig.add_trace(line)
    plot_size_helper(fig, T_SIZE, AX_SIZE, LBL_SIZE, LEG_SIZE)
    fig.update_layout(xaxis_range=[2010, 2028])
    fig.add_vline(x=2016, line_dash='dash', opacity=0.5)
    fig.show()


def plot_size_helper(fig, title_size, axes_size, label_size, legend_size):
    """
    Updates size of various elements of plotly figures.
    :param fig: given plotly figure
    :param title_size: size of title
    :param axes_size: size of axes values
    :param label_size: size of axes labels
    :param legend_size: size of legend
    """
    # Update size of elements in plot
    fig.update_layout(
        xaxis=dict(titlefont=dict(size=label_size),
                   tickfont=dict(size=axes_size)),
        yaxis=dict(titlefont=dict(size=label_size),
                   tickfont=dict(size=axes_size)),
        legend=dict(font=dict(size=legend_size)))
    # Update size of plot title
    fig['layout']['title']['font'] = dict(size=title_size)
    # Update line/dot width
    fig.update_traces(line_width=4, selector=dict(type="scatter"))
    fig.update_traces(marker=dict(size=14, line=dict(width=2, color='gray')),
                      selector=dict(mode='markers'))


def main():
    col_display(13)
    # pd.set_option('display.max_rows', None)
    race_data, age_data = get_full_data(HEALTH_FILE_NAME,
                                        CENSUS_AGE_FILE_NAME,
                                        CENSUS_RACE_FILE_NAMES)
    race_data = percentage_income(race_data)
    age_data = percentage_income(age_data)
    race_future_data, age_future_data,  = future_results(race_data, age_data)
    print(age_future_data)
    print(race_future_data)
    print(age_data)
    print(race_data)
    plot_charts(race_data, age_data, race_future_data, age_future_data)


if __name__ == '__main__':
    main()