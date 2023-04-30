"""
Marcus Jundt & Mitchell Stapelman
UW, CSE 163
Takes health spending per capita data and census income per capita data and
cleans and sorts the data into dataframes focusing on different races and age
groups.
"""

import pandas as pd
import numpy as np

INFLATION_2016 = 1.2046
INFLATION_2020 = 1.1171


def load_spending_data(file_name):
    """
    Loads the health spending data set CSV.
    :param file_name: health spending file name
    :return: dataframe with unnecessary columns sorted out, adjusted for
    inflation
    """
    df = pd.read_csv(file_name)
    df = df.loc[:, ~df.columns.isin(['upper', 'lower'])]
    df = df[(df['sex_id'] == 3) & (df['acause'] == 'all') &
            (df['type_care'] == 'Total') &
            (df['spending_unit'] == 'Spending per capita in 2016 USD')]
    df['val'] = df['val'] * INFLATION_2016
    return df[['year', 'age_group_id', 'age_group_name',
               'race_ethnicity', 'val']]


def spending_by_race(df):
    """
    Sorts the health spending data set CSV by race.
    :param df: health spending dataframe
    :return: dataframe with the different age groups removed, grouped by race
    """
    df = df[df['age_group_id'] == 22]
    return df.loc[:, ~df.columns.isin(['age_group_id', 'age_group_name'])]


def spending_by_age(df):
    """
    Sorts the health spending data set CSV by age.
    :param df: health spending dataframe
    :return: dataframe with the different races removed, grouped by age
    """
    df = df[(df['race_ethnicity'] == 'Total') & (df['age_group_id'] >= 10) &
            (df['age_group_id'] != 22) & (df['age_group_id'] != 28)]
    df = df.loc[:, df.columns != 'race_ethnicity']
    df = df.sort_values(['year', 'age_group_id'])

    # Ranges
    df_r = df[df['age_group_id'] < 20]
    df_r = df_r.reset_index(drop=True)
    agg_function = {'year': 'first', 'age_group_id': 'mean',
                    'age_group_name': ' '.join, 'val': 'mean'}
    df_r = df_r.groupby(df_r.index // 2).agg(agg_function)
    df_r['age_group_name'] = df_r['age_group_name'].apply(
                             lambda x: x.split()[0] + ' to ' + x.split()[-1])

    # 75 plus
    df_plus = df[df['age_group_id'] >= 20]
    df_plus = df_plus.groupby('year', as_index=False).mean()
    df_plus['age_group_name'] = '75+'

    result = pd.concat([df_r, df_plus])
    result = result.sort_values(['year', 'age_group_id'])
    return result[['year', 'age_group_name', 'val']]


def income_by_race(file_names):
    """
    Combines and fixes the messy census race CSVs.
    :param file_names: list of census income by race messy CSVs
    :return: dataframe showing per capita income of each race from the CSVs
    """
    # White
    df1 = pd.read_csv(file_names[0])
    df1 = census_column_helper(df1).loc[8:28]
    df1['race_ethnicity'] = 'White (non-Hispanic)'
    # Asian
    df2 = pd.read_csv(file_names[1])
    df2 = census_column_helper(df2).loc[8:28]
    df2['race_ethnicity'] = \
        'Asian, Native Hawaiian, Pacific Islander (non-Hispanic)'
    # Black
    df3 = pd.read_csv(file_names[2])
    df3 = census_column_helper(df3).loc[8:28]
    df3['race_ethnicity'] = 'Black (non-Hispanic)'
    # Hispanic
    df4 = pd.read_csv(file_names[3])
    df4 = census_column_helper(df4).loc[7:27]
    df4['race_ethnicity'] = 'Hispanic'
    # Combine
    combined = pd.concat([df1, df2, df3, df4], ignore_index=True)
    combined = combined[['year', 'pop (1000s)',
                         'race_ethnicity', 'PCI 2020 $']]
    combined = year_footnotes_fix(combined, 'race_ethnicity')
    return combined[['year', 'race_ethnicity', 'PCI 2020 $']]


def census_column_helper(data):
    """
    Fixes columns of the messy census race dataframes.
    :param data: census race dataframes
    :return: census race dataframes with 'year', 'pop (1000s)',
    'PCI (current $)', 'PCI 2020 $' as columns.
    """
    data.columns = ['year', 'pop (1000s)', 'PCI (current $)', 'PCI 2020 $']
    return data.dropna()


def income_by_age(file_name):
    """
    Takes the messy census age CSV and makes a dataframe sorted by year and age
    group showing per capita income.
    :param file_name: census income by age CSV
    :return: dataframe showing income per year for each age in the ranges
    '25 to 34', '35 to 44', '45 to 54', '55 to 64', '65 to 74', and '75+'.
    """
    df = pd.read_csv(file_name)
    df.columns = ['year', 'pop (1000s)', 'median (current $)',
                  'median (2020 $)', 'PCI (current $)', 'PCI 2020 $']
    df = df[['year', 'pop (1000s)', 'PCI 2020 $']]
    ranges_list = ['25 to 34', '35 to 44', '45 to 54',
                   '55 to 64', '65 to 74', '75 Years and Over']
    df_list = []
    for ages in ranges_list:
        ind = np.where(df['year'].str.contains(ages, na=False))[0][0]
        df_range = df[ind+3:ind+24].copy(deep=False)
        df_range['age_group_name'] = ages
        df_list.append(df_range)
    combined = pd.concat(df_list, ignore_index=True)
    combined = combined.sort_values(['year', 'age_group_name'])
    combined = year_footnotes_fix(combined, 'age_group_name')
    combined.loc[combined['age_group_name'] == '75 Years and Over',
                                               'age_group_name'] = '75+'
    return combined[['age_group_name', 'year', 'PCI 2020 $']]


def year_footnotes_fix(df, group):
    """
    Removed the footnotes for some years in the census age dataframe and
    combined duplicate years.
    :param df: census income by age dataframe
    :param group: either 'race_ethnicity' or 'age_group_name'
    :return: dataframe with removed footnotes and combined duplicate years,
    adjusts per capita income for inflation.
    """
    df['year'] = df['year'].str.split().str[0].astype(int)
    col = ['pop (1000s)', 'PCI 2020 $']
    df[col] = df[col].replace(',', '', regex=True).astype(int)
    df['PCI 2020 $'] = df['PCI 2020 $'] * INFLATION_2020
    function = {'year': 'first', 'pop (1000s)': 'sum', 'PCI 2020 $':
                (lambda x:
                 np.average(x, weights=df.loc[x.index, 'pop (1000s)']))}
    return df.groupby(['year', group], as_index=False).agg(function)


def merged_race_data(df1, df2):
    """
    Merges the census income race data and health spending race data into one
    dataframe. Updates column names to 'Year', 'Race/Ethnicity',
    'Health spending per capita ($)', 'Income per capita ($)'.
    :param df1: health spending race dataframe
    :param df2: census income race dataframe
    :return: inner merged dataframe of df1 and df2 by year and race
    """
    merged = merge_data(df1, df2, 'race_ethnicity')
    merged.columns = ['Year', 'Race/Ethnicity',
                      'Health spending per capita ($)',
                      'Income per capita ($)']
    return merged


def merged_age_data(df1, df2):
    """
    Merges the census income age data and health spending age data into one
    dataframe. Updates column names to 'Year', 'Age group',
    'Health spending per capita ($)', 'Income per capita ($)'.
    :param df1: health spending age dataframe
    :param df2: census income age dataframe
    :return: inner merged dataframe of df1 and df2 by year and age
    """
    merged = merge_data(df1, df2, 'age_group_name')
    merged.columns = ['Year', 'Age group', 'Health spending per capita ($)',
                      'Income per capita ($)']
    return merged


def merge_data(df1, df2, factor):
    """
    Performs inner merge of two given dataframes df1, df2 by year and given
    factor.
    :param df1: health spending age or race dataframe
    :param df2: census income age or race dataframe
    :param factor: either 'race_ethnicity' or 'age_group_name'
    :return: inner merge of df1 and df2 by year and factor.
    """
    return df1.merge(df2, how='inner', left_on=['year', factor],
                     right_on=['year', factor])


def col_display(num_col):
    """
    Changes display to show columns up to the given number.
    :param num_col: limit of number of columns
    """
    width = 320
    pd.set_option('display.width', width)
    np.set_printoptions(linewidth=width)
    pd.set_option('display.max_columns', num_col)


def get_full_data(health_file_name, age_file_name, race_file_names):
    """
    Takes the health spending and census file names and returns a tuple of age
    dataframe and race dataframe showing income and health spending per group.
    :param health_file_name: health spending file name
    :param age_file_name: census income by age file name
    :param race_file_names: census income by race file name
    :return: merged and cleaned tuple of dataframes from the given CSV files
    """
    df_health = load_spending_data(health_file_name)
    df_health_age = spending_by_age(df_health)
    df_census_age = income_by_age(age_file_name)
    df_health_race = spending_by_race(df_health)
    df_census_race = income_by_race(race_file_names)
    return merged_race_data(df_health_race, df_census_race), \
        merged_age_data(df_health_age, df_census_age)


def get_age_data(health_file_name, age_file_name):
    """
    Takes the health spending and census age file name and returns a merged
    dataframe from the two CSVs showing health spending and income per age.
    :param health_file_name: health spending file name
    :param age_file_name: census income by age file name
    :return: merged dataframe showing health spending and income per age
    """
    df_health_age = spending_by_age(load_spending_data(health_file_name))
    df_census_age = income_by_age(age_file_name)
    return merged_age_data(df_health_age, df_census_age)


def get_race_data(health_file_name, race_file_names):
    """
    Takes the health spending and census race file names and returns a merged
    dataframe from the CSVs showing health spending and income per race.
    :param health_file_name: health spending file name
    :param race_file_names: census income by race file name
    :return: merged dataframe showing health spending and income per race
    """
    df_health_race = spending_by_race(load_spending_data(health_file_name))
    df_census_race = income_by_race(race_file_names)
    return merged_race_data(df_health_race, df_census_race)