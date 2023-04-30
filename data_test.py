"""
Marcus Jundt & Mitchell Stapelman
UW, CSE 163
Test file for the data processing and analysis.
"""

from cse163_utils import assert_equals
import pandas as pd
import data_processing as pr
import data_analysis as an


def test_year_footnotes_fix(data_r, data_a, df_w, df_a):
    """
    Tests year_footnotes_fix from data_processing.py, makes sure weighted mean
    is correct.
    """
    pci = int(df_w.iloc[16, 3].replace(',', ''))
    pci_f = int(df_w.iloc[17, 3].replace(',', ''))
    pop = int(df_w.iloc[16, 1].replace(',', ''))
    pop_f = int(df_w.iloc[17, 1].replace(',', ''))
    weighted_mean = ((pci * pop + pci_f * pop_f)/(pop + pop_f)) * \
        pr.INFLATION_2020
    data_r = data_r[(data_r['Race/Ethnicity'] == 'White (non-Hispanic)') &
                    (data_r['Year'] == 2013)]
    assert_equals(weighted_mean, data_r.loc[47, 'Income per capita ($)'])

    pci = int(df_a.iloc[120, 5].replace(',', ''))
    pci_f = int(df_a.iloc[121, 5].replace(',', ''))
    pop = int(df_a.iloc[120, 1].replace(',', ''))
    pop_f = int(df_a.iloc[121, 1].replace(',', ''))
    weighted_mean = ((pci * pop + pci_f * pop_f) / (pop + pop_f)) * \
        pr.INFLATION_2020
    data_a = data_a[(data_a['Age group'] == '25 to 34') &
                    (data_a['Year'] == 2013)]
    assert_equals(weighted_mean, data_a.loc[66, 'Income per capita ($)'])


def test_spending_by_age(df_old, df_ne):
    """
    Tests spending_by_age method from data_processing.py, makes sure the
    combined age groups have the correct mean values.
    """
    spe_25 = df_old.loc[16, 'val'] * pr.INFLATION_2016
    spe_30 = df_old.loc[17, 'val'] * pr.INFLATION_2016
    df_ne = df_ne[(df_ne['Age group'] == '25 to 34') & (df_ne['Year'] == 2010)]
    assert_equals((spe_25 + spe_30)/2,
                  df_ne.loc[48, 'Health spending per capita ($)'])


def test_model_results(data):
    """
    Tests the accuracy of the linear regression model. The accuracy has
    variance between runs, so is possible to return an error though unlikely.
    """
    acc = an.predict_future(data, 'Race/Ethnicity', acc=True)
    train_err = acc['Train error']
    test_err = acc['Test error']
    assert_equals(True, train_err < 3)  # usually < 1.5
    assert_equals(True, test_err < 3)  # usually < 1.5


def test_percentage_income(data_r, data_a):
    """
    Tests the percentage_income function from data_analysis.py, makes sure the
    calculated percentages are accurate.
    """
    assert_equals((data_r.iloc[12, 2] / data_r.iloc[12, 3]) * 100,
                  data_r.iloc[12, 4])
    assert_equals((data_a.iloc[20, 2] / data_a.iloc[20, 3]) * 100,
                  data_a.iloc[20, 4])


def main():
    # Initialize datasets
    pr.col_display(13)
    pd.set_option('display.max_rows', None)
    race_data, age_data = pr.get_full_data(an.HEALTH_FILE_NAME,
                                           an.CENSUS_AGE_FILE_NAME,
                                           an.CENSUS_RACE_FILE_NAMES)
    race_data = an.percentage_income(race_data)
    age_data = an.percentage_income(age_data)
    df_w = pd.read_csv(an.CENSUS_RACE_FILE_NAMES[0])
    df_a = pd.read_csv(an.CENSUS_AGE_FILE_NAME)
    df_s = pd.read_csv('data/small_health_data_2002-2016.csv')
    # Run tests
    test_year_footnotes_fix(race_data, age_data, df_w, df_a)
    test_spending_by_age(df_s, age_data)
    test_model_results(race_data)
    test_percentage_income(race_data, age_data)


if __name__ == '__main__':
    main()