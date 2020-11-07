from constants import ROOT_DIR
import pandas as pd
from datetime import datetime
import numpy as np
import wget

# Try to load, otherwise download
try:
    countries_to_fit = pd.read_csv(ROOT_DIR + '/real_data/countries.csv')
except:
    # url of the raw csv dataset
    urls = [
        'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv',
        'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv',
        'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
    ]
    [wget.download(url, out=ROOT_DIR + '/real_data/') for url in urls]

    confirmed_df = pd.read_csv(ROOT_DIR + '/real_data/time_series_covid19_confirmed_global.csv')
    deaths_df = pd.read_csv(ROOT_DIR + '/real_data/time_series_covid19_deaths_global.csv')
    recovered_df = pd.read_csv(ROOT_DIR + '/real_data/time_series_covid19_recovered_global.csv')

    dates = confirmed_df.columns[4:]
    confirmed_df_long = confirmed_df.melt(
        id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
        value_vars=dates,
        var_name='Date',
        value_name='Confirmed'
    )
    deaths_df_long = deaths_df.melt(
        id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
        value_vars=dates,
        var_name='Date',
        value_name='Deaths'
    )
    recovered_df_long = recovered_df.melt(
        id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
        value_vars=dates,
        var_name='Date',
        value_name='Recovered'
    )

    recovered_df_long = recovered_df_long[recovered_df_long['Country/Region'] != 'Canada']

    # Merging confirmed_df_long and deaths_df_long
    full_table = confirmed_df_long.merge(
        right=deaths_df_long,
        how='left',
        on=['Province/State', 'Country/Region', 'Date', 'Lat', 'Long']
    )
    # Merging full_table and recovered_df_long
    full_table = full_table.merge(
        right=recovered_df_long,
        how='left',
        on=['Province/State', 'Country/Region', 'Date', 'Lat', 'Long']
    )

    full_table['Date'] = pd.to_datetime(full_table['Date'])
    full_table['Recovered'] = full_table['Recovered'].fillna(0)
    ship_rows = full_table['Province/State'].str.contains('Grand Princess') | full_table['Province/State'].str.contains(
        'Diamond Princess') | full_table['Country/Region'].str.contains('Diamond Princess') | full_table[
                    'Country/Region'].str.contains('MS Zaandam')
    full_ship = full_table[ship_rows]

    full_table = full_table[~(ship_rows)]
    # Active Case = confirmed - deaths - recovered
    full_table['Active'] = full_table['Confirmed'] - full_table['Deaths'] - full_table['Recovered']

    full_grouped = full_table.groupby(['Date', 'Country/Region'])[
        'Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

    # new cases
    temp = full_grouped.groupby(['Country/Region', 'Date', ])['Confirmed', 'Deaths', 'Recovered']
    temp = temp.sum().diff().reset_index()
    mask = temp['Country/Region'] != temp['Country/Region'].shift(1)
    temp.loc[mask, 'Confirmed'] = np.nan
    temp.loc[mask, 'Deaths'] = np.nan
    temp.loc[mask, 'Recovered'] = np.nan
    # renaming columns
    temp.columns = ['Country/Region', 'Date', 'New cases', 'New deaths', 'New recovered']
    # merging new values
    full_grouped = pd.merge(full_grouped, temp, on=['Country/Region', 'Date'])
    # filling na with 0
    full_grouped = full_grouped.fillna(0)
    # fixing data types
    cols = ['New cases', 'New deaths', 'New recovered']
    full_grouped[cols] = full_grouped[cols].astype('int')
    #
    full_grouped['New cases'] = full_grouped['New cases'].apply(lambda x: 0 if x < 0 else x)

    full_grouped.to_csv(ROOT_DIR + '/real_data/countries.csv')

countries_to_fit['Date'] = pd.to_datetime(countries_to_fit['Date'])
countries = countries_to_fit['Country/Region'].unique()

# Two dictionaries to split data pre/post-lockdown.
# They will store the data in the form:
# Country : [Infected, Removed, New Cases]
countries_dict_prelock = {}
countries_dict_postlock = {}

selected_countries = ['Italy', 'Spain', 'Greece', 'France', 'Germany', 'Switzerland', 'United Kingdom', 'Russia', 'US',
                      'Sweden', 'Belgium']

selected_countries_populations = {'Italy': 60460000,
                                  'Spain': 46940000,
                                  'Greece': 10720000,
                                  'France': 66990000,
                                  'Germany': 83020000,
                                  'Switzerland': 8570000,
                                  'United Kingdom': 66650000,
                                  'Russia': 146793000,
                                  'US': 328200000,
                                  'Sweden': 10230000,
                                  'Belgium': 11460000}

selected_countries_rescaling = {'Italy': 1}

lockdown_ends = {'Italy': datetime.strptime('04/05/2020', '%d/%m/%Y'),
                 'Spain': datetime.strptime('04/05/2020', '%d/%m/%Y'),
                 'Greece': datetime.strptime('04/05/2020', '%d/%m/%Y'),
                 'France': datetime.strptime('11/05/2020', '%d/%m/%Y'),
                 'Germany': datetime.strptime('20/04/2020', '%d/%m/%Y'),
                 'Switzerland': datetime.strptime('27/04/2020', '%d/%m/%Y'),
                 'United Kingdom': datetime.strptime('13/05/2020', '%d/%m/%Y'),
                 'Russia': datetime.strptime('11/05/2020', '%d/%m/%Y'),
                 'US': datetime.strptime('04/05/2020', '%d/%m/%Y'),
                 'Sweden': datetime.strptime('04/05/2021', '%d/%m/%Y'),
                 'Belgium': datetime.strptime('04/05/2020', '%d/%m/%Y')}

for c in selected_countries:
    # Sample data before lockdown
    rows = countries_to_fit[(countries_to_fit['Country/Region'] == c) & (countries_to_fit['Date'] < lockdown_ends[c])]

    # The column "Active" is calculate as Total Confirmed Cases - Recovered - Deaths
    infected = np.array(rows['Active'])
    recovered = np.array(rows['Recovered'])
    deaths = np.array(rows['Deaths'])
    new_cases = np.array(rows['New cases'])

    removed = recovered + deaths

    countries_dict_prelock[c] = [infected, removed, new_cases]

    # Sample data after lockdown
    rows = countries_to_fit[(countries_to_fit['Country/Region'] == c) & (countries_to_fit['Date'] >= lockdown_ends[c])]
    infected = np.array(rows['Active'])
    recovered = np.array(rows['Recovered'])
    deaths = np.array(rows['Deaths'])
    new_cases = np.array(rows['New cases'])

    removed = recovered + deaths

    countries_dict_postlock[c] = [infected, removed, new_cases]

