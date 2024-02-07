### Reproducing the Results
- Unzip data and store in folder titled 'data' within workspace or change csv references in main
- Requires installation of [plotly](https://plotly.com/python/getting-started/) (find Installation header) and [scikit-learn](https://scikit-learn.org/stable/install.html)
- Open `data_analysis.py` and run the main method. Interactive graphs will open in your browser and\
the dataframes will be printed. Optionally, uncomment `pd.set_option('display.max_rows', None)`\
within the main method to show all rows of the dataframes.
  - `get_full_data` to get tuple of the final race and age data
  - `percentage_income` to get percentage of income spent on health for each subgroup and year
  - `future_results` to run linear regression showing how the percentage is trending
  - `plot_charts` to show the results of the data analysis
  
CSE 163 Final Project - Sp22

Marcus Jundt & Mitchell Stapelman

#### Slide deck: https://docs.google.com/presentation/d/1zodWMppnY7oW4OcZ9DSCcgYz60bF3r22XitmWTrXC08/edit?usp=drivesdk
