# Trying to apply the prophet library on some Google Trends data to compare the results
# Note: the data were downloaded manually from Google trends. Pytrends, the unofficial api, seems broken or discontinued

import matplotlib.pyplot as plt 
import pandas as pd
from fbprophet import Prophet

# Csv selection
difficulties = ["1-easy", "2-medium", "3-hard", "4-no-pattern"]
difficulty = difficulties[3]
filename = 'machinelearning-wr.csv'

# Read some data and create an instance of Prophet
df = pd.read_csv('data/' + difficulty + "/" + filename)
df.head()
# Basic model
#model = Prophet()
# Tuned model
model = Prophet(interval_width=0.95, yearly_seasonality=True)
# Add standard US holidays
#model.add_country_holidays(country_name='US')

model.fit(df)

# Predict the next 24 months (default: days)
future = model.make_future_dataframe(periods=24, freq='M')

forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# Render the forecast plot
fig1 = model.plot(forecast)
plt.show()

# Render the trends plot
fig2 = model.plot_components(forecast)
plt.show()
