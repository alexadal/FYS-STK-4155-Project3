from DD import *
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import datetime

google_username = ""
google_password = ""
path = ""

# Login to Google. Only need to run this once, the rest of requests will use the same session.
pytrend = TrendReq(google_username, google_password)


#sc = StandardScaler()
sc = MinMaxScaler()

gtrends = get_daily_data('short S&P500',2013,1,2019,11)
gtrends = gtrends.drop(['isPartial'], axis='columns')
df_scaled = gtrends.copy(deep=True)
#df_scaled['S&P500'] = sc.fit_transform(df_scaled['S&P500'].values.reshape(-1,1))
print("hi")
df_scaled.to_csv("short_S&P500_trend.csv", index=False)






