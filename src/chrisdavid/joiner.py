import pandas as pd
from dateutil.parser import parse

fd_nsw = pd.read_csv('forecastdemand_nsw.csv')
fd_vic = pd.read_csv('forecastdemand_vic.csv')
fd_sa = pd.read_csv('forecastdemand_sa.csv')
fd_qld = pd.read_csv('forecastdemand_qld.csv')

temp_nsw = pd.read_csv('temperature_nsw.csv')
temp_vic = pd.read_csv('temperature_vic.csv')
temp_sa = pd.read_csv('temperature_sa.csv')
temp_qld = pd.read_csv('temperature_qld.csv')

td_nsw = pd.read_csv('totaldemand_nsw.csv')
td_vic = pd.read_csv('totaldemand_vic.csv')
td_sa = pd.read_csv('totaldemand_sa.csv')
td_qld = pd.read_csv('totaldemand_qld.csv')

files = [fd_nsw,fd_vic,fd_sa,fd_qld,temp_qld,temp_sa,temp_nsw,temp_vic,td_qld,td_vic,td_sa,td_nsw]
for file in files:
    file['timestamp'] = file['DATETIME'].apply(parse)

full_nsw = fd_nsw.merge(temp_nsw,on='timestamp', how='inner').merge(td_nsw,on='timestamp', how='inner')
full_vic = fd_vic.merge(temp_vic,on='timestamp', how='inner').merge(td_vic,on='timestamp', how='inner')
full_sa = fd_sa.merge(temp_sa,on='timestamp', how='inner').merge(td_sa,on='timestamp', how='inner')
full_qld = fd_qld.merge(temp_qld,on='timestamp', how='inner').merge(td_qld,on='timestamp', how='inner')

full_nsw['state'] = "NSW"
full_vic['state'] = "VIC"
full_qld['state'] = "QLD"
full_sa['state'] = "SA"



full = pd.concat([full_nsw,full_vic,full_sa,full_qld], ignore_index=True)
full = full[['timestamp','state','FORECASTDEMAND','TEMPERATURE','TOTALDEMAND']]
full.columns = ['TIMESTAMP','STATE','FORECASTDEMAND','TEMPERATURE','TOTALDEMAND']

full.to_csv('full_data.csv')

