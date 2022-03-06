import pandas as pd

df = pd.read_csv('label.txt',sep=' ',names=['driver','trip','0','1','face','cabin','frame'])

events = []
for i,row in df.iterrows():
    driver = row['driver']
    trip = row['trip']
    starttime = row['face'].split('_')[3].split('.')[0]
    endtime = int(starttime)+int(row['frame']*10)
    events.append([driver,trip,starttime,endtime])
out = pd.DataFrame(events)
out.columns=['driver','trip','starttime','endtime']
out = out.drop_duplicates()
out.to_csv('fhwa_events.csv')
