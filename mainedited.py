
# coding: utf-8

# In[ ]:

import urllib2 
import json
import time
import pandas as pd
import os
import datetime
k = datetime.datetime.now()
 
def fetchPreMarket(symbol, exchange):
    link = "http://finance.google.com/finance/info?client=ig&q="
    url = link+"%s:%s" % (exchange, symbol)
    u = urllib2.urlopen(url)
    #print url
    content = u.read()
    data = json.loads(content[3:])
    info = data[0]
    t = str(info["lt"])    # time stamp
    l = float(info["pcls_fix"])    # close price (previous trading day)
    p = float(info["l_cur"])   # stock price during trade hours
    return (t,l,p)
 
 

data = {"time_stamps":[],
    "previous_closeprice":[],
    "current_price":[],
    "variation":[],
    "magnitude_change":[]}

count = 0

if (k.hour > 10 and k.hour < 16) or (k.hour == 9 and k.minutes > 30):   #EDT Timings... For NasDaq trade hours.  
    while True:

        t, l, p = fetchPreMarket("AAPL","NASDAQ")
        if p > 0:
            count+=1
            print("%s\t%.2f\t%.2f\t%+.2f\t%+.2f%%" % (t, l, p, p-l,
                                                     (p/l-1)*100.))
            data["time_stamps"].append(t)
            data["previous_closeprice"].append(l)
            data["current_price"].append(p)
            data["variation"].append(p-l)
            data["magnitude_change"].append((p/l-1)*100.0)

            if count ==10:
                df = pd.DataFrame(data, columns = ["time_stamps", "previous_closeprice", "current_price", 'variation', "magnitude_change"])
                if not os.path.isfile('example.csv'):
                    df.to_csv('example.csv',header ='column_names')
                else:
                    df.to_csv('example.csv',mode = 'a',header=False)
                df.to_csv('example1.csv') 

                count = 0
                data = {"time_stamps":[],
                        "previous_closeprice":[],
                        "current_price":[],
                        "variation":[],
                        "magnitude_change":[]}

        time.sleep(60)
        
else:
    print "Not the usual trading hours"


