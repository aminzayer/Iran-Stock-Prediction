# Import Python libs
import pytse_client as tse
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display
import pandas as pd
import pandas_ta as ta


# fetch one of Tricker Stock Price
ticker = tse.download(symbols="نوری", write_to_csv=True, include_jdate=True)
ticker = tse.Ticker("نوری")

# Preprocessing Stock Persian Name (UTF-8)
reshaped_text = arabic_reshaper.reshape('نمایش قیمت و حجم سهام با نماد نوری')
Prtext = get_display(reshaped_text)

#Preprocessing & clean Data & Create Data Frame
df=ticker.history[['jdate', 'volume' ,'close']]  # Fetch History of trades & clean data
df.ta.ema(close='close',length=10,append=True)   # Add Technical feature EMA 
df=df.iloc[10:]                                # Drop 10 NaN Values


