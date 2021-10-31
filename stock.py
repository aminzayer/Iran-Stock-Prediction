import pytse_client as tse
from pytse_client import download_client_types_records
import matplotlib.pyplot as plt
from tkinter import *
import arabic_reshaper
from bidi.algorithm import get_display
import pandas as pd

ticker = tse.download(symbols="نوری", write_to_csv=True, include_jdate=True)
ticker = tse.Ticker("نوری")

reshaped_text = arabic_reshaper.reshape('نمایش قیمت و حجم سهام با نماد نوری')
Prtext = get_display(reshaped_text)

history = ticker.history
x1=history["date"]
y1=history["close"]
x2=x1
y2=history["volume"]

plt.plot(x1, y1, label = "line 1")
#plt.plot(x2, y2, label = "line 2")
#plt.bar(x2,y2)
plt.xlabel('x - Days')
plt.ylabel('y - Price & Volume')
plt.title(Prtext)
#plt.legend()
plt.show()