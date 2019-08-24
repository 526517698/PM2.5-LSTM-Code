Data Origin: https://www.kaggle.com/uciml/pm25-data-for-five-chinese-cities

Data Files:
  BeijingPM20100101_20151231.csv (This one is what is used in code)
  ChengduPM20100101_20151231.csv
  GuangzhouPM20100101_20151231.csv
  ShanghaiPM20100101_20151231.csv
  ShenyangPM20100101_20151231.csv
  
Data Content:
  The time period for this data is between Jan 1st, 2010 to Dec 31st, 2015. Missing data are denoted as NA.
  No: row number
  year: year of data in this row
  month: month of data in this row
  day: day of data in this row
  hour: hour of data in this row
  season: season of data in this row
  PM: PM2.5 concentration (ug/m^3), PM_XX means the data is from XX company, PM_US_POST is used in code
  DEWP: Dew Point (Celsius Degree)
  TEMP: Temperature (Celsius Degree)
  HUMI: Humidity (%)
  PRES: Pressure (hPa)
  cbwd: Combined wind direction
  Iws: Cumulated wind speed (m/s)
  precipitation: hourly precipitation (mm)
  Iprec: Cumulated precipitation (mm)
  
Plot Image:
  Beijing_USPOST_PM2.5 Origin.png  (The original data plotted)
  Beijing_USPOST_PM2.5 Shrinked.png   (The orginal data after convert from hourly basis to daily basis)
  Beijing_USPOST_PM2.5 Shrinked with Moving mean and std.png   (The shrinked data with moving mean and standard deviation)
  
  
LSTM & RNN Code:
  First part(line 1 to 54): reading and modifying the image
  Second part(line 69 to end): Code for training LSTM and RNN

Training Result:
  training result final.csv consist of result loss of Training data for the following data:
    batch_size and window_size: 1, 4, 7, 10, 13, 19
    hidden_layer: 200, 230, 260, 290, 320, 350, 380
    clip_margin: 4
    learning_rate = 0.001
    epochs = 1000.0
    
    
    
    
    
    
    
