`expected_return.csv`
- 65 columns, last column is the date of trading
- date starts 2012-01-04 and ends on 2021-12-30
- each of the first 64 columns corresponds to a different stock ticker, and the value is the daily return as a fraction
- there are no missing values

`side_info.csv`
- 71 columns
- first 64 columns correspond to the same 64 stock tickers in expected_return.csv
    - values are the volume of trading
- 65th column is the date of trading (this column is identical to the DATE column in expected_return.csv)
- last 6 columns correspond to various stock indices
    - values are the stock index price
    - values are missing for 3 dates: row 1199 (2016-10-10), row 1223 (2016-11-11), row 1526 (2018-01-29)
