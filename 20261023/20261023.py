from os import environ

import yfinance as yf


tickers = [
    'BCD', 'BITW', 'BWX', 'CAOS', 'CTA', 'DBMF', 'EMLC', 'FOXY', 'IMF',
    'SGOL', 'STIP', 'TLT', 'USDU', 'VEA', 'VTI', 'VWO', 'WIP'
]
for ticker in tickers:
    etf = yf.Ticker(ticker)
    df = etf.history(period="max", interval="1d", auto_adjust=True)
    bwd = environ['BUILD_WORKSPACE_DIRECTORY']
    csv_filename = f"{bwd}/{ticker}.csv"
    df.to_csv(csv_filename)
    json_filename = f"{bwd}/{ticker}.json"
    df.to_json(json_filename, orient="index", date_format="iso")
