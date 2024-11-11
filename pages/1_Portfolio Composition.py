# Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

st.title("Risk Analysis")

data = []
for ticker in st.session_state['weights'].keys():
  data.append(yf.Ticker(ticker))

vis0, vis1, vis2, vis3, vis4, vis5, vis6, vis7, vis8 = st.tabs(['Weights','Fundamental','Diversification','Key Ratios','News','Charts','Balance Sheet','Cash flow','Income Statement'])
with vis0:
  st.write('### Weights: ',st.session_state['weights'])

with vis1:
  st.write('### Fundamental Analysis')
  fund_df = pd.DataFrame()
  for t in data:
    try:
      info = {}
      for x in ['symbol','industry','sectorKey', 'enterpriseValue', 'beta', 'earningsGrowth', 'revenueGrowth', 'profitMargins', 'grossMargins', 'trailingPE', 'forwardPE', 'debtToEquity', 'returnOnAssets', 'returnOnEquity', 'quickRatio', 'currentRatio', 'freeCashflow', 'ebitdaMargins', 'heldPercentInsiders', 'heldPercentInstitutions', 'shortRatio','fiftyDayAverage', 'twoHundredDayAverage']:#,'priceToSalesTrailing12Months', 'currency', 'floatShares', 'sharesOutstanding', 'sharesShort', 'sharesShortPriorMonth', 'sharesShortPreviousMonthDate', 'dateShortInterest', 'sharesPercentSharesOut', 'shortPercentOfFloat', 'impliedSharesOutstanding', 'bookValue', 'priceToBook', 'lastFiscalYearEnd', 'nextFiscalYearEnd', 'mostRecentQuarter', 'earningsQuarterlyGrowth', 'netIncomeToCommon', 'trailingEps', 'forwardEps', 'pegRatio', 'enterpriseToRevenue', 'enterpriseToEbitda', '52WeekChange', 'SandP52WeekChange', 'exchange', 'quoteType','targetHighPrice', 'targetLowPrice', 'targetMeanPrice', 'targetMedianPrice','ebitda', 'totalDebt', 'totalRevenue',  'revenuePerShare', 'operatingCashflow', 'operatingMargins', 'financialCurrency', 'trailingPegRatio']:
        info[x] = t.info[x] 
      if len(fund_df)>0:
        fund_df = pd.concat([fund_df,pd.DataFrame.from_dict(info,orient='index',columns=[t.info['symbol']])],axis=1)
      else: 
        fund_df = pd.DataFrame.from_dict(info,orient='index',columns=[t.info['symbol']])
    except:
      pass
  st.session_state['fundamentals'] = fund_df
  st.write(fund_df)

from matplotlib import pyplot as plt
with vis2:
  st.write("### Portfolio Diversification")
  s1 = fund_df.T['sectorKey'].value_counts()
  s2 = fund_df.T['industry'].value_counts()
  fig, (ax1,ax2) = plt.subplots(2,1,figsize=(20,8))
  ax1.pie(s1,labels = s1.index)
  ax1.set_title('Sector')
  ax2.pie(s2,labels = s2.index)
  ax2.set_title('Industry')
  st.pyplot(fig)

with vis3:
  col1, col2 = st.columns(2)
  weights_df = pd.DataFrame.from_dict(st.session_state['weights'],orient='index')

  current_ratio = fund_df.T['currentRatio'].mul(weights_df.T).sum(axis=1)[0]
  current_goal = 1.5
  col1.metric('Weighted Current Ratio', current_ratio, delta=current_ratio-current_goal)

  debt_equity_ratio = fund_df.T['debtToEquity'].mul(weights_df.T).sum(axis=1)[0]/100
  debt_equity_goal = 1
  col1.metric('Weighted Debt/Equity', debt_equity_ratio, delta=debt_equity_ratio-debt_equity_goal, delta_color='inverse')

  gross_margin = fund_df.T['grossMargins'].mul(weights_df.T).sum(axis=1)[0]
  gross_goal = 0.4
  col1.metric('Weighted Gross Margins', gross_margin, delta=gross_margin-gross_goal)

  profit_margin = fund_df.T['profitMargins'].mul(weights_df.T).sum(axis=1)[0]
  profit__goal = 0.1
  col1.metric('Weighted Profit Margins', profit_margin, delta=profit_margin-profit__goal)

  return_equity = fund_df.T['returnOnEquity'].mul(weights_df.T).sum(axis=1)[0]
  return_equity_goal = 0.15
  col1.metric('Weighted Return on Equity', return_equity, delta=return_equity-return_equity_goal)

  insiders = fund_df.T['heldPercentInsiders'].mul(weights_df.T).sum(axis=1)[0]
  insiders_goal = 0.2
  col2.metric('Weighted %insiders', insiders, delta=insiders-insiders_goal)

  pe_ratio = fund_df.T['trailingPE'].mul(weights_df.T).sum(axis=1)[0]
  pe_goal = 20
  col2.metric('Weighted PE', pe_ratio, delta=pe_ratio-pe_goal, delta_color='inverse')
  
  change_pe_ratio = (fund_df.T['forwardPE']/fund_df.T['trailingPE']).mul(weights_df.T).sum(axis=1)[0]
  change_pe_goal = 0.5
  col2.metric('Weighted Change in PE', change_pe_ratio, delta=change_pe_ratio-change_pe_goal, delta_color='inverse')

  pcf_ratio = (fund_df.T['enterpriseValue']/fund_df.T['freeCashflow']).mul(weights_df.T).sum(axis=1)[0]
  pcf_goal = 20
  col2.metric('Weighted PCF', pcf_ratio, delta=pcf_ratio-pcf_goal, delta_color='inverse')

  trend = (fund_df.T['fiftyDayAverage']/fund_df.T['twoHundredDayAverage']).mul(weights_df.T).sum(axis=1)[0]
  trend_goal = 1
  col2.metric('Weighted Trend (50/200MA)', trend, delta=trend-trend_goal)

with vis4:
  st.write('### Related News')
  st.write('Macroeconomy trends')
  st.write('[Weekly global economic update](https://www2.deloitte.com/us/en/insights/economy/global-economic-outlook/weekly-update.html)')
  news_list = []
  for t in data:
    st.write(t.info['symbol'])
    try:
      news_list.append([t.news[x]['title'].replace('$','USD') for x in range(max(5,len(t.news)))])
      st.write(f"[{t.news[0]['title'].replace('$','USD')}]({t.news[0]['link']}) | [{t.news[1]['title']}]({t.news[1]['link']}) | [{t.news[2]['title']}]({t.news[2]['link']})")
    except:
      continue

with vis5:
  st.session_state['macro'] = []
  files = []
  for File in os.listdir("downloaded"):
    if File.endswith(".csv"):
      files += [File]
      st.session_state['macro'] += [pd.read_csv('downloaded/'+File).set_index('date').tail(120).to_json()]

  graphSel = st.selectbox('Macro Graph', files)
  st.line_chart(pd.read_csv('downloaded/'+graphSel).set_index('date').to_dict())


#t.funds_data.description
with vis6:
  balanceSheet_df = pd.DataFrame()
  for t in data:
    if len(balanceSheet_df)>0:
      balanceSheet_df = pd.concat([balanceSheet_df,t.balance_sheet.add_prefix(t.info['symbol']+'_')],axis=1)
    else: 
      balanceSheet_df = t.balance_sheet.add_prefix(t.info['symbol']+'_')
  balanceSheet_df
  
with vis7:
  cashFlow_df = pd.DataFrame()
  for t in data:
    if len(cashFlow_df)>0:
      cashFlow_df = pd.concat([cashFlow_df,t.cash_flow.add_prefix(t.info['symbol']+'_')],axis=1)
    else: 
      cashFlow_df = t.cash_flow.add_prefix(t.info['symbol']+'_')
  cashFlow_df

with vis8:
  incomeStmt_df = pd.DataFrame()
  for t in data:
    if len(incomeStmt_df)>0:
      incomeStmt_df = pd.concat([incomeStmt_df,t.income_stmt.add_prefix(t.info['symbol']+'_')],axis=1)
    else: 
      incomeStmt_df = t.income_stmt.add_prefix(t.info['symbol']+'_')
  incomeStmt_df


report = {
  'allocation weights for gemini to evaluate diversification, quantitative risks and returns': st.session_state['historical_stats'].to_json(),
  'fundamentals for gemini to evaluate company health, diverisification,': fund_df.to_json(),
  #'balanceSheet': balanceSheet_df.to_json(),
  #'cashflow': cashFlow_df.to_json(),
  #'incomeStatement': incomeStmt_df.to_json(),
  'macroeconomic trends for gemini to evaluate the market risks in short and long term': st.session_state['macro'],
  'recent news for gemini to evaluate the market sentiment and risks surrounding selected companies': news_list,
}

with st.sidebar:
  st.download_button('Download Diagnostic Report', json.dumps(report), 'result.json')

st.session_state.diagnosis = report