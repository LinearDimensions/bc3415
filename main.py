# Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import skew, kurtosis
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

import json
import helpful_functions as hf

# Set up Streamlit app
st.set_page_config(page_title="Gimme BC3415 A++", layout="wide")

# Side Bar for Parameters and Weights Allocation
with st.sidebar:
        
    # Create Tabs for customisation
    tab1, tab2 = st.tabs(["Upload Portfolio ⬆️", "Create Portfolio ➕"])

    with tab1:
        weight_dict = st.file_uploader("Portfolio with Tickers and Weights", type=["json"])
        if weight_dict != None:
            weight_dict = json.loads(weight_dict.getvalue().decode("utf-8"))
            st.session_state['weights'] = weight_dict
        else:
            weight_dict = {}

    with tab2:
        stockList = pd.read_csv("spy_price(2024).csv").set_index('Date').columns
        options = st.multiselect("Enter Stocks Tickers",stockList)
        sigma_p = st.select_slider('Risk Appetite (Drawdown tolerable %)',options=range(51),value=30)/300

        with st.container(height=200):#min(470,230+100*len(options))):
            for option in options:
                weight_dict[option] = st.select_slider(option+' weightage (%)',options=range(101))
        if len(options) > 0:
            st.download_button('Download Json', json.dumps(weight_dict), 'portfolio.json')

    #@st.dialog('Portfolio', width='large')
    def display(weights,sigma_p=0.1):
        weight_dict = dict(weights)
        year = st.radio('No. of Years', [1,3,5,10], horizontal=True)
        df = hf.selectStocks(weight_dict.keys())
        plot,optimalW,optimal,portfolio = hf.stResults(
            df.fillna(df.shift(1))[-year*252:],
            weights= weight_dict,
            query='coverage>=0 & returns>0',
            freq=31,
            sigma_p = sigma_p
            )
        col1, col2 = st.columns(2)

        col1.plotly_chart(px.line(plot))
        col2.write(optimalW.drop('coverage',axis=1).rename({'weightsOptimal':'wOpt','weightsPortfolio':'wPort'}))
        tabOptimal, tabCurrent = col2.tabs(["Optimal / Recommended Allocation", "Your Porfolio Allocation"])
        with tabOptimal:
            tabOptimal.write(f'**Optimal Returns**: {optimal[0]:.4f} **Optimal VolSqr**: {optimal[1]:.4f}')
            if tabOptimal.button("Select Optimal Weights"):
                st.session_state['historical_stats'] = optimalW.drop('weightsPortfolio',axis=1)
                st.session_state['weights'] = optimalW.weightsOptimal.to_dict()
            
        with tabCurrent:
            tabCurrent.write(f'**Porfolio Returns**: {portfolio[0]:.4f}  **Portfolio VolSqr**: {portfolio[1]:.4f}')
            if tabCurrent.button("Select Portfolio Weights"):
                temp = np.array(list(weight_dict.values()))
                temp = temp/temp.sum()
                weights = dict(zip(list(weight_dict.keys()),temp))
                st.session_state['historical_stats'] = optimalW.drop('weightsOptimal',axis=1)
                st.session_state['weights'] = weights
        return "Proceed to Portfolio Analysis"



    if weight_dict:
        portfolio = pd.DataFrame.from_dict(weight_dict,orient='index',columns=['weight'])
        
    else:
        st.write("Please upload a portfolio CSV to begin analysis.")


# Main App
st.title("Portfolio Analysis")
st.write("Prepared as Quant Portfolio Analysis™ tool for BC3415 Project")
if weight_dict != None and weight_dict != {}:
    display(weight_dict.items(),sigma_p)










# Define functions for analysis
def calculate_ratios(data):
    ratios = {
        "Mean Return": data['Return'].mean(),
        "Volatility": data['Return'].std(),
        "Sharpe Ratio": data['Return'].mean() / data['Return'].std(),
        "Skewness": skew(data['Return']),
        "Kurtosis": kurtosis(data['Return']),
    }
    return pd.DataFrame(ratios, index=[0])


def diversification_analysis(data):
    sector_count = data['Sector'].value_counts()
    st.write("### Portfolio Diversification by Sector")
    st.bar_chart(sector_count)

def plot_correlation_matrix(data):
    corr = data.corr()
    st.write("### Correlation Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

def options_strategies(stock_data):
    st.write("### Options Strategy Suggestions")
    st.write("Select strategies based on implied volatility, stock momentum, etc.")
    st.write("Example: Covered Call for income generation or Protective Put for downside protection.")



## python -m streamlit run main.py
## python -m  pipreqs.pipreqs