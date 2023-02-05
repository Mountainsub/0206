# Machine learning
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import mstats

# For data manipulation
import pandas as psd
import numpy as np
import datetime
import yfinance as yf
# load correct trading day format
format_list = pd.read_csv('C:/Users/kose/Desktop/J-REIT/J-REIT Example/Trading day format_jp_202206.csv', index_col=0, parse_dates=True)

# load credit rating format
credit_rating = pd.read_csv('C:/Users/kose/Desktop/J-REIT/J-REIT Example/Rating score.csv',parse_dates=True)
credit_score = dict(zip(credit_rating['Rating'], credit_rating['Score']))

# read different lenghth for cutting

# load data length prior 2006
data_prior_2006 = pd.read_csv('C:/Users/kose/Desktop/J-REIT/J-REIT Example/j-reit-starttime_2006.csv', index_col=0)
# input the ticker
ticker = 8955

# the number of data to skip before 2006, refer to J-REIT summary for list
len_to_skip = data_prior_2006.loc[ticker,'StartBefore2006']


ticker = str(ticker)
ticker_symbol = ticker + '.T'  # for Japanese ticker only

# read quarterly data
fundamental_data = pd.read_excel('C:/Users/kose/Desktop/J-REIT/J-REIT Example/Stock data_20220631_jreit_tobeprocessed_only_8955.xlsx', sheet_name=ticker, index_col=0, parse_dates=True)
# use only rows without NA, because J-reit is reported semi-annually
fundamental_data_2 = fundamental_data.dropna(subset = ['TOTAL_EQUITY'])

#     Load saved share price file
file_path = 'C:/Users/kose/Desktop/J-REIT/J-REIT Example/'
file_name = ticker
file_extension = '_price.csv'
price_list = pd.read_csv(file_path + file_name + file_extension, index_col=0, parse_dates=True)           

# use correct format's index to retrieve the trading data
post_fundamental = fundamental_data_2[len_to_skip:]  # ie. For 8951 start from 2006/3/31, ends by 2021/12/31,

idx = np.searchsorted(price_list.index, post_fundamental.index)
post_price = price_list.iloc[idx]

volume = post_price.loc[:,'Volume'].values
close_prices = post_price.loc[:,'Adj Close'].values

equity = post_fundamental.loc[:,'TOTAL_EQUITY'].values
EPS = post_fundamental.loc[:,'IS_DILUTED_EPS'].values
shares = post_fundamental.loc[:,'BS_SH_OUT'].values
FCF = post_fundamental.loc[:,'FREE_CASH_FLOW_PER_SH'].values
market_cap = post_fundamental.loc[:,'HISTORICAL_MARKET_CAP'].values
ROE = post_fundamental.loc[:,'RETURN_COM_EQY'].values
AssetTurnover = post_fundamental.loc[:,'ASSET_TURNOVER'].values
leverage = post_fundamental.loc[:,'FNCL_LVRG'].values
profit_margin = post_fundamental.loc[:,'PROF_MARGIN'].values
CFO = post_fundamental.loc[:,'CF_CASH_FROM_OPER'].values
CFI = post_fundamental.loc[:,'CF_CASH_FROM_INV_ACT'].values
CFF = post_fundamental.loc[:,'CFF_ACTIVITIES_DETAILED'].values
ebitda = post_fundamental.loc[:,'EBITDA'].values
FFO = post_fundamental.loc[:,'CF_FFO_PER_SH'].values
payout = post_fundamental.loc[:,'DVD_PAYOUT_RATIO'].values
sus_growth = post_fundamental.loc[:,'SUSTAIN_GROWTH_RT'].values
RI_rating = post_fundamental.loc[:,'RTG_RI_ISSUER_RATING'].values
JCR_rating = post_fundamental.loc[:,'RTG_JCR_LT_ISSUER'].values

# transform credit rating to scores
RI_rating_score =[]
for i in range(len(RI_rating)):
    if pd.isna(RI_rating[i]) == False:
        score = credit_score[RI_rating[i]]
    else:
        score = RI_rating[i]
    RI_rating_score.append(score)

JCR_rating_score =[]
for i in range(len(JCR_rating)):
    if pd.isna(JCR_rating[i]) == False:
        score = credit_score[JCR_rating[i]]
    else:
        score = RI_rating[i]
    JCR_rating_score.append(score)
#-----

BPS = equity / shares
EBITDA_per_share = ebitda / shares
PER = close_prices / EPS
PBR = close_prices / BPS
PCFO = close_prices / CFO
PCFI = close_prices / CFI
PFCF = close_prices / FCF
PFFO = close_prices / FFO

# Goal is to anticipate the sign of future earnings change from the financial data of the current quarter.
# If the future earnings changes is + , we assign 1, otherwise 0,  to Future change value of the current quarter

FFO_change = np.where(FFO[1:] > FFO[0:-1], 1, 0)
FFO_change = np.append(FFO_change,1)   # adjustment, not being used

# predict the FFO direction

data_2 = []
data_2 = pd.DataFrame(data_2)
data_2['PBR'] = PBR
data_2['market_cap'] = market_cap
data_2['PER'] = PER
data_2['AssetTurnover'] = AssetTurnover

X = data_2.values
y = FFO_change

train_len = 7

ffo_prediction = []
for i in range(len(y)-train_len+1):

    X_train = X[0:train_len+i-1,:]    # train until previous period, cuz y is forecasting next period
    y_train = y[0:train_len+i-1]
    X_predict = X[0:train_len+i,:] # use most recent reporting period for forecasting

    # Winsorize top 1% and bottom 1% of points.
    # Apply on X_train and X_test separately
    X_train = mstats.winsorize(X_train, limits = [0.01, 0.01])
    X_predict = mstats.winsorize(X_predict, limits = [0.01, 0.01])

    sc = StandardScaler()
    # Fit to training data and then transform it
    X_train = sc.fit_transform(X_train)
    X_predict = sc.transform(X_predict)

    # Initialize svm, rbf is a default kernel
    classifier_rbf = SVC(C = 1, kernel = 'rbf', gamma = 'auto', random_state = 0)
    # Fit the model on training data
    classifier_rbf.fit(X_train, y_train)
    # Make a prediction on testing data
    y_pred_rbf = classifier_rbf.predict(X_predict[train_len -1+i,:].reshape(1,-1))

    ffo_prediction = np.append(ffo_prediction,y_pred_rbf)

result = []
result = pd.DataFrame(result)
result['ffo_predict'] = ffo_prediction
result['PFFO'] = PFFO[train_len-1:]/2

# Export data to CSV file
export = True
file_path = 'C:/Users/kose/Desktop/J-REIT/J-REIT Example/prediction/'
file_name = ticker
file_extension = '.csv'

#     new_copy.to_csv(file_path + file_name + file_extension)
if export:
    data = result.set_index(post_fundamental.index[train_len-1:])
    data.to_csv(file_path + file_name + file_extension)
                    