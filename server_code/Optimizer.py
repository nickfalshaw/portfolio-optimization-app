import anvil.tables as tables
import anvil.tables.query as q
from anvil.tables import app_tables
from anvil import tables
import anvil.mpl_util
import anvil.server
import numpy as np
import pandas as pd
from pandas_datareader import data
import matplotlib.pyplot as plt
import yfinance as yf
#import finnhub
import datetime
import time
import torch
from dateutil.relativedelta import relativedelta
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
import plotly.express as px

start_year = 2019
# # Fucntion to get stock data from yfinance
def get_stock_data(stock_Tokens,start_Date,update_delta):
  raw_Stock_Data = yf.Tickers(stock_Tokens)
  end_Date = start_Date+update_delta
  time_Series_Data = raw_Stock_Data.history(start=start_Date,end=end_Date)
  #adj_Close_Price = time_Series_Data["Close"].dropna(axis=0).dropna(axis = 1)
  #adj_Open_Price = time_Series_Data["Open"].dropna(axis=0).dropna(axis = 1)
  adj_Close_Price = time_Series_Data["Close"].dropna(axis=1)
  adj_Open_Price = time_Series_Data["Open"].dropna(axis=1)
  
  #ASK MOSHE IF IT IS OKAY TO CHANGE THIS TO PERCENT (divide by open)
  price_derivative = (adj_Close_Price-adj_Open_Price)/adj_Open_Price
  price_der = price_derivative.to_numpy()
  stock_Tokens = adj_Close_Price.columns.tolist()
  return end_Date,adj_Close_Price,price_der,stock_Tokens

# def get_stock_data(stock_Tokens,start_Date,update_delta):
#     raw_Stock_Data = yf.Tickers(stock_Tokens)
#     end_Date = copy.copy(start_Date)+update_delta
#     time_Series_Data = raw_Stock_Data.history(start=start_Date,end=end_Date)
#     tsd_close = time_Series_Data["Close"].dropna(axis=1)
#     tsd_open = time_Series_Data["Open"].dropna(axis=1)
#     pos = tsd_close.iloc[-1,:]-tsd_close.iloc[0,:]
#     if(sum(pos>0)==0):
#         pos =np.abs(pos)
#     adj_Close_Price = tsd_close.loc[:,pos>0]
#     price_derivative = tsd_close.loc[:,pos>0]-tsd_open.loc[:,pos>0]
#     price_der = price_derivative.to_numpy()
#     stock_Tokens = adj_Close_Price.columns.tolist()
#     return end_Date,adj_Close_Price,price_der,stock_Tokens

# Functions to calculate frontier weights after stocks are selected
def find_optimal_portfolio(stock_tickers,start_year):

  #Read data from yahoo finance
  raw_Stock_Data = yf.Tickers(stock_tickers)

  #Segregate data
  #start_year = 2018
  #start_Date = datetime.datetime(start_year, 1, 1)
  #end_Date = datetime.datetime(start_year, 12, 31);
  end_Date = datetime.datetime.today()-relativedelta(days=1)
  start_Date = end_Date-relativedelta(years=1)
  time_Series_Data = raw_Stock_Data.history(start=start_Date,end=end_Date)
  #time_Series_Data.head
  adj_Close_Price = time_Series_Data["Close"]
  #adj_Close_Price.head

  #Adjust to percent gain/lose
  percent_Change = adj_Close_Price.pct_change().apply(lambda x: np.log(1+x))
  fin_Data = percent_Change

  #Calculate variance of data
  var_fin_Data = fin_Data.var()
  #var_fin_Data.head

  #Calculate volatility of data
  trading_Days_Year = 251
  vol_Fin_Data = np.sqrt(fin_Data*trading_Days_Year)

  #Calculate covariance of data
  cov_fin_Data = fin_Data.cov()

  #Calculate correlation coefficients of data
  corr_fin_Data = fin_Data.corr()

  #Only yearly for the sampled one year of data
  #yearly_Returns = fin_Data.resample('Y').sum()
  #THIS CALCULATION IS POTENTIALLY WRONG. Rethink this maybe?
  yearly_Percent_Change = (adj_Close_Price.iloc[len(adj_Close_Price)-1,:]-(adj_Close_Price.iloc[1,:]))/adj_Close_Price.iloc[1,:]

  #print(yearly_Percent_Change)

  #Initialize returns and volatility matrixes 
  portfolio_Returns = []
  portfolio_Volatility = []
  stored_Portfolio_Weights = []
  num_Stocks = len(fin_Data.columns)
  iterations = 10000

  for i in range(iterations):
    #Implement random weights and normalize 
    portfolio_Weights = np.random.random(num_Stocks)
    portfolio_Weights = portfolio_Weights/(np.sum(portfolio_Weights))
    stored_Portfolio_Weights.append(portfolio_Weights)
    #print(portfolio_Weights)

    returns = np.dot(portfolio_Weights,yearly_Percent_Change)
    portfolio_Returns.append(returns)
    #print(returns)

    variance = cov_fin_Data.mul(portfolio_Weights,axis=0).mul(portfolio_Weights,axis=1).sum().sum()
    #print(variance)

    volatility = np.sqrt(variance)*np.sqrt(trading_Days_Year)
    portfolio_Volatility.append(volatility)
    #print(volatility)

  #Define the dataframe
  global portfolios
  portfolios = pd.DataFrame({'Returns':portfolio_Returns,'Volatility':portfolio_Volatility, 'Weights':stored_Portfolio_Weights})

  #Sharpe ratio optimality calculation
  risk_Factor = .02
  global optimal_Portfolio
  optimal_Portfolio = portfolios.iloc[((portfolios['Returns']-risk_Factor)/portfolios['Volatility']).idxmax()]
  
  insert_optimal_weights_into_table(stock_tickers)
  insert_plot_into_table()

  returns = '{:.2%}'.format(optimal_Portfolio.Returns)
  volatility = '{:.2%}'.format(optimal_Portfolio.Volatility)

  #CALCULATIONS AND DATA PULL FOR THE SIMULATED NEXT YEAR
  #sim_year = start_year + 1
  #start_Date = datetime.datetime(sim_year, 1, 1)
  #end_Date = datetime.datetime(sim_year, 12, 31);
  #time_Series_Data = raw_Stock_Data.history(start=start_Date,end=end_Date)
  #adj_Close_Price = time_Series_Data["Close"]
  #yearly_Percent_Change = (adj_Close_Price.iloc[len(adj_Close_Price)-1,:]-(adj_Close_Price.iloc[1,:]))/adj_Close_Price.iloc[1,:]
  #what variable is he correct portfolio weights and how do you output it
  #sim_returns_next_year = np.dot(optimal_Portfolio.Weights,yearly_Percent_Change)
  sim_returns_next_year = 0
  
  return ("Optimization ran successfully." + "\n" + "Number of stocks selected: " + str(len(stock_tickers)) + "\n" + ', '.join(stock_tickers)), returns, volatility#, f"{100*sim_returns_next_year:.2f} %"

def insert_optimal_weights_into_table(stock_tickers):
  app_tables.optimal_weights.delete_all_rows()
  ticker_index = 0
  for weight in optimal_Portfolio.Weights:
    weight_pct = '{:.2%}'.format(weight)
    app_tables.optimal_weights.add_row(Ticker=stock_tickers[ticker_index], Weight=weight_pct, raw_weight=weight)
    ticker_index = ticker_index + 1  

def insert_plot_into_table():
  app_tables.matlab_plots.delete_all_rows()
  plt.clf()
  plt.subplots()
  plt.scatter(portfolios['Volatility'],portfolios['Returns'],marker='o',s=5)
  plt.scatter(optimal_Portfolio[1],optimal_Portfolio[0],color='r',marker='*',s=50)
  plt.xlabel("Volatility")
  plt.ylabel("Returns")
  plt.title("Efficient Frontier Graph")
  efficient_frontier = anvil.mpl_util.plot_image()
  app_tables.matlab_plots.add_row(Plots=efficient_frontier)

@anvil.server.background_task
def optimize_portfolio(method,start_year):
  tech_Stock_Tokens = ['TSLA','META','AAPL','MSFT','GOOG','AMZN','TSM','NVDA','BABA','AVGO','ORCL','ASML','CSCO','CRM','TXN','ADBE','QCOM','AMD','INTC','IBM']
  fin_Stock_Tokens = ['JPM','WFC','MS','SCHW','AXP','BLK','CB','PGR','USB','BX','BAC','QQQ','RY','C','GS','CI','BAM','CME','PNC','MA']
  energy_Stock_Tokens = ['XOM','CVX','SHEL','COP','TTE','BP','PBR','ENB','CNQ','OXY','PXD','EPD','MPC','TRP','DVN','KMI','CLR','BKR','EQT','TS']
  #list_2017 = ['AAPL',  'ABBV',  'AMGN',  'GILD',  'REGN',  'VRTX',  'MU',  'MRNA',  'BIIB',  'BNTX',  'MCHP',  'MRVL',  'CTSH',  'HPQ',  'DELL',  'ALNY',  'GMAB',  'VEEV',  'SGEN',  'CAJ',  'NEE',  'SLB',  'DUK',  'D',  'DVN',  'VLO',  'SU',  'SRE',  'WDS',  'LNG',  'TRP',  'ENPH',  'CVE',  'ET',  'XEL',  'HAL',  'CEG',  'CQP',  'WEC',  'BKR',  'AGM',  'AIHS',  'ALLY',  'ARI',  'ATAX',  'ATLC',  'AX',  'AXP',  'BCOW',  'BCSF',  'BSBK',  'BWB',  'BYFC',  'CACC',  'CARV',  'CASH',  'CFBK',  'CFFN',  'CLBK',  'CNF'] 
  stock_tickers = tech_Stock_Tokens + fin_Stock_Tokens+ energy_Stock_Tokens
  #if(start_year==2017):
    #stock_tickers = list_2017
  #output from the using the finnhub api to filter invalid tickers for over 1000 tech, fin, and energy stocks
  #200 working tickers from 2017-2020
  #stock_tickers = ['AGM', 'AIHS', 'ALLY', 'ARI', 'ATAX', 'ATLC', 'AX', 'AXP', 'BCOW', 'BCSF', 'BSBK', 'BWB', 'BYFC', 'CACC', 'CARV', 'CASH', 'CFBK', 'CFFN', 'CLBK', 'CNF', 'COF', 'COOP', 'CPSS', 'CURO', 'DCOM', 'DFS', 'ECPG', 'ELVT', 'ENVA', 'ESNT', 'ESSA', 'EZPW', 'FBC', 'FCFS', 'FFBW', 'FINV', 'FOA', 'FSBW', 'FSEA', 'GCBC', 'GDOT', 'GHLD', 'HBCP', 'HFBL', 'HIFS', 'HMNF', 'HMST', 'HRZN', 'HVBC', 'IMH', 'IOR', 'IROQ', 'JFIN', 'JT', 'KFFB', 'KREF', 'KRNY', 'LBC', 'LC', 'LDI', 'LFT', 'LRFC', 'LSBK', 'LU', 'LX', 'MBIN', 'MFIN', 'MGYR', 'MOGO', 'MSVB', 'MTG', 'NAVI', 'NECB', 'NFBK', 'NICK', 'NMFC', 'NMIH', 'NNI', 'NREF', 'NWBI', 'NYCB', 'OCFC', 'OCN', 'OFED', 'OMF', 'OPBK', 'OPRT', 'PBFS', 'PCSB', 'PDLB', 'PFC', 'PFS', 'PFSI', 'PRAA', 'PROV', 'PT', 'PTMN', 'PVBC', 'QD', 'RBKB', 'RDN', 'RKT', 'RM', 'RVSB', 'SACH', 'SBT', 'SLM', 'SMBC', 'SNFCA', 'SOFI', 'SYF', 'TBNK', 'TFSL', 'TREE', 'TRST', 'TRTX', 'TSBK', 'UPST', 'UWMC', 'VEL', 'WAFD', 'WD', 'WHF', 'WLFC', 'WMPN' ,'WNEB', 'WRLD', 'WSBF', 'WSFS', 'XYF', 'YRD', 'NEE', 'SLB', 'DUK', 'D', 'DVN', 'VLO', 'SU', 'SRE', 'WDS', 'LNG', 'TRP', 'ENPH', 'CVE', 'ET', 'XEL', 'HAL', 'CEG', 'CQP', 'WEC', 'BKR', 'FANG', 'ES', 'CTRA', 'FE', 'DTE', 'TS', 'CNP', 'CMS', 'ATO', 'LNT', 'CHK', 'NFE', 'NRG', 'NOV', 'SWN', 'OGE', 'GPOR', 'TALO', 'NS', 'USAC', 'FLNC', 'SDRL', 'NBR', 'CLNE', 'ERII', 'TDW', 'GEL', 'OII', 'FCEL', 'PUMP', 'AROC', 'UUUU', 'SLCA', 'PDS', 'WTTR', 'HLX', 'DO', 'CLB', 'BROG', 'DRQ', 'OBE', 'SD', 'LEU', 'NESR', 'VTNR', 'SXC', 'EGY', 'TTI', 'SOI']#, 'TNP', 'GTE', 'ASPN', 'TETC', 'ZT', 'OIS', 'BOOM', 'REI', 'NRGV', 'TYG', 'AMPY', 'NETC', 'NR', 'NOA', 'PEGR', 'STET', 'BRD', 'AEAE', 'URG', 'ROC', 'TUSK', 'GNE', 'HNRG', 'RNGR', 'ADSE', 'CENQ', 'SMHI', 'NINE', 'NGL', 'IREN', 'ALPS', 'KLXE', 'CCLP', 'EPSN', 'DFLI', 'AE']#, 'PNRG', 'NGS', 'VOC', 'MTRX']#, 'CBAT', 'SND', 'EOSE', 'EPOW']#, 'GIFI', 'USEG', 'NCSM']#, 'CEI', 'INDO', 'ICD', 'OESX', 'GEOS', 'CNEY', 'PFIE', 'DWSN', 'ESOA', 'BNRG', 'SPI', 'CGRN', 'HUSA', 'ENG', 'MXC', 'RCON', 'CORR', 'ENSV', 'SDPI', 'CREG', 'MIND', 'GBR', 'EFOI', 'AAPL', 'ABBV', 'AMGN', 'GILD', 'REGN', 'VRTX', 'MU', 'MRNA', 'BIIB', 'BNTX', 'MCHP', 'MRVL', 'CTSH', 'HPQ', 'DELL', 'ALNY', 'GMAB', 'VEEV', 'SGEN', 'CAJ', 'ARGX', 'MBLY', 'HPE', 'BGNE', 'INCY', 'BMRN', 'SPOT', 'NTAP', 'ALGN', 'NBIX', 'WDC', 'ASX', 'STX', 'UTHR', 'SRPT', 'PSTG', 'LEGN', 'LOGI', 'KRTX', 'HALO', 'APLS', 'DXC', 'ASND', 'IONS', 'EXAS', 'INSP', 'GRFS', 'EXEL', 'DNA', 'AMKR', 'TDOC', 'CERE', 'NTRA', 'EDU', 'MRVI', 'MTSI', 'IGT', 'CRSP', 'CYTK', 'NTLA', 'DNLI', 'ALKS', 'SMCI', 'ARWR', 'MRTX', 'ROIV', 'ABCM', 'HRMY', 'OMCL', 'ABCL', 'BEAM', 'GLPG', 'VSH', 'TWKS', 'VIR', 'PRTA', 'RARE', 'NCR', 'FOLD', 'BPMC', 'ISEE', 'EVH', 'PTCT', 'RLAY', 'PCVX', 'MYOV', 'ACAD', 'IMCR', 'ZLAB', 'AUR', 'EQRX', 'INSM', 'PROK', 'XRX', 'XENE', 'FLYW', 'KD', 'SAGE', 'BCRX', 'IBRX', 'VERV', 'RLX', 'GDRX', 'RXDX', 'AMLX', 'FATE', 'CRDO', 'TASK', 'AKRO', 'CD', 'KRYS', 'NABL', 'TWST', 'VTYX', 'SIMO', 'CRS', 'RXRX', 'RVMD', 'RCUS', 'SDGR', 'DICE', 'NVAX', 'IRWD', 'XNCR', 'CLDX', 'KYMR', 'FGEN', 'NUVL', 'MDRX', 'AGIO', 'BBIO', 'ALLO', 'SWTX', 'IMVT', 'DAWN', 'IOVA', 'VCYT', 'CINC', 'DVAX', 'OPK', 'ZNTL', 'RYTM', 'LYEL', 'ALKT', 'RCKT', 'CPRX', 'TVTX', 'LGND', 'PHR', 'MYGN', 'KDNY', 'NXGN', 'CVAC', 'SNDX', 'KROS', 'INBX', 'CRSR', 'VCEL', 'MDGL', 'IMGN', 'AVID', 'PLRX', 'DDD', 'BLU', 'AUPH', 'MORF', 'DCPH', 'MCRB', 'RGNX', 'SANA', 'ARQT', 'RXT', 'BHVN', 'CDNA', 'KURA', 'NAAS', 'EBS', 'GOSS', 'ACLX', 'CRNX', 'CDMO', 'ERAS', 'AVXL', 'VALN', 'SSYS', 'MRUS', 'REPL', 'PNT', 'ENTA', 'THRD', 'TGTX', 'EXAI', 'MNKD', 'GRRR', 'KNSA', 'IMTX', 'QURE', 'EDIT', 'MIRM', 'DSGN', 'COGT', 'IDYA', 'ANAB', 'NYAX', 'SLP', 'ATHX', 'GERN', 'IGMS', 'VRDN', 'PAR', 'ACCD', 'PRTC', 'ALEC', 'BCYC', 'JANX', 'PHAR', 'MRSN', 'HSTM', 'RNA', 'BLTE', 'HLVX', 'TRDA', 'SRNE', 'AGEN', 'ITOS', 'TNGX', 'SGMO', 'CHRS', 'CSTL', 'MOR', 'NVTA', 'RAPT', 'ALVR', 'NOTE', 'RPTX', 'ICPT', 'NNDM', 'IPSC', 'NRIX', 'MIT', 'NKTX', 'ALT', 'CGEM', 'VNDA', 'EWTX', 'TSVT', 'CTIC', 'IMGO', 'CRBU', 'PMVP', 'GTPB', 'ATAI', 'ESPR', 'ADMA', 'VECT', 'STOK', 'INO', 'ALXO', 'AMRN', 'GHRS', 'CAN', 'FINM', 'KZR', 'CTAQ', 'AVEO', 'EPHY', 'NUVB', 'ALLK', 'CCCC', 'VERA', 'HCAT', 'BLUE', 'ARCT', 'AKUS', 'SRRK', 'TARS', 'AVTE', 'CPSI', 'GTHX', 'HRTX', 'CYXT', 'ZYME', 'CRGE', 'ATRA', 'MESO', 'IVVD', 'ALBO', 'ETNB', 'ORGO', 'ANIK', 'KODK', 'PTOC', 'VIGL', 'EGRX', 'SLN', 'GLUE', 'LXRX', 'SUAC', 'PTGX', 'QSI', 'STRO', 'PRTH', 'CNTA', 'ENTF', 'CALT', 'KNTE', 'KPTI', 'HUMA', 'KOD', 'IMRX', 'OCGN', 'AURA', 'ABUS', 'BMEA', 'FHTX', 'APTM', 'PSTX', 'PGEN', 'ATEK', 'CMPX', 'MDXG', 'SKYT', 'BTAI', 'TIL', 'GTPA', 'VKTX', 'ATNM', 'GBIO', 'MAPS', 'CELU', 'MGTX', 'ALDX', 'IMAB', 'MGNX', 'PRLD', 'FLYA', 'RANI', 'AADI', 'STRC', 'PRE', 'PEPG', 'MLTX', 'DRTS', 'GRNA', 'DMYS', 'DBVT', 'HRZN', 'NAUT', 'TYRA', 'RLYB', 'FDMT', 'ABSI', 'DSP', 'CNCE', 'TERN', 'ANTX', 'ABOS', 'OPRX', 'BCAB', 'AFMD', 'ALPN', 'URGN', 'AUTL', 'OPT', 'QUOT', 'OTLK', 'GVCI', 'ATAK', 'GTAC', 'SELB', 'MLAI', 'TCRT', 'ACIU', 'OMGA', 'IVA', 'ARDX', 'OCFT', 'CTLP', 'EVLO', 'ADAP', 'LCTX', 'FENC', 'GRTS', 'EIGR', 'PMTS', 'ANNX', 'GRCL', 'GCT', 'SMMT', 'OYST', 'THRX', 'MOLN', 'KMDA', 'VXRT', 'GNFT', 'KULR', 'GRPH', 'VYGR', 'BKSY', 'XOMA', 'DBD', 'IMMR', 'VBIV', 'TIG', 'THTX', 'VOR', 'RVLP', 'IPHA', 'ANIX', 'MTNB', 'LIAN', 'KRON', 'CMRX', 'IMMP', 'ATXS', 'VAXX', 'BBAI', 'EPIX', 'PDSB', 'DTIL', 'RAIN', 'CVM', 'VIRX', 'QMCO', 'OLMA', 'CELC', 'LYRA', 'YMAB', 'CLVS', 'PHVS', 'TCMD', 'ICVX', 'INMB', 'LRMR', 'EMBK', 'GMDA', 'NBTX', 'SVRA', 'CAPR', 'TETE', 'IMRA', 'RIGL', 'XFOR', 'BCLI', 'FSTX', 'KALV', 'RCEL', 'IPA', 'BIVI', 'ATHA', 'CLLS', 'MREO', 'TNYA', 'GLSI', 'OVID', 'STTK', 'PRTG', 'JNCE', 'LVTX', 'SESN', 'INFI', 'IFRX', 'IVAC', 'XBIT', 'ENOB', 'ALZN', 'SLS', 'ORIC', 'MNMD', 'SNTI', 'FUSN', 'ANVS', 'ARMP', 'NVCT', 'CBAT', 'VACC', 'TRHC', 'MYMD', 'CUE', 'MNOV', 'PBYI', 'WTMA', 'STRM', 'JZ', 'DMTK', 'IMPL', 'FBIO', 'IKNA', 'RPHM', 'PRAX', 'DBTX', 'VSTM', 'SPPI', 'HOWL', 'SYRS', 'TSHA', 'ELYM', 'CKPT', 'ALOT', 'ADVM', 'FIXX', 'CTMX', 'SQZ', 'BDTX', 'LPTX', 'OCX', 'GNTA', 'BCTX', 'GALT', 'PIRS', 'AXLA', 'INKT', 'RVPH', 'ASMB', 'DMYY', 'HCWB', 'KINZ', 'CYBN', 'LBPH', 'MIRO', 'SURF', 'CRIS', 'SRZN', 'LUMO', 'ONCY', 'IOBT', 'ACHL', 'ATA', 'LGVN', 'GMTX', 'ANEB', 'MGTA', 'VTVT', 'LIFE', 'TLSA', 'HOOK', 'BIOR', 'NXTC', 'CRDF', 'SPRO', 'AGLE', 'PRQR', 'LKCO', 'CLGN', 'PASG', 'OSS', 'TCRX', 'RNLX', 'GNPX', 'INZY', 'CLNN', 'DYAI', 'RZLT', 'XLO', 'CYT', 'ICCC', 'FREQ', 'AVTX', 'NRXP', 'FNCH', 'EQ', 'TALS', 'PMCB', 'BNOX', 'PYXS', 'EVAX', 'CABA', 'GTBP', 'IMUX', 'TCRR', 'MTBC', 'SYBX', 'MBIO', 'ONCT', 'BCEL', 'VIOT', 'ORTX', 'NCNA', 'SGLY', 'CADL', 'GLTO', 'STG', 'IMNM', 'RCOR', 'BOLT', 'MDNA', 'ARAV', 'BLRX', 'PLX', 'MACK', 'OKYO', 'AQB', 'APGN', 'AKBA', 'ICAD', 'INAB', 'LTRN', 'PTN', 'OCUP', 'SLDB', 'MEIP', 'SNPX', 'CNTB', 'APTO', 'FRLN', 'GSIT', 'NH', 'ISR', 'ALGS', 'PMN', 'SNSE', 'SABS', 'TACT', 'CDTX', 'ADAG', 'APLT', 'NHWK', 'MAIA', 'CRVS', 'GANX', 'CGTX', 'GRTX', 'UBX', 'CTM', 'GLYC', 'PPBT', 'BOXL', 'PCSA', 'BTTX', 'VEDU', 'MTEM', 'APRE', 'QNCX', 'ORGS', 'AMAM', 'TARA', 'AMTI', 'YQ', 'BCDA', 'ACXP', 'MRKR', 'TCON', 'MNPR', 'ELDN', 'LSTA', 'AIKI', 'RCON', 'NYMX', 'TGL', 'JT', 'NTRB', 'WNW', 'RUBY', 'KTTA', 'DMAC', 'ATNX', 'ABIO', 'AVRO', 'TRVN', 'EVGN', 'IDRA', 'BYSI', 'HARP', 'CMRA', 'ANGN', 'SNGX', 'ABEO', 'MBRX', 'GLS', 'AIM', 'JSPR', 'AGTC', 'CDAK', 'VTGN', 'CASI', 'IMV', 'VCNX', 'ERYP', 'ASLN', 'SPRB', 'ELEV', 'TNXP', 'EFTR', 'CNTG', 'AGE', 'OGEN', 'CYCN', 'CANF', 'APM', 'CYAD', 'NLTX', 'ABVC', 'BPTH', 'PLUR', 'VERB', 'RGLS', 'CMMB', 'PXMD', 'VINC', 'ANTE', 'GOVX', 'FWP', 'SIOX', 'ENTX', 'FBRX', 'ACHV', 'GRAY', 'REUN', 'NNVC', 'BTCM', 'AIMD', 'APTX', 'IKT', 'AEZS', 'NLSP', 'ACOR', 'DGHI', 'TPST', 'ONCR', 'NRSN', 'ALRN', 'IBIO', 'VRAX', 'YMTX', 'ATHE', 'ATNF', 'CBIO', 'WAVD', 'EDTK', 'ARDS', 'HGEN', 'SONM', 'SLNH', 'BWV', 'PYPD', 'NEXI', 'MTCR', 'CRBP', 'LMNL', 'INDP', 'SGTX', 'RNXT', 'EDSA', 'CYCC', 'PSTV', 'ONCS', 'VYNE', 'APVO', 'OBSV', 'DRUG', 'GWAV', 'NERV', 'IMMX', 'ONVO', 'YTEN', 'BRQS', 'OST', 'BIOC', 'DTST', 'SCKT', 'LEXX', 'TTOO', 'LABP', 'VRPX', 'CALA', 'CLRB', 'CING', 'TOVX', 'SONN', 'SILO', 'BRTX', 'IMRN', 'ASNS', 'VYNT', 'AYLA', 'PHGE', 'KZIA', 'XRTX', 'NXL', 'UNCY', 'IMNN', 'STSS', 'IMTE', 'DFFN', 'ADTX', 'ELYS', 'GLMD', 'BPTS', 'OPGN', 'PTIX', 'BLCM', 'ERNA', 'VBLT', 'QNRX', 'NBSE', 'QLGN', 'SVVC', 'KTRA', 'TMBR', 'RNAZ', 'MIND', 'MTP', 'ANPC', 'SLNO', 'LIXT', 'PHIO', 'NAVB', 'WORX', 'XBIO', 'XCUR', 'CFRX', 'SASI', 'STAB', 'TCBP', 'XTLB', 'CYTO', 'COEP', 'ENVB', 'DRMA', 'AIHS', 'HILS', 'CLXT', 'ADXN', 'PALI', 'CWBR', 'BHAT', 'ENSC', 'NRBO', 'REVB', 'CELZ', 'HTGM', 'KPRX', 'FRTX', 'ALLR', 'VIRI', 'SCPS', 'WINT', 'RBCN', 'KRBP', 'BNTC', 'SLRX', 'PTE', 'BXRX', 'TENX', 'SPRC', 'PT', 'PBLA', 'HSTO', 'INM', 'VLON', 'MOB', 'FWBI']
  #s&p500 stocks 
  #stock_tickers = ['WMB', 'KHC', 'O', 'DOW', 'WELL', 'MOS', 'GPC', 'VZ', 'MTB', 'BA', 'A', 'MCO', 'RHI', 'ODFL', 'LLY', 'WAT', 'CBRE', 'CI', 'DUK', 'DHR', 'BDX', 'PAYX', 'HPE', 'ABMD', 'CAT', 'HON', 'KEYS', 'PEG', 'MKTX', 'WBA', 'AIZ', 'HD', 'PHM', 'AME', 'BBWI', 'NKE', 'J', 'AAP', 'MMC', 'LIN', 'XOM', 'PXD', 'EMN', 'DHI', 'STE', 'ALB', 'CMI', 'F', 'TSLA', 'HIG', 'EFX', 'DGX', 'PPG', 'HBAN', 'TSCO', 'MDT', 'FE', 'HRL', 'MMM', 'APTV', 'MGM', 'GD', 'ORCL', 'HSIC', 'ADM', 'ALK', 'TYL', 'ETN', 'PGR', 'ECL', 'ADBE', 'WEC', 'NFLX', 'QCOM', 'GRMN', 'MAS', 'RCL', 'URI', 'TXT', 'FBHS', 'LVS', 'AAL', 'ROP', 'FDX', 'JKHY', 'ROK', 'WFC', 'WHR', 'SNPS', 'DE', 'NWL', 'NSC', 'IVZ', 'ANET', 'BIIB', 'LEN', 'GE', 'CDAY', 'NEM', 'NVDA', 'MOH', 'INVH', 'COF', 'MRNA', 'PSX', 'GL', 'HES', 'DRI', 'TSN', 'EQR', 'AEP', 'PEP', 'MSCI', 'MTCH', 'CBOE', 'AEE', 'EBAY', 'JPM', 'KDP', 'EXPD', 'FFIV', 'AXP', 'HAS', 'DIS', 'BRO', 'CRM', 'CMG', 'LKQ', 'SWK', 'MET', 'VICI', 'ES', 'PM', 'VRSK', 'BIO', 'ETR', 'AON', 'GOOG', 'ADP', 'CINF', 'SBUX', 'XRAY', 'CTAS', 'SPGI', 'ON', 'D', 'FAST', 'XYL', 'SEE', 'INTU', 'OXY', 'C', 'KEY', 'SLB', 'FCX', 'VNO', 'EXPE', 'COST', 'RTX', 'BK', 'OGN', 'EW', 'OKE', 'MNST', 'AOS', 'POOL', 'MKC', 'LRCX', 'DD', 'CZR', 'BAC', 'FOXA', 'LHX', 'SCHW', 'FIS', 'AVY', 'TROW', 'AVB', 'HOLX', 'CF', 'NDAQ', 'ROL', 'HUM', 'MA', 'EOG', 'VTR', 'ALLE', 'PKG', 'DXC', 'LW', 'LYV', 'ETSY', 'UNP', 'YUM', 'AMAT', 'NTAP', 'LUV', 'CMA', 'GWW', 'ANSS', 'HPQ', 'SO', 'BWA', 'PYPL', 'BMY', 'AKAM', 'APD', 'NWS', 'CHD', 'CRL', 'GIS', 'IP', 'MS', 'REG', 'UDR', 'TJX', 'BLK', 'ZION', 'RE', 'FMC', 'LUMN', 'WAB', 'SYF', 'ALGN', 'FTNT', 'DFS', 'CPRT', 'OTIS', 'DTE', 'USB', 'PNR', 'AFL', 'CCI', 'PFE', 'TDY', 'CNC', 'GLW', 'JNPR', 'ADSK', 'CL', 'ABBV', 'ESS', 'PEAK', 'ISRG', 'IPG', 'RF', 'TGT', 'NRG', 'WMT', 'WY', 'EQT', 'GILD', 'TEL', 'GM', 'PCAR', 'SPG', 'MDLZ', 'ZTS', 'CHTR', 'CB', 'NOC', 'KIM', 'RL', 'WYNN', 'HST', 'CPT', 'STT', 'DVA', 'CSX', 'NUE', 'EL', 'MO', 'EXC', 'ABC', 'AIG', 'VTRS', 'TRMB', 'WST', 'L', 'STZ', 'NDSN', 'PLD', 'ROST', 'TECH', 'CVS', 'JCI', 'ACN', 'GS', 'CHRW', 'SBNY', 'MCD', 'DLR', 'SRE', 'TFX', 'FTV', 'CARR', 'IRM', 'HII', 'TDG', 'WRK', 'AMT', 'ENPH', 'MAA', 'CVX', 'ITW', 'NI', 'EIX', 'LNC', 'TFC', 'REGN', 'WRB', 'PFG', 'META', 'T', 'NWSA', 'AMCR', 'IQV', 'GNRC', 'MRK', 'TXN', 'ATVI', 'MPWR', 'CTVA', 'NOW', 'LH', 'MTD', 'HAL', 'WDC', 'FRT', 'DAL', 'TMO', 'VRSN', 'VRTX', 'CLX', 'KMI', 'SYK', 'ICE', 'DXCM', 'BAX', 'CMS', 'EQIX', 'IEX', 'IBM', 'PNW', 'CFG', 'HWM', 'CCL', 'IDXX', 'EMR', 'CTLT', 'TPR', 'TMUS', 'SHW', 'VFC', 'SEDG', 'BR', 'FITB', 'UAL', 'AMGN', 'CEG', 'BSX', 'LDOS', 'COO', 'KO', 'PH', 'TRV', 'SBAC', 'TAP', 'INCY', 'EXR', 'CNP', 'ILMN', 'ARE', 'AVGO', 'MAR', 'COP', 'AMP', 'KLAC', 'ZBH', 'NEE', 'SYY', 'WM', 'SWKS', 'PTC', 'AES', 'CTSH', 'MLM', 'KMX', 'CDNS', 'AMD', 'TTWO', 'TER', 'NCLH', 'PPL', 'VLO', 'APH', 'MSI', 'GOOGL', 'JBHT', 'PAYC', 'EPAM', 'MSFT', 'XEL', 'PARA', 'GPN', 'V', 'IR', 'OMC', 'LOW', 'CDW', 'STX', 'ATO', 'JNJ', 'RMD', 'SJM', 'UHS', 'LNT', 'APA', 'NTRS', 'BXP', 'BBY', 'FRC', 'PG', 'CMCSA', 'BEN', 'CPB', 'DOV', 'ADI', 'QRVO', 'K', 'MU', 'AJG', 'AZO', 'UPS', 'MHK', 'CME', 'TT', 'FANG', 'ALL', 'ED', 'HSY', 'MCK', 'MRO', 'KMB', 'VMC', 'LMT', 'ZBRA', 'MCHP', 'AAPL', 'CAH', 'MPC', 'CSCO', 'DVN', 'IFF', 'ORLY', 'FOX', 'IT', 'PSA', 'SIVB', 'DPZ', 'INTC', 'HLT', 'SNA', 'DLTR', 'CSGP', 'KR', 'EA', 'CTRA', 'BKNG', 'WTW', 'EVRG', 'NXPI', 'PNC', 'LYB', 'ULTA', 'FDS', 'RSG', 'PWR', 'DISH', 'FLT', 'NVR', 'CAG', 'CE', 'RJF', 'ABT', 'BKR', 'PKI', 'PRU', 'AWK', 'HCA', 'PCG', 'AMZN', 'DG', 'FISV', 'UNH']
  
  #start_year = 2018
  anvil.server.task_state['status'] = 'Getting stock data...'
  #end_Date,adj_Close_Price,price_der,stock_Tokens = get_stock_data(stock_tickers,datetime.datetime(start_year,1,1),relativedelta(years=1))
  start_Date = datetime.datetime.today()-relativedelta(years=1,days=1)
  end_Date,adj_Close_Price,price_der,stock_Tokens = get_stock_data(stock_tickers,start_Date,relativedelta(years=1))
  #Do we want to define the start time and years projected as pickable variables or fix it
  anvil.server.task_state['status'] = 'Running stock selection algorithm...'

  #AP
  if(method == 'Affinity Propagation Clustering (AP)'):
    price_der = anvil.server.call('ap_bump_function', price_der)
    AP_clustering = AffinityPropagation(max_iter=5000,random_state=2).fit(np.array(price_der).T)
    token_indices = AP_clustering.cluster_centers_indices_ 
    
    AP_chosen_stocks = []
    for i in token_indices:
      AP_chosen_stocks.append(stock_Tokens[i])
    anvil.server.task_state['status'] = 'Calculating optimal portfolio...'
    return find_optimal_portfolio(AP_chosen_stocks,start_year)

  #AC
  elif(method == 'Agglomerative Clustering (AC)'):
    price_der = anvil.server.call('ac_bump_function', price_der)
    data=np.array(price_der)
    StepDists = np.zeros([data.shape[1],2])
    IterCount=0
    densities = np.ones(data.shape[1])
    Dendrogram = anvil.server.call('AgCl', data, StepDists, IterCount, densities)
    Dendrogram = np.array(Dendrogram)
    Dendrogram_diff = np.pad(np.diff(Dendrogram[:,0]),(0,1),'constant',constant_values=0)
    Dendrogram_diff = np.array(Dendrogram_diff)
    one_link_too_many = int(Dendrogram[Dendrogram_diff[:]==np.max(Dendrogram_diff[:]),1])
    
    while(one_link_too_many>Dendrogram[0,1]/2):
      Dendrogram_diff[Dendrogram_diff == np.max(Dendrogram_diff[:])]=0
      one_link_too_many = int(Dendrogram[Dendrogram_diff==np.max(Dendrogram_diff),1])
    
    id_oltm = anvil.server.call('idx1d_callable', Dendrogram[:,1], one_link_too_many)
    ideal_number_of_clusters = int(Dendrogram[id_oltm-1,1]) 

    clustering = AgglomerativeClustering(n_clusters = ideal_number_of_clusters).fit(np.transpose(data))
    cutoff_percentage = 0.05
    chosen_clusters = anvil.server.call('select_clusters', clustering.labels_, cutoff_percentage) #VERY IMPORTANT HYPERPARAMETER
    cluster_centers = anvil.server.call('get_centers', data, chosen_clusters, clustering.labels_)
    num_winners = 3
    most_rep = anvil.server.call('closest_stocks_to_centers', data, cluster_centers, clustering.labels_, chosen_clusters, num_winners)
    token_indices = np.array(most_rep).flatten()

    AC_chosen_stocks = []
    for i in token_indices:
      AC_chosen_stocks.append(stock_Tokens[i])
    anvil.server.task_state['status'] = 'Calculating optimal portfolio...'
    return find_optimal_portfolio(AC_chosen_stocks,start_year)

    #PCA
  elif(method == 'Principal Component Analysis (PCA)'):
    PCA_explained_var = .9
    #price_der = anvil.server.call('PCA_bump_function', price_der)

    PCA_stock_tokens = anvil.server.call('run_PCA', np.array(price_der), PCA_explained_var, stock_Tokens)
    anvil.server.task_state['status'] = 'Calculating optimal portfolio...'
 
    return find_optimal_portfolio(PCA_stock_tokens,start_year)

    #Autoencoder
  else:
    #Figure out where to put install commands
    #Figure out how to fux device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 15 #can make a variable or fix
    VAE_chosen_stocks = anvil.server.call('run_VAE', np.array(price_der), epochs, device, stock_Tokens)
    return find_optimal_portfolio(VAE_chosen_stocks,start_year)

@anvil.server.callable
def launch_optimization_task(method):
  """Fire off the optimization task, returning the Task object to the client."""
  task = anvil.server.launch_background_task('optimize_portfolio', method,start_year)
  return task
  
@anvil.server.callable
def get_weights():
  return app_tables.optimal_weights.search(tables.order_by("raw_weight", ascending=False))

@anvil.server.callable
def plot_efficient_frontier():
  plt = app_tables.matlab_plots.get()
  ef_plot = plt['Plots']
  return ef_plot

@anvil.server.callable
def plot_pie_chart():
  data = app_tables.optimal_weights.search()
  dicts = [{'Ticker': r['Ticker'], 'raw_weight': r['raw_weight']}
           for r in data]
  df = pd.DataFrame.from_dict(dicts)
  fig = px.pie(df, values='raw_weight', names='Ticker', title='Portfolio Composition')
  return fig