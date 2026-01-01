# Dashboard Forecasting Crude Oil Prices
A crude oil price forecasting project using a long short-term memory model based on an attention mechanism, with output in the form of an interactive Streamlit dashboard.

Problem Urgency
Indonesia has long been a net oil-importing country. Data indicates that in 2024, Indonesia's daily crude oil demand is 1.5 million barrels, whereas domestic production is only able to supply 0.5 million barrels, necessitating crude oil imports to meet the demand for crude oil, which is then processed into fuel. Furthermore, the issue of crude oil prices, characterized by volatility, non-linearity, non-stationarity, temporal dependence, and regime changes, increases the complexity of forecasting. Therefore, it is necessary to have a solution for accurately predicting crude oil prices to accommodate the efficiency and effectiveness of crude oil imports as fuel, as well as an advanced model capable of addressing these characteristics. 

Solution
Using a forecasting method with deep learning to address the complex characteristics of crude oil prices. The deep learning model used is Long Short-Term Memory (LSTM) based on the Attention Mechanism.

Data  
Using daily data from January 1986 to August 2025, the target variable is the WTI closing price, and the feature variables are the lagged WTI closing price, USD index, lagged USD index, Geopolitical Risk Index (GPRD), and lagged Geopolitical Risk Index.

Research Results
The study employed an Ablation Study to determine the combination of features that provides the most accurate forecasting performance. Case 4 of the Ablation Study, which used the feature combination of WTI lag1 and GPRD with a test MAPE of 2.61%, indicates that crude oil price forecasting is highly accurate, with an error percentage of only 2.61% from the actual WTI price values. The WTI crude oil price forecast was conducted over a period of one month, showing a positive trend from August 5, 2025, to September 3, 2025, implying that the WTI price is expected to increase over the following month. A dashboard was also created to enable users to interactively forecast WTI crude oil prices at the following link:

Dashboard link
https://dashboard-ta.streamlit.app/
