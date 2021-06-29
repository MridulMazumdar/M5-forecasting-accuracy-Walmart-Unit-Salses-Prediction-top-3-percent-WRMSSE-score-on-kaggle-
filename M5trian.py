import lightgbm as lgb
from M5preprocess import m5_preprocessing
import pickle
import pandas as pd

sales_df_path='sales_train_evaluation.csv'
calendar_df_path='calendar.csv'
price_df_path='sell_prices.csv'
weather_df_path='usa_weather.csv'

m5p=m5_preprocessing(sales_df_path,
                     calendar_df_path,
                     price_df_path,
                     weather_df_path)

sales_df,calendar_df,price_df,weather_df=m5p.load_data()

m5p.preprocess(sales_df,calendar_df,price_df,weather_df)


def m5train():
    stores=['CA_1','CA_2','CA_3','CA_4','TX_1','TX_2','TX_3','WI_1','WI_2','WI_3']

    features=['item_id', 'dept_id', 'cat_id','d',
        'release', 'sell_price', 'price_max', 'price_min', 'price_std',
       'price_mean', 'price_norm', 'price_nunique', 'item_nunique',
       'price_momentum', 'price_momentum_m', 'price_momentum_y', 'wday',
       'month', 'year', 'event_name_1', 'event_type_1', 'event_name_2',
       'event_type_2', 'snap_CA', 'snap_TX', 'snap_WI', 'tm_d', 'tm_w', 'tm_m',
       'tm_y', 'tm_wm', 'tm_dw', 'tm_w_end', 'enc_cat_id_mean',
       'enc_cat_id_std', 'enc_dept_id_mean', 'enc_dept_id_std',
       'enc_item_id_mean', 'enc_item_id_std', 'lag_28', 'lag_29', 'lag_30',
       'lag_31', 'lag_32', 'lag_33', 'lag_34', 'lag_35', 'lag_36', 'lag_37',
       'lag_38', 'lag_39', 'lag_40', 'lag_41', 'lag_42', 'rolling_mean_7',
       'rolling_std_7', 'rolling_mean_14', 'rolling_std_14', 'rolling_mean_30',
       'rolling_std_30', 'rolling_mean_60', 'rolling_std_60', 'AWND', 'PRCP',
       'TAVG']

    target='sales'

    start=710

    train_upto=1941

    horizon=28

    param={
        'objective': 'tweedie',
        'metric': 'rmse',
        'tweedie_variance_power': 1.1,
        'verbose': -1,
        'boosting_type': 'gbdt',
        'boost_from_average':False,
        'num_leaves': 2530,
        'subsample':0.6,
        'learning_rate':0.03,
        'feature_fraction': 0.6,
        'bagging_fraction': 0.9,
        'bagging_freq': 4,
        'n_estimators': 1400,
        'min_data_in_leaf': 4095,
        'max_bin':100,
        'num_boost_round':1400,
        'feature_pre_filter':False,'seed':23
      }

    for store in  stores:
        print('store: ',store)
    
        full_grid=pd.read_pickle('{}_full_grid.pkl'.format(store))    

        train_mask = (full_grid['d']>=start)& (full_grid['d']<=train_upto)
        valid_mask = (train_mask) & (full_grid['d']>(train_upto - horizon))
  
  
        train_data = lgb.Dataset(full_grid[train_mask][features], 
                       label=full_grid[train_mask][target])

        valid_data = lgb.Dataset(full_grid[valid_mask][features], 
                           label=full_grid[valid_mask][target])

  
        gbm = lgb.train(param,train_data,valid_sets=[valid_data])

        model_name = 'store_wise_models/lgb_model_'+store+'.bin'

        pickle.dump(gbm, open(model_name, 'wb'))

        return model_name
    
m5train()