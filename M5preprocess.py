
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from math import ceil

def reduce_usage_mem(df):
        for col in  df.columns:
            if str(df[col].dtype)=='int64':
                df[col]=df[col].astype('int16')
            if str(df[col].dtype)=='float64':
                df[col]=df[col].astype('float16')
        return df

def merge_by_concat(df1, df2, on_col):
    merged_df = df1[on_col]
    merged_df = merged_df.merge(df2, on=on_col, how='left')
    new_columns = [col for col in list(merged_df) if col not in on_col]
    df1 = pd.concat([df1, merged_df[new_columns]], axis=1)
    return df1


class m5_preprocessing(object):

    def __init__(self,sales_df_path,calendar_df_path,price_df_path,weather_df_path):
        self.sales_df_path=sales_df_path
        self.calendar_df_path=calendar_df_path
        self.price_df_path=price_df_path
        self.weather_df_path=weather_df_path
        return

    def load_data(self):
        if (self.sales_df_path[-4:] != '.csv'):
            logging.error('The sales_df file is not a .csv file')
        
        if (self.calendar_df_path[-4:] != '.csv'):
            logging.error('The calendar_df file is not a .csv file')

        if (self.price_df_path[-4:] !='.csv'):
            logging.error('The price_df file is not a .csv file')

        if (self.weather_df_path[-4:] != '.csv'):
            logging.error('The weather_df file is not a .csv file')

        else:
            logging.debug('Loading data')
            sales_df=reduce_usage_mem(pd.read_csv(self.sales_df_path))
            calendar_df=reduce_usage_mem(pd.read_csv(self.calendar_df_path))
            price_df=reduce_usage_mem(pd.read_csv(self.price_df_path))
            weather_df=reduce_usage_mem(pd.read_csv(self.weather_df_path))
        return sales_df,calendar_df,price_df,weather_df

    def preprocess(self,sales_df,calendar_df,price_df,weather_df):
        
        logging.debug('Preparing store wise grid')
        
        index_cols=['id','item_id','dept_id','cat_id','store_id','state_id']
        TARGET='sales'

        grid_df = pd.melt(sales_df, 
                  id_vars = index_cols, 
                  var_name = 'd', 
                  value_name = TARGET)
        
        del sales_df
        
        for col in index_cols:
            grid_df[col] = grid_df[col].astype('category')
        
        grid_df['d'] = grid_df['d'].apply(lambda x: x[2:]).astype('int16')
        calendar_df['d'] = calendar_df['d'].apply(lambda x: x[2:]).astype('int16')

        release_df = price_df.groupby(['store_id','item_id'])['wm_yr_wk'].agg(['min']).reset_index()
        release_df.columns = ['store_id','item_id','release']

        grid_df = merge_by_concat(grid_df, release_df, ['store_id','item_id'])
        del release_df
        grid_df = merge_by_concat(grid_df, calendar_df[['wm_yr_wk','d']], ['d'])
        grid_df = grid_df[grid_df['wm_yr_wk']>=grid_df['release']]
        grid_df = grid_df.reset_index(drop=True)
        grid_df['release'] = grid_df['release'] - grid_df['release'].min()
        grid_df['release'] = grid_df['release'].astype('int16')

        grid_df = grid_df.assign(**{
            'lag_{}'.format(lag): grid_df.groupby(['id'])['sales'].transform(lambda x: x.shift(lag)).astype('float16')
            for lag in range(28,43)
            })


        for i in [7,14,30,60]:
            print('Rolling period:', i)
            grid_df['rolling_mean_'+str(i)] = grid_df.groupby(['id'])['sales'].transform(lambda x: x.shift(28).rolling(i).mean()).astype('float16')
            grid_df['rolling_std_'+str(i)]  = grid_df.groupby(['id'])['sales'].transform(lambda x: x.shift(28).rolling(i).std()).astype('float16')
        
        
        price_df['price_max'] = price_df.groupby(['store_id','item_id'])['sell_price'].transform('max')
        price_df['price_min'] = price_df.groupby(['store_id','item_id'])['sell_price'].transform('min')
        price_df['price_std'] = price_df.groupby(['store_id','item_id'])['sell_price'].transform('std')
        price_df['price_mean'] = price_df.groupby(['store_id','item_id'])['sell_price'].transform('mean')

        price_df['price_norm'] = price_df['sell_price']/price_df['price_max']
        price_df['price_nunique'] = price_df.groupby(['store_id','item_id'])['sell_price'].transform('nunique') 
        price_df['item_nunique'] = price_df.groupby(['store_id','sell_price'])['item_id'].transform('nunique')

        calendar_price = calendar_df[['wm_yr_wk','month','year']]
        calendar_price = calendar_price.drop_duplicates(subset=['wm_yr_wk'])

        price_df = price_df.merge(calendar_price[['wm_yr_wk','month','year']], on=['wm_yr_wk'] )
        del calendar_price

        price_df['price_momentum'] = price_df['sell_price']/price_df.groupby(['store_id','item_id'])['sell_price'].transform(lambda x: x.shift(1))
        price_df['price_momentum_m'] = price_df['sell_price']/price_df.groupby(['store_id','item_id','month'])['sell_price'].transform('mean')
        price_df['price_momentum_y'] = price_df['sell_price']/price_df.groupby(['store_id','item_id','year'])['sell_price'].transform('mean')

        del price_df['month'], price_df['year']

        grid_df = reduce_usage_mem(grid_df)
        price_df = reduce_usage_mem(price_df)

        grid_df = grid_df.merge(price_df, on=['store_id','item_id','wm_yr_wk'])
        grid_df = reduce_usage_mem(grid_df)

        grid_df['item_id']=grid_df['item_id'].astype('category')
        grid_df['store_id']=grid_df['store_id'].astype('category')


        calendar_df['date'] = pd.to_datetime(calendar_df['date'])
        calendar_df['event_name_1']=LabelEncoder().fit_transform(calendar_df['event_name_1']).astype('int16')
        calendar_df['event_type_1']=LabelEncoder().fit_transform(calendar_df['event_type_1']).astype('int16')
        calendar_df['event_name_2']=LabelEncoder().fit_transform(calendar_df['event_name_2']).astype('int16')
        calendar_df['event_type_2']=LabelEncoder().fit_transform(calendar_df['event_type_2']).astype('int16')

        calendar_df['tm_d'] = calendar_df['date'].dt.day.astype(np.int16)
        calendar_df['tm_w'] = calendar_df['date'].dt.week.astype(np.int16)
        calendar_df['tm_m'] = calendar_df['date'].dt.month.astype(np.int16)
        calendar_df['tm_y'] = calendar_df['date'].dt.year
        calendar_df['tm_y'] = (calendar_df['tm_y'] - calendar_df['tm_y'].min()).astype(np.int16)
        calendar_df['tm_wm'] = calendar_df['tm_d'].apply(lambda x: ceil(x/7)).astype(np.int16) 

        calendar_df['tm_dw'] = calendar_df['date'].dt.dayofweek.astype(np.int16) 
        calendar_df['tm_w_end'] = (calendar_df['tm_dw']>=5).astype(np.int16)

        del calendar_df['date'] 
        del calendar_df['weekday'] 
        del calendar_df['wm_yr_wk']
        del grid_df['wm_yr_wk']
        grid_df = grid_df.merge(calendar_df, on=['d'])

        icols =  [
            ['cat_id'],
            ['dept_id'],
            ['item_id'],
            ]

        for col in icols:
            print('Encoding', col)
            col_name = '_'+'_'.join(col)+'_'
            grid_df['enc'+col_name+'mean'] = grid_df.groupby(col)['sales'].transform('mean').astype(np.float16)
            grid_df['enc'+col_name+'std'] = grid_df.groupby(col)['sales'].transform('std').astype(np.float16)


        grid_df['item_id']=LabelEncoder().fit_transform(grid_df['item_id']).astype('int16')
        grid_df['dept_id']=LabelEncoder().fit_transform(grid_df['dept_id']).astype('int16')
        grid_df['cat_id']=LabelEncoder().fit_transform(grid_df['cat_id']).astype('int16')
        
        weather_df['d'] = weather_df['d'].apply(lambda x: x[2:]).astype('int16')

        grid_df=grid_df.merge(weather_df[['AWND','PRCP','TAVG','state_id','d']],on=['state_id','d'])
        
        for store in grid_df['store_id'].unique():
            logging.debug('exporting {}_full_grid.pkl'.format(store))
            grid_df[(grid_df['store_id']==store) & (grid_df['d']<=1941)].to_pickle('train_data/{}_full_grid.pkl'.format(store))
            grid_df[(grid_df['store_id']==store) & (grid_df['d']>1941)].to_pickle('test_data/{}_test_grid.pkl'.format(store))
            
            return '{}_full_grid.pkl'.format(store)