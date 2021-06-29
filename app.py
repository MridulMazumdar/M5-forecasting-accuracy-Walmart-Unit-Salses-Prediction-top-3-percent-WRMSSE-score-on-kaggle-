from flask import Flask,render_template,request
import pandas as pd
import pickle
import plotly.express as px
import json
import plotly
app=Flask(__name__)

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

@app.route('/')
def webpage():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    stores=['CA_1','CA_2','CA_3','CA_4','TX_1','TX_2','TX_3','WI_1','WI_2','WI_3']

    if request.method=='POST':
        store_id=request.form['store']
        item_id=request.form['item']
    
    if store_id not in stores:
        return ('{} store not found'.format(store_id))
    
    else:
        print('store: ',store_id)
        full_grid=pd.read_pickle('test_data/{}_test_grid.pkl'.format(store_id))
    
    if '{}_{}_evaluation'.format(item_id,store_id) not in list(full_grid['id'].unique()):
        return ('{} item not found in store {}'.format(item_id,store_id))        
    
    else:
        full_grid=full_grid[(full_grid['id']=='{}_{}_evaluation'.format(item_id,store_id))][features]
    
        model_path = 'store_wise_models/lgb_model_'+store_id+'.bin'

        reg = pickle.load(open(model_path, 'rb'))
        
        pred_1=reg.predict(full_grid)
        
        pred=[item_id,store_id]+list(pred_1)

        fig=px.line(y=pred[2:],x=[i for i in range(1,29)])
        fig.update_layout(title='Unit sales prediction of {} in {} store for next 28 days'.format(item_id,store_id),
                      xaxis_title='Days',
                      yaxis_title='Units')
        fig.write_html('templates/graph.html',full_html=True,)

        return render_template('graph.html')

if __name__=='__main__':
    app.run(debug=True)