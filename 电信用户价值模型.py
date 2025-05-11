import pandas as pd
import numpy as np

'''
读取数据
'''
# 读取hdf5
store =  pd.HDFStore('GZunicom.h5')
# 用户明细提取
llbUser = store.get("llb/user/2023-01-03")
# PV明细提取
llbPV = {}
for (path, subgroups, subkeys) in store.walk():
    if 'pv' in path:
        for subkey in subkeys:
            key = "/".join([path, subkey])
            llbPV[subkey] = store.get(key)
# 订单明细提取
llbOrder = {}
for (path, subgroups, subkeys) in store.walk():
    if 'order' in path:
        for subkey in subkeys:
            key = "/".join([path, subkey])
            llbOrder[subkey] = store.get(key)
store.close()

'''
用户模型
'''
# 整体运营用户模型
def level_mark(pvdata):
    # 数据清洗
    pvdata['month'] = pvdata['startTime'].dt.month
    pvdata['month'] = pvdata['month'].replace(12,-1)# 跨年特殊处理
    
    # 整体运营用户表
    uv = pvdata.drop_duplicates(['userId','CREATE_DATE']).groupby(['userId','month']).size().reset_index(name='frequency')
    # 访问月频次、平均日频次
    uv_frequency = uv.groupby('userId').agg({'month':['count','max'],'frequency':'mean'}).reset_index()
    uv_frequency.columns = ['userId','访问月频次','最后访问月份','月内平均访问日频次']
    # 订购月频次、平均日频次
    order_frequency = pvdata[pvdata['isOrder']].groupby(['userId','month']).agg({'isOrder':'sum'}).reset_index()
    order_frequency1 = order_frequency.groupby('userId').agg({'month':['count','max'],'isOrder':'mean'}).reset_index()
    order_frequency1.columns = ['userId','订购月频次','最后订购月份','月内平均订购频次']
    # 产品合集
    good_union = pvdata[pvdata['isOrder']].groupby(['userId','month','GOODS_TYPE']).size().reset_index()
    good_union1 = good_union.groupby(['userId','month']).agg({'GOODS_TYPE':'-'.join,0:'sum'}).reset_index()
    good_union2 = good_union1.sort_values('month').groupby('userId').agg({'GOODS_TYPE':[', '.join,'count'],0:'sum'}).reset_index()
    good_union2.columns = ['userId','订购产品汇总','订购产品月频次','订购总次数']
    # 合并
    uv = uv_frequency.merge(order_frequency1,how='left',on='userId'
                           ).merge(good_union2,how='left',on='userId')
    
    # 用户打标
    user_mark = {
        '沉默用户': uv[uv['最后访问月份']<3],
        '流失用户': uv[(uv['最后访问月份']>=3)&(uv['订购月频次'].isnull())],
        '订购用户': uv[(uv['最后访问月份']>=3)&(uv['订购月频次'].notnull())]
    }
    return user_mark

## RFM模型（整体用户 - 3个月的订购记录）
llbOrders = pd.concat(llbOrder.values())
# R——最近一次购买时间
recency = llbOrders.groupby('userId')['startTime'].max().reset_index().rename(columns={'startTime':'FIRST_ORDERTIME'})
recency['rcency'] = (pd.to_datetime('20230601') - recency['FIRST_ORDERTIME']).dt.days
# F——购买频次
frequency = llbOrders.groupby('userId').size().reset_index(name='frequency')
# M——购买金额
monetary = llbOrders.groupby('userId')['PRODUCT_PRICE'].sum().reset_index(name='monetary')

# RFM模型合成
rfm = pd.merge(recency,frequency, how='outer',on='userId'
              ).merge(monetary, how='outer',on='userId')
# RFM打分
# 2：0-30天，1：31-91天
rfm['r_score'] = pd.cut(rfm['rcency'], bins=[-1,30,91], labels=range(2,0,-1))
# 1：1-2次，2：2次以上
rfm['f_score'] = pd.cut(rfm['frequency'], bins=[0,2,45],labels=range(1,3))
# 1：20元以下，2：20元以上
rfm['m_score'] = pd.cut(rfm['monetary'], bins=[-1,10,20,804], labels=range(1,4))
rfm['rfm_score'] = rfm['r_score'].astype(str)+rfm['f_score'].astype(str)+rfm['m_score'].astype(str)

# RFM释义
rfm_type = {
    '111': ['一般挽留用户','无法立即带来较大利润'],
    '211': ['一般发展用户','低价值用户'],
    '222': ['重要价值用户','防止流失'],
    '122': ['重要保持用户','优质客户'],
    '221': ['一般价值用户','较难获取更多利润'],
    '121': ['一般保持用户','活跃但购买力有限'],
    '112': ['重要挽留用户','购买金额大有潜在价值'],
    '212': ['重要发展用户','精准化提高购买率']
}
rfm = rfm.merge(pd.DataFrame.from_dict(rfm_type,orient='index',columns=['rfm_type','rfm_desc']
                                      ).reset_index().rename(columns={'index':'rfm_score'}),how='left',on='rfm_score')

## 转化漏斗模型
# PV明细数据清洗——转化节点、访问参数
llbPVs = pd.concat(llbPV.values())
# 用户转化节点
def invert_Mark(pvdata):
    '''
    转化节点打标：
    4:进专区, 3:访问详情页, 2:点击订购, 1:订购成功
    '''
    pvdata['invertNode'] = '4'
    # 节点-条件
    invert_key = {
        '1': (pvdata['BUSINESS_ID'].str.startswith('result'))&(pvdata['RESULT_INFO']=='true'),
        '3': pvdata['BUSINESS_ID'].str.startswith('discountDetail'),
        '2': pvdata['BUSINESS_ID'].str.startswith('discountDetail_attend')
    }
    # 遍历打标
    for key,value in invert_key.items():
        pvdata.loc[value, 'invertNode'] = key
    
    return pvdata
# PV去重
def duplicates_invert(pvdata):
    '''
    去重：-空GOODS，+all 订单
    '''
    # 去重1：排invertNode，去重'SERIAL_NUMBER','PRODUCT_ID'；
    dupdata = pvdata.sort_values('invertNode'
                                ).drop_duplicates(['SERIAL_NUMBER','PRODUCT_ID'],keep='first')
    
    # 去重2：剔['GOODS_ID'].isnull(); ['PRODUCT_ID'].isnull()，排invertNode，去重'SERIAL_NUMBER','GOODS_ID'
    df2 = dupdata[dupdata['GOODS_ID'].notnull()]

    df3 = df2[df2['SERIAL_NUMBER'].isin(df2.loc[df2['PRODUCT_ID'].isnull(),'SERIAL_NUMBER'])
             ].sort_values('invertNode').drop_duplicates(['SERIAL_NUMBER','GOODS_ID'],keep='first')
    # 单独记录加到‘去重1’
    dupdata = dupdata[dupdata['PRODUCT_ID'].notnull()
                     ].append(df3[df3['PRODUCT_ID'].isnull()])
    # 无产品信息的浏览记录加到‘去重1’
    dupdata = dupdata.append(dupdata[(dupdata['GOODS_ID'].isnull())
                                     &(~dupdata['SERIAL_NUMBER'
                                               ].isin(dupdata.loc[dupdata.duplicated('SERIAL_NUMBER'),'SERIAL_NUMBER']))])
    
    # 去重3：+全部订购，去重'SERIAL_NUMBER','CREATE_TIME','invertNode'
    dupdata = dupdata.append(pvdata[pvdata['invertNode']=='1']
                            ).drop_duplicates(['SERIAL_NUMBER','CREATE_TIME','invertNode'])
    
    return dupdata

# 合并——用户属性+用户转化+用户价值
rfm_cols = [
    'SERIAL_NUMBER', 'FIRST_ORDERTIME', 'rcency', 'frequency', 'monetary', 'rfm_score','rfm_type']
PC_cols = [
    'SERIAL_NUMBER','CREATE_TIME','GOODS_ID','GOODSNAME','PRODUCT_NAME','CHANNEL_DESC','invertNode']

ZQ = pd.merge(llbUser,pd.merge(llbPVs_Ms[PC_cols],rfm[rfm_cols], how='left',on='SERIAL_NUMBER'
                              ), how='right', left_on='号码',right_on='SERIAL_NUMBER')

## 相关性维度分析
# 维度清洗
# +['是否副卡','是否超套']
ZQ['是否副卡'] = ZQ['副卡数量'].notnull()
ZQ['是否超套'] = (ZQ['1-3月平均超套费用']>0)
# +['是否流失']、['是否重价值']
ZQ['是否流失'] = ZQ['invertNode']!='1'
ZQ['是否重价值'] = ZQ['rfm_type'].str.contains('重要')
# +['生命周期']
ZQ['生命周期'] = ((pd.to_datetime('20230401')-pd.to_datetime(ZQ['入网时间'])).dt.days/30).astype(str).str.split('.',expand=True)[0]+'月'
# +出账等级
ZQ['出账等级'] = pd.cut(ZQ['1-3月平均出账'],bins=[0,51,82,123,1281])
# +使用流量等级
ZQ['使用流量等级'] = pd.cut(ZQ['1-3月平均使用流量'],bins=[0,20,37,60,1991])

# 相关性评估
ZQ_factorize = ZQ.apply(lambda x: pd.factorize(x)[0])
corr = ZQ_factorize.corr()