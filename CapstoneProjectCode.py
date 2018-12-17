# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns
import os
get_ipython().magic('matplotlib inline')


# ## Check how they update the data
'''
the followings are the shell commant to check the items in the depository
pwd

ls

cd ~/Documents/UMLMSBA/Capstone/Data/quote_to_order/
'''

# import all the data
df_0413 = pd.read_csv('opportunity_to_quote_04_13.txt', sep = '|').copy()
df_0417 = pd.read_csv('opportunity_to_quote_0417.txt', sep = '|').copy()
df_0501 = pd.read_csv('opportunity_to_quote_0501.txt', sep = '|').copy()
df_0515 = pd.read_csv('opportunity_to_quote_0515.txt', sep = '|').copy()
df_0308p1 = pd.read_csv('opportunity_to_quote_to_order_0308pt1.txt', sep = '|').copy()
df_0308p2 = pd.read_csv('opportunity_to_quote_to_order_0308pt2.txt', sep = '|').copy()
df_0316 = pd.read_csv('opportunity_to_Quote_0316.txt', sep = '|').copy()
df_0319 = pd.read_csv('opportunity_to_Quote_to_order_0319.txt', sep = '|').copy()
df_0322 = pd.read_csv('opportunity_to_quote_0322.txt', sep = '|').copy()
df_0327 = pd.read_csv('opportunity_to_Quote_03_27.txt', sep = '|').copy()

# -------------------------------------------------------------------------------------

def getus(dataframe):
    # get the subset with ship to country is us
    dataframe = dataframe.loc[dataframe['ship_to_country'] == 'United States']
    return dataframe
# get the us only data
df_0308p1 = getus(df_0308p1)
df_0308p2 = getus(df_0308p2)
df_0316 = getus(df_0316)
df_0319 = getus(df_0319)
df_0322 = getus(df_0322)
df_0327 = getus(df_0327)
df_0413 = getus(df_0413)
df_0417 = getus(df_0417)
df_0501 = getus(df_0501)
df_0515 = getus(df_0515)
# detect the difference among the data structures
df_0308p1.shape, df_0308p2.shape, df_0316.shape, df_0319.shape, df_0322.shape, df_0327.shape
# get the columns of the first part of the data contains majority records.
header = df_0308p1.columns.tolist()
df_0308p2 = df_0308p2[header]
df_0316 = df_0316[header]
df_0319 = df_0319[header]
df_0322 = df_0322[header]
df_0327 = df_0327[header]
# append all the data
df_08_27 = df_0308p1.append(df_0308p2).append(df_0316).append(df_0319).append(df_0322).append(df_0327)

df_08_27.shape
# export the data
df_08_27.to_csv('opp_to_order_08_27_Mar.csv', index = False)
# check the volume of the master quote number
len(df_0413.master_quote_number.value_counts()) + len(df_0417.master_quote_number.value_counts()) + len(df_0501.master_quote_number.value_counts()) + len(df_0515.master_quote_number.value_counts())

# ## Concatenate the data
# read the full data previously ensembled
df = pd.read_csv('/Users/wuzirong/Documents/UMLMSBA/Capstone/Data/quote_to_order/opp_to_order_08_27_Mar.csv', low_memory=True).copy()
df.master_quote_number.value_counts()
# drop the loaddate
df_0413.drop('loaddate', axis = 1, inplace = True)
df_0417.drop('loaddate', axis = 1, inplace = True)
df_0501.drop('loaddate', axis = 1, inplace = True)
df_0515.drop('loaddate', axis = 1, inplace = True)
# reset the anomalies value
df_0515.rptg_dt = np.where(df_0515.rptg_dt == "4712-12-31 00:00:00", "2012-12-31 00:00:00", df_0515.rptg_dt)
df_0515.opp_close_dt = np.where(df_0515.opp_close_dt == "4712-12-31 00:00:00", "2012-12-31 00:00:00", df_0515.opp_close_dt)
# concate the data with the new data
df = df.append(df_0413).append(df_0417).append(df_0501).append(df_0515)
df.info()
df.change_flg.value_counts()
# create tables based on their change name
df_original = df.loc[df.change_flg == 'Original']
df_other_change = df.loc[df.change_flg == 'Other Changes']
df_New_Quote = df.loc[df.change_flg == 'New Quote']
df_Opport_stg_nm_Change = df.loc[df.change_flg == 'Opport stg_nm Change']
df_Prior_Win = df.loc[df.change_flg == 'Prior Win']
df_New_Win = df.loc[df.change_flg == 'New Win']
df_New_SO = df.loc[df.change_flg == 'Prior Win - New SO']
# dedup
df_Prior_Win.drop_duplicates(inplace = True)
df_other_change.drop_duplicates(inplace = True)
df_New_Quote.drop_duplicates(inplace = True)
df_New_SO.drop_duplicates(inplace = True)
df_New_Win.drop_duplicates(inplace = True)
df_Opport_stg_nm_Change.drop_duplicates(inplace = True)
print(df_other_change.shape[0], df_New_Quote.shape[0], df_New_SO.shape[0], df_New_Win.shape[0], df_Opport_stg_nm_Change.shape[0], df_original.shape[0], df_Prior_Win.shape[0])
df_original.sort_values(['quote_header_wid', 'oppty_header_cur_wid', 'master_quote_number', 'material'], inplace = True)
# change the datatype of dates into datetime
dt_list = ['quote_created_date', 'rptg_dt', 'quote_created_date', 'opp_close_dt', 'opportunity_update_date',
          'quote_update_date', 'initial_quote_dt']
df_list = [df_original, df_Prior_Win, df_other_change, df_New_Quote, df_New_SO, df_New_Win, df_Opport_stg_nm_Change]
for i in dt_list:
    for j in df_list:
        j[i] = pd.to_datetime(j[i])


# # Figure with prior win and original tables

# Prior Win – these are changes to quotes or opportunities that were flagged as wins in the prior weeks’ dataset – change may be just closing the opportunity or updating the stage name or quote information.
# 
# What to do:
# 
# - inner join and left join tables 
# - update the values from prior win table to original table(inner join table)
# - append the inner join table to left join table(dropped the inner joined part previously)
# 
# ## Update all values from y to x

# assign value from y to x
def assignvalu(dfr):
    dfr['oppty_header_cur_wid_x'] = dfr['oppty_header_cur_wid_y']
    dfr['configured_quantity_x'] = dfr['configured_quantity_y']
    dfr['quote_quantity_x'] = dfr['quote_quantity_y']
    dfr['config_model_x'] = dfr['config_model_y']
    dfr['fmly_x'] = dfr['fmly_y']
    dfr['prod_ln_x'] = dfr['prod_ln_y']
    dfr['quote_row_wid_x'] = dfr['quote_row_wid_y']
    dfr['quote_opportunity_link_flag_x'] = dfr['quote_opportunity_link_flag_y']
    dfr['quote_order_type_x'] = dfr['quote_order_type_y']
    dfr['bookbill_order_type_x'] = dfr['bookbill_order_type_y']
    dfr['quote_reason_code_x'] = dfr['quote_reason_code_y']
    dfr['sales_organization_x'] = dfr['sales_organization_y']
    dfr['opportunity_number_x'] = dfr['opportunity_number_y']
    dfr['quote_order_link_flag_x'] = dfr['quote_order_link_flag_y']
    dfr['quote_created_date_x'] = dfr['quote_created_date_y']
    dfr['deal_registration_number_x'] = dfr['deal_registration_number_y']
    dfr['gro_rejection_reason_x'] = dfr['gro_rejection_reason_y']
    dfr['gro_rejection_comment_x'] = dfr['gro_rejection_comment_y']
    dfr['opp_row_wid_x'] = dfr['opp_row_wid_y']
    dfr['opp_num_x'] = dfr['opp_num_y']
    dfr['quote_vrsn_x'] = dfr['quote_vrsn_y']
    dfr['so_num_x'] = dfr['so_num_y']
    dfr['win_problty_x'] = dfr['win_problty_y']
    dfr['rptg_dt_x'] = dfr['rptg_dt_y']
    dfr['opp_close_dt_x'] = dfr['opp_close_dt_y']
    dfr['opportunity_update_date_x'] = dfr['opportunity_update_date_y']
    dfr['quote_update_date_x'] = dfr['quote_update_date_y']
    dfr['initial_quote_dt_x'] = dfr['initial_quote_dt_y']
    dfr['ship_to_country_x'] = dfr['ship_to_country_y']
    dfr['cust_emc_sub_vert_x'] = dfr['cust_emc_sub_vert_y']
    dfr['cust_idc_maj_vert_x'] = dfr['cust_idc_maj_vert_y']
    dfr['cust_idc_maj_vert_grp_x'] = dfr['cust_idc_maj_vert_grp_y']
    dfr['dnb_amo_seg_x'] = dfr['dnb_amo_seg_y']
    dfr['emc_amo_seg_x'] = dfr['emc_amo_seg_y']
    dfr['empl_tot_seg_x'] = dfr['empl_tot_seg_y']
    dfr['enty_rgn_cd_x'] = dfr['enty_rgn_cd_y']
    dfr['stg_nm_x'] = dfr['stg_nm_y']
    dfr['ld_src_x'] = dfr['ld_src_y']

# test_list_1 = [i for i in df_original.columns.tolist() if i not in pk]

# test_list_1

pk = ['quote_header_wid', 'quote_version_number', 'quote_line_number', 'master_quote_number', 'material']


'''test_list = ['deal_registration_number', 'initial_quote_dt', 'opp_row_wid', 'opp_num', 'prod_ln', 'quote_row_wid']
test the insert addtional items in primary key found before
# # test single addition of Primary key
# for i in test_list_1:
#     pk1 = pk.append(i)
#     try:
#         df_op_inner = df_original.merge(df_Prior_Win, on = pk1, copy = False)
#         if df_op_inner.shape[0] == df_Prior_Win.shape[0]:
#             print(i)
#         else:
#             print('{} not.'.format(i))
#     except:
#         print('No single addition item match.')

'''


# inner join original and prior win
df_op_inner = df_original.merge(df_Prior_Win, on = pk, copy = False)
df_op_left = df_original.merge(df_Prior_Win, how = 'left', on = pk, copy = False,
                               indicator = True)

df_op_inner['diff_qcd'] = (df_op_inner['quote_created_date_y'] - df_op_inner['quote_created_date_x']) / np.timedelta64(1, 'D')
df_op_inner['diff_rptg'] = (df_op_inner['rptg_dt_y'] - df_op_inner['rptg_dt_x']) / np.timedelta64(1, 'D')
df_op_inner['diff_opp_close'] = (df_op_inner['opp_close_dt_y'] - df_op_inner['opp_close_dt_x']) / np.timedelta64(1, 'D')
df_op_inner['diff_opp_uda'] = (df_op_inner['opportunity_update_date_y'] - df_op_inner['opportunity_update_date_x']) / np.timedelta64(1, 'D')
df_op_inner['diff_quo_upd'] = (df_op_inner['quote_update_date_y'] - df_op_inner['quote_update_date_x']) / np.timedelta64(1, 'D')
df_op_inner['diff_ini_qd'] = (df_op_inner['initial_quote_dt_y'] - df_op_inner['initial_quote_dt_x']) / np.timedelta64(1, 'D')


# stg_nm change
df_op_inner['stg_nm_chg'] = df_op_inner.apply(lambda row: None if row['stg_nm_x'] == row['stg_nm_y'] else row['stg_nm_y'], axis = 1)

# derive configure quan & quote quan
df_op_inner['configure_quantity_chg'] = df_op_inner.configured_quantity_y - df_op_inner.configured_quantity_x
df_op_inner['quo_quan_chg'] = df_op_inner.quote_quantity_y - df_op_inner.quote_quantity_x

#drop diff_qcd, diff_ini_qd
droplist = ['diff_qcd', 'diff_ini_qd']
df_op_inner.drop(droplist, axis = 1, inplace = True)

assignvalu(df_op_inner)

df_op_inner.info()

# drop the extended cols
df_op_inner.drop(df_op_inner.columns.tolist()[44:-7], axis = 1, inplace = True)
colnames = dict(zip(df_op_inner.columns.tolist()[:-7], df_original.columns.tolist()))
df_op_inner.rename(columns = colnames, inplace = True)
df_op_inner.info()
# change the change flag value
df_op_inner.change_flg = 'Updated'
# select the unmerged rows
df_op_left = df_op_left.loc[df_op_left._merge == 'left_only']
# drop the no value cols
df_update = df_op_left.drop(df_op_left.columns.tolist()[44:], axis = 1)
df_update.info()
# rename the cols
df_update.rename(columns = colnames, inplace = True)
# get the extra derived cols
extra_col = df_op_inner.columns.tolist()[-7:]
# assign nan to the extra col in original table since it has no changed.
for i in extra_col:
    df_update[i] = np.nan
# append the op inner to update
df_update = df_update.append(df_op_inner)

#df_update.drop('_merge', axis = 1, inplace = True)
# export the data processed with previous steps
df_update.to_csv('update(priorwintooriginal).csv', index = False)
# # Merge other change name tables
df_op = pd.read_csv('update(priorwintooriginal).csv')
# merge two tables to detect whether they have the same records
df_merge_nq_op_in = df_op.merge(df_New_Quote, on = pk, copy = False)
df_merge_nq_op_in.shape
# directly append new quote with op
extra_col = df_New_Quote.shape[1] - df_op.shape[1]
df_op.columns.tolist()[extra_col:]
for i in df_op.columns.tolist()[extra_col:]:
    df_New_Quote[i] = np.nan
df_New_Quote.info()
df_opn = df_op.append(df_New_Quote)
df_opn.info()

# # New So(Prior Win - New So)
# inner join and left join
df_in_opn_nso = df_opn.merge(df_New_SO, on = pk, copy = False)
df_left_opn_nso = df_opn.merge(df_New_SO, how = 'left', on = pk, indicator=True)


# In[111]:


# change the datatype of dates into datetime
dt_list1 = ['quote_created_date_y', 'quote_created_date_x', 'rptg_dt_y', 'rptg_dt_x', 'opp_close_dt_y', 'opp_close_dt_x',
           'opportunity_update_date_y', 'opportunity_update_date_x', 'quote_update_date_y', 'quote_update_date_x',
          'initial_quote_dt_y', 'initial_quote_dt_x']
#df_list = [df_original, df_Prior_Win, df_other_change, df_New_Quote, df_New_SO, df_New_Win, df_Opport_stg_nm_Change]
for j in dt_list1:
    df_in_opn_nso[j] = pd.to_datetime(df_in_opn_nso[j])

df_in_opn_nso['diff_qcd'] = (df_in_opn_nso['quote_created_date_y'] - df_in_opn_nso['quote_created_date_x']) / np.timedelta64(1, 'D')
df_in_opn_nso['diff_rptg'] = (df_in_opn_nso['rptg_dt_y'] - df_in_opn_nso['rptg_dt_x']) / np.timedelta64(1, 'D')
df_in_opn_nso['diff_opp_close'] = (df_in_opn_nso['opp_close_dt_y'] - df_in_opn_nso['opp_close_dt_x']) / np.timedelta64(1, 'D')
df_in_opn_nso['diff_opp_uda'] = (df_in_opn_nso['opportunity_update_date_y'] - df_in_opn_nso['opportunity_update_date_x']) / np.timedelta64(1, 'D')
df_in_opn_nso['diff_quo_upd'] = (df_in_opn_nso['quote_update_date_y'] - df_in_opn_nso['quote_update_date_x']) / np.timedelta64(1, 'D')
df_in_opn_nso['diff_ini_qd'] = (df_in_opn_nso['initial_quote_dt_y'] - df_in_opn_nso['initial_quote_dt_x']) / np.timedelta64(1, 'D')


difflist = ['diff_qcd', 'diff_rptg', 'diff_opp_close', 'diff_opp_uda', 'diff_quo_upd', 'diff_ini_qd']
for i in difflist:
    print(df_in_opn_nso[i].value_counts())


# In[112]:


#drop diff_qcd, diff_ini_qd
droplist = ['diff_qcd', 'diff_ini_qd']
df_in_opn_nso.drop(droplist, axis = 1, inplace = True)


# # function for assign new updated value the date cols
# def upcols(df):
#     # upcols is to assign update values to rptg, opp close, opp ud, quo up date
#     df.rptg_dt_x = df.rptg_dt_y
#     df.opp_close_dt_x = df.opp_close_dt_y
#     df.opportunity_update_date_x = df.opportunity_update_date_y
#     df.quote_update_date_x = df.quote_update_date_y

# # assign the new value in opp_update, opp_close, rptg
# upcols(df_in_opn_nso)

# # assign so num from update to original
# df_in_opn_nso.so_num_x = df_in_opn_nso.so_num_y

# # check the flag
# df_in_opn_nso.loc[(df_in_opn_nso.quote_order_link_flag_x == np.nan) & (df_in_opn_nso.quote_order_link_flag_y == 'X')]

# In[113]:


assignvalu(df_in_opn_nso)


# In[114]:


# keep the cols we want
df_in_opn_nso_app = df_in_opn_nso[df_in_opn_nso.columns.tolist()[: df_opn.shape[1]]]

df_in_opn_nso_app.change_flg_x = 'Updated'


# In[115]:


# drop the records in left
df_opns = df_left_opn_nso.loc[df_left_opn_nso._merge == 'left_only']

df_opns = df_opns[df_opns.columns.tolist()[: df_opn.shape[1]]]


# In[116]:


df_opns.info()


# In[117]:


df_in_opn_nso_app.info()


# In[118]:


df_opns = df_opns.append(df_in_opn_nso_app)

df_opns.rename(columns=colnames, inplace = True)


# In[119]:


df_opns.info()


# # New Win

# New Win  – these are quotes that now have the quote_order_link_flag = 'X' where it was null (or the quote was not listed) in the prior weeks’ dataset!

# In[120]:


def dt_diff(df):
    # change the datetime variable dtype to datetime
    # generate the differences of datetime variables
    dt_list1 = ['quote_created_date_y', 'quote_created_date_x', 'rptg_dt_y', 'rptg_dt_x', 'opp_close_dt_y', 'opp_close_dt_x',
           'opportunity_update_date_y', 'opportunity_update_date_x', 'quote_update_date_y', 'quote_update_date_x',
          'initial_quote_dt_y', 'initial_quote_dt_x']
    #df_list = [df_original, df_Prior_Win, df_other_change, df_New_Quote, df_New_SO, df_New_Win, df_Opport_stg_nm_Change]
    for j in dt_list1:
        df[j] = pd.to_datetime(df[j])

    #df['diff_qcd'] = (df['quote_created_date_y'] - df['quote_created_date_x']) / np.timedelta64(1, 'D')
    df['diff_rptg'] = (df['rptg_dt_y'] - df['rptg_dt_x']) / np.timedelta64(1, 'D')
    df['diff_opp_close'] = (df['opp_close_dt_y'] - df['opp_close_dt_x']) / np.timedelta64(1, 'D')
    df['diff_opp_uda'] = (df['opportunity_update_date_y'] - df['opportunity_update_date_x']) / np.timedelta64(1, 'D')
    df['diff_quo_upd'] = (df['quote_update_date_y'] - df['quote_update_date_x']) / np.timedelta64(1, 'D')
    #df['diff_ini_qd'] = (df['initial_quote_dt_y'] - df['initial_quote_dt_x']) / np.timedelta64(1, 'D')

# inner join and left join with previous finished table
df_opnsw_in = df_opns.merge(df_New_Win, on=pk, copy=False)

df_opnsw_in.shape

df_opns.head()

df_New_Win.head()

df_opnsw_left = df_opns.merge(df_New_Win, how = 'left', on=pk, copy=False, indicator = True)

df_opnsw_right = df_opns.merge(df_New_Win, how = 'right', on=pk, copy=False, indicator = True)

df_opnsw_right.info()

df_New_Win.shape

# generate the diff variables
dt_diff(df_opnsw_in)

'''
xlist = df_opnsw_right.columns.tolist()[8:44]
ylist = df_opnsw_right.columns.tolist()[55:-1]
df_opnsw_right[xlist] = df_opnsw_right[ylist]

ecolx = ['oppty_header_cur_wid_x', 'configured_quantity_x', 'quote_quantity_x']
ecoly = ['oppty_header_cur_wid_y', 'configured_quantity_y', 'quote_quantity_y']
df_opnsw_right[ecolx] = df_opnsw_right[ecoly]
'''

df_opnsw_in.info()
# assign the value from right table to left table
assignvalu(df_opnsw_right)
assignvalu(df_opnsw_in)
df_opnsw_in['_merge'] = 'both'
df_opnsw = df_opnsw_left.loc[df_opnsw_left._merge == 'left_only']

# replace the inner tables to both in left tables and append the right tables right only
df_opnsw = df_opnsw.append(df_opnsw_in)

df_opnsw = df_opnsw.append(df_opnsw_right.loc[df_opnsw_right._merge == 'right_only'])

# select the target cols
df_opnsw = df_opnsw[df_opnsw.columns.tolist()[: df_opn.shape[1]]]
# done

df_opnsw.rename(columns=colnames, inplace = True)

df_opnsw.info()

df_opnsw.change_flg.value_counts()

df_opnsw.to_csv('opnsw.csv', index = False)


# # Other Change

# Other Changes (985 distinct quotes, 29,679 rows) – these are usually changes to the quote (added or removed rows, added another quote version, etc.)![image.png](attachment:image.png)

# - inner, left, right join tables
# - detect differences among three
# - replace old value by new value

# In[137]:


# inner left, right join
df_oc_in = df_opnsw.merge(df_other_change, on = pk, copy = False, indicator = True)
df_oc_left = df_opnsw.merge(df_other_change, how = 'left', on = pk, copy = False, indicator = True)
df_oc_right = df_opnsw.merge(df_other_change, how = 'right', on = pk, copy = False, indicator = True)

df_other_change.shape, df_oc_in.shape, df_oc_left.shape, df_oc_right.shape

df_oc_right._merge.value_counts()

# apply date diff
dt_diff(df_oc_in)

# assign value from y to x(inner and right join)
assignvalu(df_oc_in)
assignvalu(df_oc_right)


# df_oc_in['configure_quantity_chg'] = df_oc_in.configured_quantity_y - df_oc_in.configured_quantity_x
# df_oc_in['quo_quan_chg'] = df_oc_in.quote_quantity_y - df_oc_in.quote_quantity_x

# select left only then append both
df_oc_left = df_oc_left.loc[df_oc_left._merge == 'left_only']

# append the both table
df_opnswo = df_oc_left.append(df_oc_in)

# append right only
df_opnswo = df_opnswo.append(df_oc_right.loc[df_oc_right._merge == 'right_only'])

df_opnswo.rename(columns=colnames, inplace = True)

df_opnswo = df_opnswo[df_opnsw.columns.tolist()]

df_opnswo.info()


# # New Oppor stg nm

'''Opport stg_nm Change (227 distinct quotes, 6080 rows)
these are rows that do not meet the above criteria that have a different stg_nm different from what was reported in the prior week.
  Some of these are wins where the quote_order_link_flag has not been set (yet), 
  some are losses and some have just advanced a stage or had a change to the probability of a win.  
  Note that if the stg_nm is a win, you should consider the sale as being made, 
  even if the quote_order_link_flag is not set.![image.png](attachment:image.png)'''

df_Opport_stg_nm_Change.change_flg.value_counts()

df_opnc_in = df_opnswo.merge(df_Opport_stg_nm_Change, on = pk, copy = False)
df_opnc_left = df_opnswo.merge(df_Opport_stg_nm_Change, how = 'left', on = pk, copy = False, indicator = True)
df_opnc_right = df_opnswo.merge(df_Opport_stg_nm_Change, how = 'right', on = pk, copy = False, indicator = True)

dt_diff(df_opnc_in)

assignvalu(df_opnc_in)
assignvalu(df_opnc_right)

df_mar = df_opnc_left.loc[df_opnc_left._merge == 'left_only']

df_opnc_in['_merge'] = 'both'

df_opnc_in.info()

df_mar = df_mar.append(df_opnc_in)
df_mar = df_mar.append(df_opnc_right.loc[df_opnc_right._merge == 'right_only'])

df_mar.rename(columns=colnames, inplace = True)

df_mar = df_mar[df_opnswo.columns.tolist()]
# check point export
df_mar.to_csv('mergedrawmarchdata.csv', index = False)

df_mar = pd.read_csv('mergedrawmarchdata.csv')

df_mar1 = df_mar.copy()

df_mar1.change_flg.value_counts()

df_mar1.info()

# check q_o_flag
df_mar1.quote_order_link_flag.value_counts()
# if so num is not null, qo_flag is null
df_mar1.loc[(df_mar1.so_num != np.nan) & (df_mar1.quote_order_link_flag == np.nan)]

# if stg_nm is win - 100%, qo_flag is null
df_mar1.loc[(df_mar1.stg_nm == 'Win - 100%') & (df_mar1.quote_order_link_flag == np.nan)]

# drop [-8:] cols
df_mar1.drop(df_mar1.columns.tolist()[-8:], axis = 1, inplace = True)

# dedup
df_mar1.drop_duplicates(inplace = True)

df_mar1.info()

# cut the accumulative top 50 material
topcount = df_mar1.material.value_counts().index.tolist()

# according to 80/20 rules
topcount20 = int(len(topcount) * .2)

materialist = topcount[: topcount20]

df_mtl_30 = df_mar1.loc[df_mar1.material.isin(materialist)]
# export the filtered data
df_mtl_30.to_csv('material20.csv', index = False)
# Where to save the figures
PROJECT_ROOT_DIR = ".."
CHAPTER_ID = "plots"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, CHAPTER_ID)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

'''
pwd
ls
cd ~
'''
# read the file as df
df = pd.read_csv('material20.csv', low_memory=True)

# ship to country
df.drop('ship_to_country', axis = 1, inplace = True)


# - drop ship to country
# - Categorize dnb_amo_seg, emc_amo_seg, empl_tot_seg, enty_rgn_cd, stg_nm, ld_src, sales_organization
# - try to apply one hot encoder to materials
# - select the highest version and generate the version activate

from scipy.stats import mode

modelist = {
 'configured_quantity': 'sum',
 'quote_quantity': 'sum',
 'quote_order_type': lambda x: mode(x)[0],
 'bookbill_order_type': lambda x: mode(x)[0],
 'quote_reason_code': lambda x: mode(x)[0],
 'sales_organization': lambda x: mode(x)[0],
 'quote_order_link_flag': lambda x: mode(x)[0],
 'gro_rejection_reason': lambda x: mode(x)[0],
 #'quote_vrsn': lambda x: mode(x)[0],
 'cust_emc_sub_vert': lambda x: mode(x)[0],
 'cust_idc_maj_vert': lambda x: mode(x)[0],
 'cust_idc_maj_vert_grp': lambda x: mode(x)[0],
 'dnb_amo_seg': lambda x: mode(x)[0],
 'emc_amo_seg': lambda x: mode(x)[0],
 'empl_tot_seg': lambda x: mode(x)[0],
 'enty_rgn_cd': lambda x: mode(x)[0],
 'ld_src': lambda x: mode(x)[0],
 'vrsnact': lambda x: mode(x)[0],
 'quote_created_date_quarter': lambda x: mode(x)[0],
 'quote_update_date_quarter': lambda x: mode(x)[0],
 'initial_quote_dt_quarter': lambda x: mode(x)[0],
 'quote_created_date': lambda x: mode(x)[0],
 'quote_update_date': lambda x: mode(x)[0],
 'initial_quote_dt': lambda x: mode(x)[0],
 'win_problty': lambda x: mode(x)[0]
}

modelist

# drop the columns which are not useful
droplist = ['material', 'config_model', 'fmly', 'prod_ln', 'quote_opportunity_link_flag', 'gro_rejection_comment',
           'deal_registration_number', 'so_num', 'opp_num', 'rptg_dt', 'opp_close_dt', 'opportunity_update_date',
           'opportunity_number', 'stg_nm']
df.drop(droplist, axis = 1, inplace = True)

df.info()
# extra drop
extra_drop = ['quote_header_wid', 'oppty_header_cur_wid', 'quote_line_number', 'quote_vrsn', 'quote_row_wid']
df.drop(extra_drop, axis = 1, inplace = True)
# transform the target label as 1 or 0, 1 indicates to converted
df.quote_order_link_flag = np.where(df.quote_order_link_flag == 'X', 1, 0)
df.quote_order_link_flag.value_counts()
# fix the datetime columns' data type 
datetime = ['quote_created_date', 'quote_update_date', 'initial_quote_dt']
for i in datetime:
    df[i] = pd.to_datetime(df[i])

df.info()

# all the objects have to transform to numeric
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
# make a copy of original dataframe
df1 = df.copy()
# define a function for transform the categorical date to numeric
def masktran(data, seriesname):
    # mask the series without null value and binarize the label
    classdict = {}
    for i in seriesname:
        print('Total null value in {0} is {1}'.format(i, data[i].isnull().sum()))
        mask = data[i].notnull()
        data[i][mask] = lb.fit_transform(data[i][mask])
        print('Successful processed', data[i].head())
        classdict['{0}'.format(i)] = lb.classes_
    return classdict
# select the object columns
ojlist = df.select_dtypes(include=['object']).columns.tolist()
ojlist
# transform the objects to numeric
classdict = masktran(df1, ojlist)
classdict
# fill the null value of gro rejection reason with 0.
df.gro_rejection_reason.fillna(0, inplace = True)
df.info()
# imputer
from sklearn.preprocessing import Imputer
imp = Imputer(strategy='most_frequent')
# impute the missing value with the most frequenct value of the column
df1[ojlist] = imp.fit_transform(df1[ojlist])
# pop out the keys not in df1 columns
for i in list(modelist.keys()):
    if i not in df1.columns.tolist():
        modelist.pop(i)
    else:
        pass
# aggregate the df1 with groupby quote number and version number
df_gb = df1.groupby(['master_quote_number', 'quote_version_number']).agg(modelist)
df_gb.info()
# reverse the numeric value to categorical value
for i in df_gb.sales_organization.value_counts().index:
    print(i)
classdict['dnb_amo_seg'][1]
    # create the copy
df_gb1 = df_gb.copy()
for i in list(classdict.keys()):
    for j in df_gb1[i].value_counts().index:
        df_gb1[i].replace(j, classdict[i][int(j)], inplace = True)
df_gb1.head()
# export the aggregate table without materials
df_gb1.to_csv('aggregatednormal.csv')
df_gb1.reset_index(inplace = True)
# plot the distribution of quote order link flag
plt.figure(figsize = (10, 6))
df_gb1.quote_order_link_flag.hist()
# plot the distribution of quote version number
plt.figure(figsize = (10, 6))
sns.distplot(df_gb1.quote_version_number, color = 'g')
plt.title('Distribution of quote version number')
save_fig('vrsndist')
# plot the distribution of quote version number with bin change
plt.figure(figsize = (10, 6))
df_gb1.quote_version_number.hist(bins = 10)
# plot the histogram of win probability
df_gb1.win_problty.hist()
# filter out the quote contains multiple version
multivrsnquote = df_gb1.master_quote_number.value_counts()[df_gb1.master_quote_number.value_counts() > 2].index.tolist()
# extract the quote contains more then two versions
df_mvrsn = df_gb1.loc[df_gb1.master_quote_number.isin(multivrsnquote)]
# extract the quote contains five versions
fivevrsn = df_mvrsn.master_quote_number.value_counts()[df_mvrsn.master_quote_number.value_counts() == 5].index.tolist()
# # extract the quote contains five versions
df_mvrsn.loc[df_mvrsn.master_quote_number.isin(fivevrsn)][['master_quote_number', 'quote_version_number', 'win_problty']]
# plot the histogram of quote version number
df_mvrsn.quote_version_number.hist()
len(df_mvrsn.master_quote_number.value_counts())
# export the multiple version 
df_mvrsn.to_csv('multivrsn.csv', index = False)
# [Level of amo seg](http://www.digium.com/blog/2016/02/18/smb-sme-large-enterprise-size-business-matters/)
# 
# Enterprise > Mid-market > SMB
df.dnb_amo_seg = pd.Categorical(df.dnb_amo_seg, ordered=True, categories = ['ENTERPRISE', 'MID-MARKET', 'SMB'])

df.emc_amo_seg = pd.Categorical(df.emc_amo_seg, ordered=True, categories = ['ENTERPRISE', 'MID-MARKET', 'SMB'])

df.empl_tot_seg = pd.Categorical(df.empl_tot_seg, ordered=True, categories = ['ENTERPRISE', 'MID-MARKET', 'SMB', 'UNCL'])


# df.replace('Commit', 'Commit - 90%', inplace = True)

# df.replace('Strong Upside', 'Upside', inplace = True)

# df.stg_nm.value_counts().index.tolist()
# 
# df.stg_nm = pd.Categorical(df.stg_nm, ordered = True, 
#                           categories = ['Closed', 'Lost, Cancelled - 0%', 'Plan - 1%', 'Upside', 'Discover - 10%',
#                                         'Booked', 'Qualify - 30%', 'Pipeline', 'Propose - 60%', 'Commit - 90%', 
#                                         'Order Submitted - 99%', 'Eval', 'Win - 100%'])

df.enty_rgn_cd.value_counts()
df.dnb_amo_seg.value_counts()
df.emc_amo_seg.value_counts()
df.cust_idc_maj_vert_grp.value_counts()
df.info()
df.quote_order_link_flag.value_counts()
df.quote_order_link_flag = np.where(df.quote_order_link_flag == 'X', 1, 0)
df.config_model.value_counts()
dt_list = ['quote_created_date', 'quote_created_date', 'opp_close_dt', 'opportunity_update_date',
          'quote_update_date', 'initial_quote_dt', 'rptg_dt']
for j in dt_list:
    df[j] = pd.to_datetime(df[j])


# # one hot encoder for material and quantity

# - find out the way to encode the orinal categorical data
# - encode the flag to 10
# - drop all the unnecessary cols
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, LabelEncoder
from sklearn.feature_extraction import DictVectorizer
df_m = df[['material', 'quote_quantity']]
df_m.info()
# initial enc and dictvec
lb = LabelBinarizer()
enc = OneHotEncoder(sparse=False)
dictv = DictVectorizer(sparse=False)
le = LabelEncoder()
# convert all material of the data to dummies
df_lb = pd.DataFrame(lb.fit_transform(df_m.material), columns=lb.classes_)
# concatenate all the columns
df_lbd = pd.concat([df_m, df_lb], axis = 1)
df_lb.shape
df_m.quote_quantity.shape
df_lbd.head()
# multiply the quote quantity with material dummies
for i in df_lbd.columns.tolist()[2:]:
    df_lbd[i] = df_lbd.quote_quantity * df_lbd[i]
df1 = pd.concat([df, df_lbd[df_lbd.columns.tolist()[2:]]], axis = 1)
df_test1 = df1.copy()
# export the data
df1.to_csv('dummymaterial.csv', index = False)

df_test1 = pd.read_csv('dummymaterial.csv', low_memory=True).copy()
df_test1.columns.tolist()[18]
# bin the rptg_dt, opp_close_dt, opportunity update date into quarters

droplist = ['material', 'config_model', 'fmly', 'prod_ln', 'quote_opportunity_link_flag', 'gro_rejection_comment',
            'deal_registration_number', 'so_num', 'opp_num', 'rptg_dt', 'opp_close_dt', 'opportunity_update_date',
            'opportunity_number', 'stg_nm']
df_test1.drop(droplist, axis=1, inplace=True)
df_test1.dnb_amo_seg = pd.Categorical(df_test1.dnb_amo_seg, ordered=True, categories=[
                                      'ENTERPRISE', 'MID-MARKET', 'SMB'])
df_test1.emc_amo_seg = pd.Categorical(df_test1.emc_amo_seg, ordered=True, categories=[
                                      'ENTERPRISE', 'MID-MARKET', 'SMB'])
df_test1.empl_tot_seg = pd.Categorical(df_test1.empl_tot_seg, ordered=True, categories=[
                                       'ENTERPRISE', 'MID-MARKET', 'SMB', 'UNCL'])
df_test1.gro_rejection_reason.fillna(0, inplace=True)
df_test1.gro_rejection_reason = np.where(
    df_test1.gro_rejection_reason != 0, 1, 0)
df_test1.gro_rejection_reason.value_counts()
df_test1.shape
df_test1.quote_order_link_flag.value_counts()
df_test1[df_test1.columns.tolist()[:30]].info()
catlist = df_test1.select_dtypes(include=['category']).columns.tolist()
ojlist = df_test1.select_dtypes(include=['object']).columns.tolist()
ojlist = [i for i in ojlist if i not in [
    'quote_created_date', 'quote_update_date', 'initial_quote_dt']]
df_test1[ojlist].info()
df_test1.bookbill_order_type.value_counts()
df_test1[df_test1.columns.tolist()[:30]].info()


# # quote vrsn contains missing value
# df_test1.quote_vrsn = df_test1.quote_vrsn.apply(lambda x: x['quote_version_number'] if x['quote_vrsn'] == np.nan else x['quote_vrsn'], axis = 1)
# convert 'quote_created_date', 'quote_update_date', 'initial_quote_dt' to quarter
df_test1['quote_created_date_quarter'] = df_test1.quote_created_date.apply(
    lambda x: pd.Timestamp(x).quarter)
df_test1['quote_update_date_quarter'] = df_test1.quote_update_date.apply(
    lambda x: pd.Timestamp(x).quarter)
df_test1['initial_quote_dt_quarter'] = df_test1.initial_quote_dt.apply(
    lambda x: pd.Timestamp(x).quarter)
df_test1.quote_reason_code = df_test1.quote_reason_code.apply(lambda x: str(x))
df_test1.quote_order_link_flag.value_counts()


def masktran(data, seriesname):
    # mask the series without null value and binarize the label
    for i in seriesname:
        print('Total null value in {0} is {1}'.format(
            i, data[i].isnull().sum()))
        mask = data[i].notnull()
        print('mask:', mask[0:5])
        data[i][mask] = lb.fit_transform(data[i][mask])
        print('Successful processed', data[i].head())


masktran(df_test1, ojlist)
for i in catlist:
    df_test1[i] = df_test1[i].cat.codes
df_test1[ojlist].head()
ojlist
# imputer
from sklearn.preprocessing import Imputer
imp = Imputer(strategy='most_frequent')
df_test1[ojlist] = imp.fit_transform(df_test1[ojlist])
df_test1.columns.tolist()[:30]
df_test1.quote_update_date.value_counts()
df_test1.bookbill_order_type.value_counts()
df_test1.quote_order_type.value_counts()
df_test1.quote_created_date.value_counts()
df_cat = df1.select_dtypes(exclude=[np.number])
dtlist = df_test1.select_dtypes(exclude=[np.number]).columns.tolist()
for j in dtlist:
    df_test1[j] = pd.to_datetime(df_test1[j])
df_test1[df_test1.columns.tolist()[:30]].info()


# # Encode the object and cat cols

# df_cat.info()

# df_cat.so_num.value_counts()

# df2 = df1.loc[df1.master_quote_number == 6001512672]

# df2.head()

# df3 = df2.select_dtypes(include= [np.number])

# df1.info(max_cols=100)
for i in ['quote_created_date', 'quote_update_date', 'initial_quote_dt']:
    df_test1[i] = pd.to_datetime(df_test1[i])
# generate the version active
df_test1['vrsnact'] = ((df_test1.quote_created_date - df_test1.initial_quote_dt) /
                       np.timedelta64(1, 'D')) / df_test1.quote_version_number
df_test1[df_test1.columns.tolist()[:30]].info()
df_test1.vrsnact.value_counts()
df_test1.drop('quote_vrsn', axis=1, inplace=True)
df_test1.to_csv('numericmaterial.csv', index=False, )
df_test1 = pd.read_csv('numericmaterial.csv', low_memory=True)
df_test1.columns.tolist()[-4:]
df_test1.columns.tolist()[:27]
materialist = df_test1.columns.tolist()[27: -4]
materialist[-10:]
sumlist = ['sum'] * len(materialist)
agg = dict(zip(materialist, sumlist))
agg.update(modelist)
agg
len(materialist)
df1.head(10)
aggrega = {'configured_quantity': 'sum',
           'quote_quantity': 'sum',
           'quote_order_type': 'mode',
           }
df_test1.drop(['oppty_header_cur_wid', 'opp_row_wid',
               'quote_header_wid', 'quote_line_number'], axis=1, inplace=True)
df5 = df_test1.groupby(['master_quote_number', 'quote_version_number',
                        'quote_header_wid', 'quote_row_wid']).agg(agg)
df4 = df_test1.groupby(
    ['master_quote_number', 'quote_version_number']).agg(agg)
df4.to_csv('aggbyvrsnquote.csv', index=True)

# coding: utf-8

# import the packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns
import os
sns.set_style(style = 'darkgrid')

# Where to save the figures
PROJECT_ROOT_DIR = ".."
CHAPTER_ID = "plots"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, CHAPTER_ID)
# set up the figure save path
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# read the structured table
df = pd.read_csv('aggbyvrsnquote.csv', low_memory=True)

df.info()
# ## Create the different time among three time variables.

# select the three time variable
dt_list = df.select_dtypes(include=['object']).columns.tolist()

dt_list

# change the time variable data type to datetime
for i in dt_list:
    df[i] = pd.to_datetime(df[i])

# generate the diff time of the 3 datetime value
df['diff_create_initial'] = (df.quote_created_date - df.initial_quote_dt) / np.timedelta64(1, 'D')
df['diff_update_initial'] = (df.quote_update_date - df.initial_quote_dt) / np.timedelta64(1, 'D')
df['diff_create_update'] = (df.quote_update_date - df.quote_created_date) / np.timedelta64(1, 'D')

# drop master quote number
df.drop('master_quote_number', axis = 1, inplace=True)
# create X contains all predictors and y contains target labels
X = df.drop(['quote_order_link_flag', 'win_problty'], axis = 1).select_dtypes(include=[np.number])
y = df.quote_order_link_flag

# extract the feature names
featurenames = X.columns
# import the sklearn models
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, validation_curve, learning_curve, cross_validate, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# scoring list for cross validation scoring.
scores = ['precision', 'recall', 'f1', 'accuracy', 'roc_auc']

def crosspred(classifier):
    '''function of applying cross validation predict'''
    return cross_val_predict(classifier, X_train, y_train, cv = 10)

def tenfold(classifier):
    '''function of applying 10-fold crosee validation along with scorings'''
    return cross_validate(classifier, X_train, y_train, cv = 10, scoring=scores)

# def function for ploting feature importance
def plot_feature_importance(classifier, figname, n_features = 10):
    '''This function is to plot the feature importance of Tree model i.e Decision Tree and RandomForest
        
        Classifier: the classifier that already fit the model.
        n_features: number of features to plot.
        figname: the figure name to save.
    '''
    feature_importance = pd.DataFrame(classifier.feature_importances_.reshape(1, -1), 
                                  columns=X_train.columns.tolist()).T
    feature_importance.columns = ['importance']
    feature_imp = feature_importance.loc[feature_importance.importance != 0].sort_values(by='importance',
                                                                                     ascending=False)
    fig = plt.figure(figsize = (10, 6))
    plt.title('Feature Importance')
    plt.barh(range(n_features), feature_imp.importance[:n_features], color = 'b', align = 'center')
    plt.yticks(range(n_features), feature_imp.index[:n_features])
    plt.xlabel('Relative Importance')
    plt.gca().invert_yaxis()
    fig.savefig('{0}.png'.format(figname), dpi = fig.dpi)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 42, stratify = y)

# get the columns names
cols = X_train.columns.tolist()

X_train.shape, X_test.shape


# ## Decision Tree

# create the instance of standardscaler
std = StandardScaler()
# standardize the X_train
X_train = std.fit_transform(X_train)

# create instance of Decision Tree
clf = DecisionTreeClassifier(max_depth = 10, criterion='entropy', random_state=42)

# fit the decision tree
clf.fit(X_train, y_train)

# create max depth = 5 decision tree
# clf5 = DecisionTreeClassifier(max_depth=5, criterion='gini')
# 
# clf5.fit(X_train, y_train)

# create the plot of feature importance
plot_feature_importance()

# make the predictions
y_pred_dt = clf.predict(X_test)

y_pred_dt5 = clf5.predict(X_test)
# output the accuracy
accuracy_score(y_test, y_pred_dt)
# measure the auc score
roc_auc_score(y_test, y_pred_dt)
# calculate the specificity
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_dt).ravel()
specificity_dt = tn / (tn + fp)
print(specificity_dt)


# ### Grid Search
parameters = {'criterion': ['entropy', 'gini'],
             'max_depth': [10, 100, 200]}
gs_dt = GridSearchCV(estimator=clf, param_grid=parameters, scoring='accuracy', cv = 5, n_jobs=5)
gs_dt.fit(X_train, y_train)
# gather the best parameters and best score from grid search
gs_dt.best_params_, gs_dt.best_score_
y_pred_gs_dt = gs_dt.predict(X_test)
print(classification_report(y_test, y_pred_gs_dt), '\n', accuracy_score(y_test, y_pred_gs_dt))
# plot the feature importance
gs_dt.feature_importance
feature_importance = pd.DataFrame(clf.feature_importances_.reshape(1, -1), 
                                  columns=cols).T

feature_importance.columns = ['importance']
feature_imp = feature_importance.loc[feature_importance.importance != 0].sort_values(by='importance', ascending=False)
feature_imp.index[:10]
feature_imp.importance[:10]
importances = clf.feature_importances_
indices = np.argsort(importances)
fig = plt.figure(figsize = (10, 6))
plt.title('Feature Importance')
plt.barh(range(10), feature_imp.importance[:10], color = 'b', align = 'center')
plt.yticks(range(10), feature_imp.index[:10])
plt.xlabel('Relative Importance')
plt.gca().invert_yaxis()
fig.savefig('DecisionTreeFeatureImp.png', dpi = fig.dpi)
features = feature_imp.index.tolist()


# create new X with feature importance is not 0.
X = df[features]
y = df['quote_order_link_flag']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, stratify = y)
clf_dt = DecisionTreeClassifier(criterion='entropy', max_depth=10)
# 10 fold validation
score_dt = tenfold(clf_dt)
score_dt
clf_dt.fit(X_train, y_train)
# plot the feature importance
plot_feature_importance(clf_dt, 'decisiontree')
# get the test auc score from 10 fold cross validation
np.mean(score_dt['test_roc_auc'])
# make the predict for test set
y_pred_dt = clf_dt.fit(X_train, y_train).predict(X_test)
y_pred_dt5 = clf5.predict(X_test)
print(classification_report(y_test, y_pred_dt5))
roc_auc_score(y_test, y_pred_dt5)


# ### Random Forest
# reset the training set and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 42, stratify = y)
clf_rf = RandomForestClassifier(max_depth = 100,
                                n_estimators=400, random_state=0, n_jobs=4, criterion='entropy')
clf_rf.fit(X_train, y_train)
plot_feature_importance(classifier=clf_rf, figname='Randomforest')
# 10 fold cross validation
score_rf = cross_val_score(clf_rf, X_train, y_train, cv = 10, n_jobs=4, scoring='roc_auc')
np.mean(score_rf)
# grid search
parameters = {'n_estimators': [200, 300, 400],
            'criterion': ['entropy'],
             'max_depth': [100, 200, 500]}
gs_rf = GridSearchCV(estimator=clf_rf, param_grid=parameters, scoring='accuracy', cv = 5, n_jobs=5)
gs_rf.fit(X_train, y_train)
y_pred_rf = gs_rf.predict(X_test)
gs_rf.best_params_, gs_rf.best_score_
clf_rf = RandomForestClassifier(max_depth = 100, n_estimators=400, random_state=0, n_jobs=4, criterion='entropy')
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)
print(classification_report(y_test, y_pred_rf), '\n', accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
y_pred_rf = clf_rf.fit(X_train, y_train).predict(X_test)
roc_auc_score(y_test, y_pred_rf)
# specificity
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_rf).ravel()
specificity_rf = tn / (tn + fp)
print(specificity_rf)
# ## XGB
# import the xgboost 
import xgboost as xgb
from sklearn.metrics import confusion_matrix


# dm = xgb.DMatrix(data = X_train, label=y_train)
# params = {'objective': 'binary:logistic', 'n_estimators': 10}

# xgb_log = xgb.train(params=params, dtrain=dm, num_boost_round=50)
# create a instance of xgb classifier
clf_xgb = xgb.XGBClassifier(n_jobs=5, n_estimators=806, max_depth=5)
clf_xgb.fit(X_train, y_train)
clf_xgb.feature_importances_
plot_feature_importance(clf_xgb, 'xgb')
y_pred_xgb = clf_xgb.predict(X_test)
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_dt)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_xgb)
plt.figure(figsize = (10, 6))
plt.plot([0, 1], [0, 1], 'k--', lw = 2, color = 'navy')
plt.plot(fpr_dt, tpr_dt, label='DecisionTree', lw = 2)
plt.plot(fpr_rf, tpr_rf, label='RandomForest', lw = 2)
plt.plot(fpr_xgb, tpr_xgb, label='XGBoost', lw = 2)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve of all models applied')
plt.legend(loc='best')
save_fig('ROC_Curve');
y_pred_prob = clf_xgb.predict_proba(X_test)
y_pred_prob
print(confusion_matrix(y_test, y_pred_xgb))

print(classification_report(y_test, y_pred_xgb))

print(roc_auc_score(y_test, y_pred_xgb))
# grid search
params = {'gamma': [0.1, 0.5, 1],
         'learning_rate': [0.1, 1, 10]}

gs_xgb = GridSearchCV(param_grid=params, estimator=clf_xgb, n_jobs=5, cv = 10)

gs_xgb.fit(X_train, y_train)

y_pred_gsxgb = gs_xgb.predict(X_test)

print(classification_report(y_test, y_pred_gsxgb))
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_xgb).ravel()
specificity_xgb = tn / (tn + fp)
print(specificity_xgb)
xgb.plot_importance(clf_xgb, max_num_features=10, importance_type='weight', )
xgb.to_graphviz(clf_xgb)
fig, ax = plt.subplots(figsize=(25, 15))
xgb.plot_tree(clf_xgb, ax=ax);


# ## Logistic Regression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
pipeline = Pipeline([('pca', PCA(random_state=42)),
                     ('Logit', LogisticRegressionCV())])

from sklearn.linear_model import LogisticRegressionCV
clf_lg = LogisticRegression(random_state=42)
clf_lg = LogisticRegressionCV(cv = 10)
clf_lg.fit(X_train, y_train)
len(cols), len(clf_lg.coef_.item)
# grid search
paramlg = {'C': [10, 100],
          'penalty': ['l1', 'l2']}
gs_lg = GridSearchCV(clf_lg, param_grid=paramlg, n_jobs=5, cv = 10, scoring='accuracy')
gs_lg.fit(X_train, y_train)
gs_lg.best_score_
coef = pd.DataFrame({'features': cols, 'coef': clf_lg.coef_.tolist()[0]})
coef.sort_values(by = 'coef', ascending=False)[:10]
coef.sort_values(by = 'coef')[:10]
y_pred_lg = pipeline.fit(X_train, y_train).predict(X_test)
y_pred_lg1 = clf_lg.predict(X_test)
print(classification_report(y_test, y_pred_lg1))
print(accuracy_score(y_test, y_pred_lg1))
print(roc_auc_score(y_test, y_pred_lg1))
pipeline.coef_
pipeline.steps.pop(1)
pipeline.steps.append(('Logit', LogisticRegression(random_state=42)))
# grid search
params_lg = {'Logit__C': [1, 10, 100]}
clf_gs_lg = GridSearchCV(pipeline, param_grid=params_lg, cv = 5, n_jobs=5, scoring='accuracy')
clf_gs_lg.fit(X_train, y_train)
print(clf_gs_lg.best_params_, clf_gs_lg.best_score_)
# import the package to print out the tree
from sklearn.externals.six import StringIO
import pydot
from sklearn import tree
from sklearn import metrics
tree.export_graphviz(clf5, out_file = 'tree.dot')
out_data = StringIO()
tree.export_graphviz(clf5,out_file=out_data,
                    feature_names = featurenames,
                    class_names=clf5.classes_.astype(int).astype(str),
                    filled=True, rounded=True,
                    special_characters=True,
                    node_ids=1,)
graph = pydot.graph_from_dot_data(out_data.getvalue())
#graph[0].write_pdf('decisionTree.pdf')
graph[0].write_png('dtmax5_new.png')

