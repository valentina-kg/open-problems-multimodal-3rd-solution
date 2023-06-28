# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# %%capture output
# !pip install --upgrade pip
# !pip install --upgrade pandas
# !pip install tables   
# necessary for pd.read_hdf()

# !pip install ipywidgets
# !pip install --upgrade jupyter
# !pip install IProgress
# !pip install catboost
# !pip install shap
# !pip install anndata
# -

import os
import random
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GroupKFold
import scipy
import anndata as ad
import shap

# +
# %matplotlib inline
from tqdm.notebook import tqdm
import gc
import pickle

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda")
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
# -

# ## data load

# +
lrz_path = '/dss/dssfs02/lwp-dss-0001/pn36po/pn36po-dss-0001/di93zoj/open-problems-multimodal-3rd-solution/'

model_path_for_now = '/dss/dsshome1/02/di93zoj/valentina/open-problems-multimodal-3rd-solution/'

raw_path =  lrz_path + 'input/raw/'  # '../../../input/raw/'

cite_target_path = lrz_path + 'input/target/cite/'   # '../../../input/target/cite/'
cite_feature_path = lrz_path + 'input/features/cite/'   # '../../../input/features/cite/'
cite_mlp_path = lrz_path + 'model/cite/mlp/'   # '../../../model/cite/mlp/'   # '../../../model/cite/mlp/'
cite_cb_path = lrz_path + 'model/cite/cb/'   # '../../../model/cite/cb/'

multi_target_path = lrz_path + 'input/target/multi/'   # '../../../input/target/multi/'
multi_feature_path = lrz_path + 'input/features/multi/'   # '../../../input/features/multi/'
multi_mlp_path = lrz_path + 'model/multi/mlp/'   # '../../../model/multi/mlp/'
multi_cb_path = lrz_path + 'model/multi/cb/'   # '../../../model/multi/cb/'

index_path = lrz_path + 'input/preprocess/cite/'

output_path = lrz_path + 'output/'   # '../../../output/'
# -

# ## Cite

# +
# get model name
#mlp_model_path = os.listdir(cite_mlp_path)
# -

mlp_model_name = [
    'corr_add_con_imp',
    'corr_last_v3', 
    'corr_c_add_w2v_v1_mish_flg',
    'corr_c_add_w2v_v1_flg',
    'corr_c_add_84_v1',
    'corr_c_add_120_v1',
    'corr_w2v_cell_flg',
    'corr_best_cell_120',
    'corr_cluster_cell',
    'corr_w2v_128',
    'corr_imp_w2v_128',
    'corr_snorm',
    'corr_best_128',
    'corr_best_64',
    'corr_cluster_128',
    'corr_cluster_64',
    'corr_svd_128',
    'corr_svd_64',
             ]

# +
model_name_list = []

for i in mlp_model_name:
    for num, j in enumerate(os.listdir(cite_mlp_path)):
        if i in j:
            model_name_list.append(j)

len(model_name_list)
model_name_list

# +
weight = [1, 0.3, 1, 1, 1, 1, 1, 1, 1, 0.8, 0.8, 0.8, 0.8, 0.5, 0.5, 0.5, 1, 1, 2, 2]
weight_sum = np.array(weight).sum()
weight_sum

model_feat_dict = {model_name_list[0]:['X_test_add_con_imp.pickle', 1],
                   model_name_list[1]:['X_test_last_v3.pickle', 0.3],
                   model_name_list[2]:['X_test_c_add_w2v_v1.pickle', 1],
                   model_name_list[3]:['X_test_c_add_w2v_v1.pickle', 1],
                   model_name_list[4]:['X_test_c_add_84_v1.pickle', 1],
                   model_name_list[5]:['X_test_c_add_v1.pickle', 1],
                   
                   model_name_list[6]:['X_test_feature_w2v_cell.pickle', 1],
                   model_name_list[7]:['X_test_best_cell_128_120.pickle', 1],
                   model_name_list[8]:['X_test_cluster_cell_128.pickle', 1],
                   
                   model_name_list[9]:['X_test_feature_w2v.pickle', 0.8],
                   model_name_list[10]:['X_test_feature_imp_w2v.pickle',0.8],
                   model_name_list[11]:['X_test_feature_snorm.pickle', 0.8],
                   model_name_list[12]:['X_test_best_128.pickle', 0.8],
                   model_name_list[13]:['X_test_best_64.pickle', 0.5],
                   model_name_list[14]:['X_test_cluster_128.pickle', 0.5],
                   model_name_list[15]:['X_test_cluster_64.pickle', 0.5],
                   model_name_list[16]:['X_test_svd_128.pickle', 1],
                   model_name_list[17]:['X_test_svd_64.pickle', 1],
                   
                   'best_128':['X_test_best_128.pickle', 2],
                   'best_64':['X_test_best_64.pickle', 2],
                  }


# -

# ### cite model

def std(x):
    x = np.array(x)
    return (x - x.mean(1).reshape(-1, 1)) / x.std(1).reshape(-1, 1)


class CiteDataset(Dataset):
    
    def __init__(self, feature, target):
        
        self.feature = feature
        self.target = target
        
    def __len__(self):
        return len(self.feature)
    
    def __getitem__(self, index):
                
        d = {
            "X": self.feature[index],
            "y" : self.target[index],
        }
        return d


class CiteDataset_test(Dataset):
    
    def __init__(self, feature):
        self.feature = feature
        
    def __len__(self):
        return len(self.feature)
    
    def __getitem__(self, index):
                
        d = {
            "X": self.feature[index]
        }
        return d


# +
def partial_correlation_score_torch_faster(y_true, y_pred):
    """Compute the correlation between each rows of the y_true and y_pred tensors.
    Compatible with backpropagation.
    """
    y_true_centered = y_true - torch.mean(y_true, dim=1)[:,None]
    y_pred_centered = y_pred - torch.mean(y_pred, dim=1)[:,None]
    cov_tp = torch.sum(y_true_centered*y_pred_centered, dim=1)/(y_true.shape[1]-1)
    var_t = torch.sum(y_true_centered**2, dim=1)/(y_true.shape[1]-1)
    var_p = torch.sum(y_pred_centered**2, dim=1)/(y_true.shape[1]-1)
    return cov_tp/torch.sqrt(var_t*var_p)

def correl_loss(pred, tgt):
    """Loss for directly optimizing the correlation.
    """
    return -torch.mean(partial_correlation_score_torch_faster(tgt, pred))


# -

class CiteModel(nn.Module):
    
    def __init__(self, feature_num):
        super(CiteModel, self).__init__()
        
        self.layer_seq_256 = nn.Sequential(nn.Linear(feature_num, 256),
                                           nn.Linear(256, 128),
                                       nn.LayerNorm(128),
                                       nn.ReLU(),
                                      )
        self.layer_seq_64 = nn.Sequential(nn.Linear(128, 64),
                                       nn.Linear(64, 32),
                                       nn.LayerNorm(32),
                                       nn.ReLU(),
                                      )
        self.layer_seq_8 = nn.Sequential(nn.Linear(32, 16),
                                         nn.Linear(16, 8),
                                       nn.LayerNorm(8),
                                       nn.ReLU(),
                                      )
        
        self.head = nn.Linear(128 + 32 + 8, 140)
                   
    def forward(self, X, y=None):
        
        from_numpy = False
        
      ##
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
            from_numpy = True
        X = X.to(device)  # Move the input to the appropriate device if necessary
        ##
        X_256 = self.layer_seq_256(X)
        X_64 = self.layer_seq_64(X_256)
        X_8 = self.layer_seq_8(X_64)
        
        X = torch.cat([X_256, X_64, X_8], axis = 1)
        out = self.head(X)
        
        if from_numpy:
            out = out.cpu().detach().numpy()
            
        return out


class CiteModel_mish(nn.Module):
    
    def __init__(self, feature_num):
        super(CiteModel_mish, self).__init__()
        
        self.layer_seq_256 = nn.Sequential(nn.Linear(feature_num, 256),
                                           nn.Linear(256, 128),
                                       nn.LayerNorm(128),
                                       nn.Mish(),
                                      )
        self.layer_seq_64 = nn.Sequential(nn.Linear(128, 64),
                                       nn.Linear(64, 32),
                                       nn.LayerNorm(32),
                                       nn.Mish(),
                                      )
        self.layer_seq_8 = nn.Sequential(nn.Linear(32, 16),
                                         nn.Linear(16, 8),
                                       nn.LayerNorm(8),
                                       nn.Mish(),
                                      )
        
        self.head = nn.Linear(128 + 32 + 8, 140)
                   
    def forward(self, X, y=None):
    
        X_256 = self.layer_seq_256(X)
        X_64 = self.layer_seq_64(X_256)
        X_8 = self.layer_seq_8(X_64)
        
        X = torch.cat([X_256, X_64, X_8], axis = 1)
        out = self.head(X)
        
        return out


def train_loop(model, optimizer, loader, epoch):
    
    losses, lrs = [], []
    model.train()
    optimizer.zero_grad()
    #loss_fn = nn.MSELoss()
    
    with tqdm(total=len(loader),unit="batch") as pbar:
        pbar.set_description(f"Epoch{epoch}")
        
        for d in loader:
            X = d['X'].to(device)
            y = d['y'].to(device)
            
            logits = model(X)
            loss = correl_loss(logits, y)
            #loss = torch.sqrt(loss_fn(logits, y))
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({"loss":loss.item()})
            pbar.update(1)

    return model


def valid_loop(model, loader, y_val):
    
    model.eval()
    partial_correlation_scores = []
    oof_pred = []
    
    for d in loader:
        with torch.no_grad():
            val_X = d['X'].to(device).float()
            val_y = d['y'].to(device)
            logits = model(val_X)
            oof_pred.append(logits)
    
    #print(torch.cat(oof_pred).shape, torch.cat(oof_pred).detach().cpu().numpy().shape)
    cor = partial_correlation_score_torch_faster(torch.tensor(y_val).to(device), torch.cat(oof_pred))
    cor = cor.mean().item()
    logits = torch.cat(oof_pred).detach().cpu().numpy()
    
    return logits, cor


def test_loop(model, loader):
    
    model.eval()
    predicts=[]

    for d in tqdm(loader):
        with torch.no_grad():
            X = d['X'].to(device)
            logits = model(X)
            predicts.append(logits.detach().cpu().numpy())
            
    return np.concatenate(predicts)


# ### pred

# +
pred = np.zeros([48203, 140])

for num, i in enumerate(model_feat_dict.keys()):
    
    print(i)
    
    if 'mlp' in i:

        try:
            test_file = model_feat_dict[i][0]
            test_weight = model_feat_dict[i][1]
            X_test = pd.read_pickle(cite_feature_path  + test_file)   
            # print(cite_feature_path  + test_file)
            X_test = np.array(X_test)
            feature_dims = X_test.shape[1]

            test_ds = CiteDataset_test(X_test)
            test_dataloader = DataLoader(test_ds, batch_size=128, pin_memory=True, 
                                         shuffle=False, drop_last=False, num_workers=4)

            if 'mish' in i:
                model = CiteModel_mish(feature_dims)
            else:
                model = CiteModel(feature_dims)

            model = model.to(device)
            model.load_state_dict(torch.load(f'{cite_mlp_path}/{i}'))

            result = test_loop(model, test_dataloader).astype(np.float32)
            result = std(result) * test_weight / weight_sum
            pred += result

            torch.cuda.empty_cache()

        except Exception as e: 
            print(i)
            print(e)             # TODOOOOOOOOOOOOOO
        
    else:
        test_file = model_feat_dict[i][0]
        test_weight = model_feat_dict[i][1]
        X_test = pd.read_pickle(cite_feature_path  + test_file)
        
        cb_pred = np.zeros([48203, 140])
        
        for t in tqdm(range(140)): 
            cb_model_path = [j for j in os.listdir(cite_cb_path) if f'cb_{t}_{i}' in j][0]
            cb = pickle.load(open(cite_cb_path + cb_model_path, 'rb'))
            cb_pred[:,t] = cb.predict(X_test)
            
        cb_pred = cb_pred.astype(np.float32)
        pred += std(cb_pred) * test_weight / weight_sum
        
        del cb_pred
# -

cite_sub = pd.DataFrame(pred.round(6))
cite_sub

# +
#cite_sub.to_csv('../../../../../summary/output/submit/cite_submit.csv')

# +
# model #16: cite_mlp_corr_svd_128_flg_donor_val_30
pred_16 = np.zeros([48203, 140])

i = 'cite_mlp_corr_svd_128_flg_donor_val_30'
        
test_file = model_feat_dict[i][0]
test_weight = model_feat_dict[i][1]
X_test = pd.read_pickle(cite_feature_path  + test_file)
X_test = np.array(X_test)
feature_dims = X_test.shape[1]

test_ds = CiteDataset_test(X_test)
test_dataloader = DataLoader(test_ds, batch_size=128, pin_memory=True, 
                              shuffle=False, drop_last=False, num_workers=4)

if 'mish' in i:
    model = CiteModel_mish(feature_dims)
else:
    model = CiteModel(feature_dims)
    
model = model.to(device)
model.load_state_dict(torch.load(f'{cite_mlp_path}/{i}'))

result = test_loop(model, test_dataloader).astype(np.float32)
pred_16 += result

torch.cuda.empty_cache()
        
pd.DataFrame(pred_16)   # double check train_cite_targets.h5  -> omnipath
# -

# ### prediction with private test input -> should get private test target

private_test_input = ad.read('/dss/dssfs02/lwp-dss-0001/pn36po/pn36po-dss-0001/di93zoj/large_preprocessed_files/private_test_input.h5ad')
private_test_input

private_test_target = ad.read('/dss/dssfs02/lwp-dss-0001/pn36po/pn36po-dss-0001/di93zoj/large_preprocessed_files/private_test_target.h5ad')
private_test_target

private_test_target_raw = ad.read('/dss/dssfs02/lwp-dss-0001/pn36po/pn36po-dss-0001/di93zoj/large_preprocessed_files/private_test_target_raw.h5ad')
private_test_target_raw

with open('private_X_test_svd.pkl', 'rb') as f:  # private_X_test_svd

    private_X_test_svd = pickle.load(f)
private_X_test_svd.shape


with open('private_X_test_svd_from_raw.pkl', 'rb') as f:  # private_X_test_svd

    private_X_test_svd_from_raw = pickle.load(f)
private_X_test_svd_from_raw.shape

# +
# model #16: cite_mlp_corr_svd_128_flg_donor_val_30
pred_16 = np.zeros([26867, 140])

i = 'cite_mlp_corr_svd_128_flg_donor_val_30'
        
# test_file = model_feat_dict[i][0]
# test_weight = model_feat_dict[i][1]
X_test = private_X_test_svd_from_raw
X_test = np.array(X_test)
feature_dims = X_test.shape[1]

test_ds = CiteDataset_test(X_test)
test_dataloader = DataLoader(test_ds, batch_size=128, pin_memory=True, 
                              shuffle=False, drop_last=False, num_workers=4)

if 'mish' in i:
    model = CiteModel_mish(feature_dims)
else:
    model = CiteModel(feature_dims)
    
model = model.to(device)
model.load_state_dict(torch.load(f'{cite_mlp_path}/{i}'))

result = test_loop(model, test_dataloader).astype(np.float32)
pred_16 += result

torch.cuda.empty_cache()
        
pd.DataFrame(pred_16)
# -

pd.DataFrame(private_test_target.X)

pd.DataFrame(private_test_target_raw.X.toarray())

pd.concat([pd.DataFrame(pred_16), pd.DataFrame(private_test_target.X)]).corr().head()

# +
# TODO check svd output to see if svd model is correct
# -

# ### - add cell_ids to train and test data
# ### - SHAP

# +
train_ids = np.load(index_path + "train_cite_raw_inputs_idxcol.npz", allow_pickle=True)
test_ids = np.load(index_path + "test_cite_raw_inputs_idxcol.npz", allow_pickle=True)

train_index = train_ids["index"]
train_column = train_ids["columns"]
test_index = test_ids["index"]
print(len(list(train_index)))
print(len(list(test_index)))
X_train_cell_ids = pd.read_pickle(cite_feature_path  + 'X_svd_128.pickle')   # = X_svd_128 in make-features second to last cell
X_train_cell_ids.index = train_index
X_train_cell_ids
# -

# cell type from metadata
X_test_cell_ids = pd.read_pickle(cite_feature_path  + test_file)
X_test_cell_ids.index = test_index
X_test_cell_ids

metadata = pd.read_csv('/dss/dssfs02/lwp-dss-0001/pn36po/pn36po-dss-0001/di93zoj/neurips_competition_data/' + 'metadata.csv')
metadata.head()

X_test_cell_ids = X_test_cell_ids.reset_index().rename(columns = {'index': 'cell_id'})
X_test_cell_ids = X_test_cell_ids.merge(metadata[['cell_id', 'cell_type']], on = 'cell_id', how = 'left')

X_test_cell_ids['cell_type'].value_counts()

X_test_cell_ids

# +
samples_per_cell_type = 5

grouped = X_test_cell_ids.groupby('cell_type')

X_test_shap = pd.DataFrame()

# Iterate over each group (cell_type)
for cell_type, group in grouped:
    sampled_rows = group.sample(n=samples_per_cell_type, replace=False)
#     X_test_shap = X_test_shap.append(sampled_rows)   # deprecated
    X_test_shap = pd.concat([X_test_shap, sampled_rows])

X_test_shap = X_test_shap.reset_index(drop=True)
print(X_test_shap.shape)
X_test_shap.head()
# -

# rename imp_ columns to gene ids:
gene_ids = ['ENSG00000075340_ADD2', 'ENSG00000233968_AL157895.1',
        'ENSG00000029534_ANK1', 'ENSG00000135046_ANXA1',
        'ENSG00000130208_APOC1', 'ENSG00000047648_ARHGAP6',
        'ENSG00000101200_AVP', 'ENSG00000166710_B2M',
        'ENSG00000130303_BST2', 'ENSG00000172247_C1QTNF4',
        'ENSG00000170458_CD14', 'ENSG00000134061_CD180',
        'ENSG00000177455_CD19', 'ENSG00000116824_CD2',
        'ENSG00000206531_CD200R1L', 'ENSG00000012124_CD22',
        'ENSG00000272398_CD24', 'ENSG00000139193_CD27',
        'ENSG00000105383_CD33', 'ENSG00000174059_CD34',
        'ENSG00000135218_CD36', 'ENSG00000004468_CD38',
        'ENSG00000010610_CD4', 'ENSG00000026508_CD44',
        'ENSG00000117091_CD48', 'ENSG00000169442_CD52',
        'ENSG00000135404_CD63', 'ENSG00000173762_CD7',
        'ENSG00000137101_CD72', 'ENSG00000019582_CD74',
        'ENSG00000105369_CD79A', 'ENSG00000085117_CD82',
        'ENSG00000114013_CD86', 'ENSG00000010278_CD9',
        'ENSG00000002586_CD99', 'ENSG00000166091_CMTM5',
        'ENSG00000119865_CNRIP1', 'ENSG00000100368_CSF2RB',
        'ENSG00000100448_CTSG', 'ENSG00000051523_CYBA',
        'ENSG00000116675_DNAJC6', 'ENSG00000142227_EMP3',
        'ENSG00000143226_FCGR2A', 'ENSG00000167996_FTH1',
        'ENSG00000139278_GLIPR1', 'ENSG00000130755_GMFG',
        'ENSG00000169567_HINT1', 'ENSG00000206503_HLA-A',
        'ENSG00000234745_HLA-B', 'ENSG00000204287_HLA-DRA',
        'ENSG00000196126_HLA-DRB1', 'ENSG00000204592_HLA-E',
        'ENSG00000171476_HOPX', 'ENSG00000076662_ICAM3',
        'ENSG00000163565_IFI16', 'ENSG00000142089_IFITM3',
        'ENSG00000160593_JAML', 'ENSG00000055118_KCNH2',
        'ENSG00000105610_KLF1', 'ENSG00000139187_KLRG1',
        'ENSG00000133816_MICAL2', 'ENSG00000198938_MT-CO3',
        'ENSG00000107130_NCS1', 'ENSG00000090470_PDCD7',
        'ENSG00000143627_PKLR', 'ENSG00000109099_PMP22',
        'ENSG00000117450_PRDX1', 'ENSG00000112077_RHAG',
        'ENSG00000108107_RPL28', 'ENSG00000198918_RPL39',
        'ENSG00000145425_RPS3A', 'ENSG00000198034_RPS4X',
        'ENSG00000196154_S100A4', 'ENSG00000197956_S100A6',
        'ENSG00000188404_SELL', 'ENSG00000124570_SERPINB6',
        'ENSG00000235169_SMIM1', 'ENSG00000095932_SMIM24',
        'ENSG00000137642_SORL1', 'ENSG00000128040_SPINK2',
        'ENSG00000072274_TFRC', 'ENSG00000205542_TMSB4X',
        'ENSG00000133112_TPT1', 'ENSG00000026025_VIM']

new_columns = []
for col in X_test_shap.columns:
    if col.startswith('imp_'):
        col = gene_ids[int(col.split('_')[1])]
    new_columns.append(col)
X_test_shap.columns = new_columns
print(X_test_shap.shape)
X_test_shap.head()

# +
# X_train for model #16: 'X_svd_128.pickle'
X_train = pd.read_pickle(cite_feature_path  + 'X_svd_128.pickle')
X_train = np.array(X_train)
print('X_train: ', X_train.shape)
print('X_test: ', X_test.shape)

explainer = shap.KernelExplainer(model, shap.sample(X_train, 1000))
explainer
# -

private_test_input.shape

private_test_target.shape

xtest = X_test_shap#.drop(['cell_id', 'cell_type'], axis=1)

print(X_test_shap.shape)
X_test_shap.head()

with open('X_test_shap_16.pkl', 'wb') as f:
    pickle.dump(X_test_shap, f)

# +
# features: genes and svd -> omnipath: genes
# model: mostly relying on genes or svd? -> later

# +
# don't need to run again: np.load('shap_values.npy', allow_pickle=True)
# # %timeit
# shap_values = explainer.shap_values(xtest, nsamples=300)  #500? 
# print(len(shap_values)) # -> 140 genes
# print(len(shap_values[0])) # -> number of samples in xtest
# print(shap_values[0].shape)

# np.save('shap_values_16.npy', np.array(shap_values, dtype=object), allow_pickle=True)
# -

shap_values = np.load('shap_values_16.npy', allow_pickle=True).astype(float)

shap_values[0]

# ### plot shap values per cell type similar to shap.summary_plot(shap_values[0], xtest)

# shap_values that are plotted in beeswarm below
# [0,:,2] == base_svd_2; [0,:,1] == base_svd_1 etc
second_element = shap_values[0, :, 9]
print(second_element)
print(min(second_element))
max(second_element)

# print top 10 features plotted in beeswarm plot below
shap_sum = np.abs(shap_values[0]).sum(axis=0)
top_features_indices = np.argsort(shap_sum)[::-1][:10]  # Get the indices of the top 10 features
top_feature_names_shap = xtest.drop(['cell_id', 'cell_type'], axis=1).columns[top_features_indices]
top_feature_names_shap

shap_cell_types = pd.DataFrame({'SHAP svd_2': shap_values[0, :, 2], 
                                'SHAP svd_9': shap_values[0, :, 9], 
                                'SHAP svd_1': shap_values[0, :, 1], 
                                'SHAP svd_16': shap_values[0, :, 16], 
                                'SHAP svd_11': shap_values[0, :, 11], 
                                'SHAP svd_7': shap_values[0, :, 7], 
                                'SHAP svd_4': shap_values[0, :, 4], 
                                'SHAP svd_6': shap_values[0, :, 6], 
                                'SHAP CD36': shap_values[0, :, xtest.columns.get_loc('ENSG00000135218_CD36')+1], # check +1
                                'SHAP svd_21': shap_values[0, :, 21], 
                                'Cell Type': xtest['cell_type']})
print(shap_cell_types.shape)
shap_cell_types.head()


# +
# Assign different colors to each class
colors = {'BP': 'red', 'EryP': 'blue', 'HSC': 'green', 'MasP': 'orange', 'MkP': 'purple', 'MoP': 'yellow', 'NeuP': 'pink'}

## legend ##
legend_fig, legend_ax = plt.subplots(figsize=(2, 2))

for class_label, color in colors.items():
    legend_ax.scatter([], [], color=color, label=f'Cell type {class_label}')

legend_ax.legend(loc='center', bbox_to_anchor=(0.5, 1.2), ncol=len(colors), frameon=False)
legend_ax.axis('off')
## legend ##


# plot shap values:
fig, ax = plt.subplots(figsize=(8, 10))

x_limit = (-0.2, 0.5)

# Remove y-axis ticks and labels
ax.yaxis.set_visible(False)

# Iterate over the columns in shap_cell_types to create subplots
for i, column in enumerate(shap_cell_types.columns[:-1], start=1):

    ax = fig.add_subplot(len(shap_cell_types.columns)-1, 1, i)

    # Set the x-axis limits and label
    ax.set_xlim(x_limit)
    ax.set_xlabel(column)

    ax.yaxis.set_visible(False)

    # Plot the dots for the current column
    for index, row in shap_cell_types.iterrows():
        shap_value = row[column]
        class_label = row['Cell Type']
        color = colors[class_label]
        ax.plot(shap_value, 0, marker='o', color=color)

        
fig.tight_layout(rect=[0, 0.1, 1, 1]) 
legend_fig.subplots_adjust(top=0.1, bottom=0.05)

plt.show()

# +
shap.initjs()
shap.summary_plot(shap_values[0], xtest.drop(['cell_id', 'cell_type'], axis=1), feature_names=xtest.drop(['cell_id', 'cell_type'], axis=1).columns)

# goal: get this plot as SHAP / features instead of SHAP / SVD components + features

# -


shap.initjs()
shap.summary_plot(list(shap_values), plot_type = 'bar', feature_names = xtest.drop(['cell_id', 'cell_type'], axis=1).columns)
# 140 classes = each regression output

# ### svd contributions:

svd_comp_norm = np.loadtxt('/dss/dsshome1/02/di93zoj/valentina/open-problems-multimodal-3rd-solution/code/2.preprocess_to_feature/cite/svd_comp_norm.txt', delimiter=',')
print(pd.DataFrame(svd_comp_norm).shape)
pd.DataFrame(svd_comp_norm).head()

# base_svd_2 important feature
svd_comp_norm[2]         # => contribution x_2 = -0.00001809 * geneA - 0.00000149 * geneB + 0.0000030917 * geneC + ... + 0.0001264 geneX + 0.00026577 geneY + 0.000175 geneZ

pd.DataFrame(shap_values[0]).head()

# +
# multiply SHAP(svd_n)*contribution of gene A to component n -> sum
# each dot in summary_plot is attribution for one cell -> loop over all cells

# 212 features: 128 svd and 84 genes
# cells: 35
# predicted "classes": 140

# contribution of gene A to component n: svd_comp_norm
# SHAP(svd_n) for the 128 svd (=first 128 columns)


# TODO delete following at some point, should be represented in next cell as 3D
# attr_genes_only = np.zeros((len(xtest), 22001))  # Initialize the output array, 35x22001

# for cell in range(len(xtest)):
#     for gene in range(22001):
#         attr_gene = 0
#         for svd in range(128):
#             attr_gene += shap_values[0][cell][svd] * svd_comp_norm[svd][gene]   # this is for one cell -> loop over all cells
#         attr_genes_only[cell][gene] = attr_gene
# attr_genes_only

# +
# add third dimension -> 140x35x22001
attr_genes_only = np.zeros((shap_values.shape[0], len(xtest), 22001))  # Initialize the output array, 140x35x22001

for pred in range(shap_values.shape[0]):
    for cell in range(len(xtest)):
        attr_genes_only[pred, cell] = np.sum(shap_values[pred, cell, :128, None] * svd_comp_norm[:128], axis=0)

print(attr_genes_only.shape)
attr_genes_only[0]
# -

# load gene names of 22001 genes (excl. 84 handselected genes)
# gene_ids contains gene names of 84 handselected genes
all_genes = np.loadtxt('/dss/dsshome1/02/di93zoj/valentina/open-problems-multimodal-3rd-solution/code/2.preprocess_to_feature/cite/all_genes_names.txt', dtype=str)
all_genes   # len 22001   (excl. the 84 other genes)   -> need list(all_genes) + gene_ids

# create attr_all_22085_genes: pass gene and cell names -> for shap plot, "new shap values" with backpropagated svd weights
attr_all_22085_genes = np.hstack((attr_genes_only[0], shap_values[0][:,-84:]))
print(pd.DataFrame(attr_all_22085_genes, columns=list(all_genes)+gene_ids).shape)
pd.DataFrame(attr_all_22085_genes, columns=list(all_genes)+gene_ids).head()

# +
test_inputs = scipy.sparse.load_npz(index_path + "test_cite_raw_inputs_values.sparse.npz")
test_inputs = pd.DataFrame(test_inputs.toarray(), columns=list(all_genes)+gene_ids)

# get cell ids
test_ids = np.load(index_path + "test_cite_raw_inputs_idxcol.npz", allow_pickle=True)
test_index = test_ids["index"]
len(test_index)

test_inputs.index = test_index

test_inputs
# -

# cell ids used in xtest and SHAP:
sample_cells = np.array(xtest['cell_id'])
sample_cells

# create xtest_all_genes: for shap plot, containing all 22085 genes instead of svd components
xtest_all_genes = test_inputs.loc[sample_cells]
print(xtest_all_genes.shape)
xtest_all_genes.head()

shap.initjs()
shap.summary_plot(attr_all_22085_genes, xtest_all_genes, feature_names=xtest_all_genes.columns)

# top 20 genes:
shap_sum = np.abs(attr_all_22085_genes).sum(axis=0)
top_features_indices = np.argsort(shap_sum)[::-1][:20]  # Get the indices of the top 20 features
top_feature_names_shap = xtest_all_genes.columns[top_features_indices]
top_feature_names_shap

# This cell shows that the top 20 attributing genes are from both the handselected and the other genes with similar distribution
for gene in top_feature_names_shap:
    if gene in list(all_genes):
        print(gene, 'among 22001 genes.')
    if gene in gene_ids:
        print(gene, 'among handselected 84 genes.')

# ### analyze gene attribution distributions

# +
abs_mean_attributions = np.mean(np.abs(attr_all_22085_genes), axis=0)

distribution = np.histogram(abs_mean_attributions, bins=10)

print("Distribution of absolute values of mean attributions per gene for all genes:")
for value, count in zip(distribution[1], distribution[0]):
    print("Range: {:.5f} - {:.5f}, Count: {}".format(value, value + distribution[1][1], count))

# +
abs_mean_attributions_others = np.mean(np.abs(attr_all_22085_genes[:, :22001]), axis=0)

# log_bins = np.logspace(np.log2(abs_mean_attributions.min() + 1e-9), np.log2(abs_mean_attributions.max()), 10)  # for plot 

distribution_others = np.histogram(abs_mean_attributions_others, bins=distribution[1])  # bins=log_bins  # 10

print("Distribution of absolute values of mean attributions per column of 22001 genes:")
for value, count in zip(distribution_others[1], distribution_others[0]):
    print("Range: {:.5f} - {:.5f}, Count: {}".format(value, value + distribution_others[1][1], count))

# get top attributing genes
genes_higher_attr_others_idx = np.where(abs_mean_attributions_others >= 0.00766)[0]
print(genes_higher_attr_others_idx)
genes_higher_attr_others = all_genes[genes_higher_attr_others_idx]
genes_higher_attr_others

# +
abs_mean_attributions_handselected = np.mean(np.abs(attr_all_22085_genes[:, -84:]), axis=0)

# log_bins = np.logspace(np.log2(abs_mean_attributions.min() + 1e-9), np.log2(abs_mean_attributions.max()), 10)

distribution_handselected = np.histogram(abs_mean_attributions_handselected, bins=10)  # bins=log_bins

print("Distribution of absolute values of mean attributions per handselected gene:")
for value, count in zip(distribution_handselected[1], distribution_handselected[0]):
    print("Range: {:.5f} - {:.5f}, Count: {}".format(value, value + distribution_handselected[1][1], count))

# get genes with count 1 (top attributing genes)
genes_higher_attr_handselected_idx = np.where(abs_mean_attributions_handselected >= 0.00684)[0]
print(genes_higher_attr_handselected_idx)
genes_higher_attr_handselected = [gene_ids[idx] for idx in genes_higher_attr_handselected_idx]
genes_higher_attr_handselected
# -

print(abs_mean_attributions_handselected.tolist().index(0))
gene_ids[14]

# +
values = np.sort(abs_mean_attributions_handselected)
ranks = np.arange(1, len(values) + 1)

plt.plot(ranks, values, marker='o', linestyle='-', color='blue')
plt.xlabel('Rank')
plt.ylabel('Value')
plt.title('Handselected genes')
plt.grid(True)
plt.show()

# +
values = np.sort(abs_mean_attributions_others)
ranks = np.arange(1, len(values) + 1)

plt.plot(ranks, values, marker='o', linestyle='-', color='blue')
plt.xlabel('Rank')
plt.ylabel('Value')
plt.title('Other genes')
plt.grid(True)
plt.show()

# +
# TODO plot these in one plot: seaborn scatterplot hue

# +
# get abs_mean_attribution == 0
#print(np.sort(abs_mean_attributions_handselected))
#print(np.sort(abs_mean_attributions_others)[:500])

# +
for i in genes_higher_attr_others:
    print(i in top_feature_names_shap)

for i in genes_higher_attr_handselected:
    print(i in top_feature_names_shap)


# +
# TODO add label of gini index per plot
def gini(arr):    # range from 0 (total equality) to 1 (absolute inequality)
    ## first sort
    sorted_arr = arr.copy()
    sorted_arr.sort()
    n = arr.size
    coef_ = 2. / n
    const_ = (n + 1.) / n
    weighted_sum = sum([(i+1)*yi for i, yi in enumerate(sorted_arr)])
    return coef_*weighted_sum/(sorted_arr.sum()) - const_

print(gini(abs_mean_attributions_handselected))
print(gini(abs_mean_attributions_others))


# TODO put plots into one with same y-axis scale

# +
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.hist(abs_mean_attributions_handselected, bins=distribution[1], alpha=0.5)  # bins=distribution[1]  # bins=log_bins
# ax1.set_xscale('log')
ax1.text(0.5, 0.95, f'Gini Index: {gini(abs_mean_attributions_handselected):.3f}', transform=ax1.transAxes, ha='center', color='green', fontsize=12)

ax1.set_xlabel('Absolute Values of Mean Attributions')
ax1.set_ylabel('Count')
ax1.set_title('handselected genes attribution distribution')

ax2.hist(abs_mean_attributions_others, bins=distribution[1], alpha=0.5)
# ax2.set_xscale('log')
ax2.text(0.5, 0.95, f'Gini Index: {gini(abs_mean_attributions_others):.3f}', transform=ax2.transAxes, ha='center', color='green', fontsize=12)

ax2.set_xlabel('Absolute Values of Mean Attributions')
ax2.set_ylabel('Count')
ax2.set_title('other genes attribution distribution')

plt.tight_layout()
plt.show()

# +
fig, ax = plt.subplots(figsize=(12, 6))

ax.hist(abs_mean_attributions_handselected, bins=distribution[1], alpha=0.5, label='Handselected')
ax.set_xlabel('Absolute Values of Mean Attributions')
ax.set_ylabel('Count')
ax.set_title('Comparison of attribution distribution of handselected and the other genes')

# Create a twin axis for the second histogram
ax2 = ax.twinx()

ax2.hist(abs_mean_attributions_others, bins=distribution[1], alpha=0.5, color='orange', label='Others')
ax2.set_ylabel('Count')

# Adjust the y-axis limits of the twin Axes to match the primary Axes
# ax.set_ylim(ax2.get_ylim())
print(ax2.get_ylim())
ax.set_ylim((0,55))
ax2.set_ylim((0,800))

# Add legend
handles, labels = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax.legend(handles + handles2, labels + labels2, loc='upper right')

# Add Gini index labels
ax.text(0.5, 0.95, f'Gini Index (Handselected): {gini(abs_mean_attributions_handselected):.3f}', transform=ax.transAxes, ha='center', color='blue', fontsize=12)
ax2.text(0.5, 0.90, f'Gini Index (Others): {gini(abs_mean_attributions_others):.3f}', transform=ax2.transAxes, ha='center', color='orange', fontsize=12)

# Adjust the layout and display the plot
plt.tight_layout()
plt.show()


# +
# TODO same density for both plots (same area of histograms) -> adjusts y-axes

# +
# analyze svd components distr.

mean_svd_contr = np.mean(np.abs(svd_comp_norm), axis=0)
top_20_indices = np.argsort(mean_svd_contr)[-20:][::-1]

top_20_mean_contributions = mean_svd_contr[top_20_indices]
top_features_svd = []

# Print the top 20 columns and their mean contributionsb
print("Top 20 Columns with Highest Mean Contributions to SVDs:")
for column_index, mean_contribution in zip(top_20_indices, top_20_mean_contributions):
    print("Column: {}, Mean Contribution: {}".format(all_genes[column_index], mean_contribution))
    top_features_svd.append(all_genes[column_index])
# top_features_svd
# -


# compare resulting genes
for i in top_features_svd:
    if i in top_feature_names_shap:
        print(i, 'among top 20 shap features')
    
    elif i in genes_higher_attr_handselected:
        print(i, 'among handselected genes with highest attributions')
    elif i in genes_higher_attr_others:
        print(i, 'among other genes with highest attributions')
    else:
        print(i)

# +
# 50 per cell type (after test run below) 1
# private_test_target & private_test_input 0
# feed test set to shap instead of 35 samples -> sample x pr cell type from test set
# target test set: double check predictions above

# +
# scanpy clustering 3
# 140 35 212
# per cell type avg across 35 rows -> 140x212 matrix
# -> cluster 140 rows -> 
# clusters of proteins with similar attributions
# -

# ### shap model #16 for private test data

model

# +
#X_train_p = pickle.load(open('/dss/dssfs02/lwp-dss-0001/pn36po/pn36po-dss-0001/di93zoj/kaggle/full_data/20220830_citeseq_rna_count_train.pkl', 'rb'))
#X_train_p = np.array(X_train_p)

X_test_p = ad.read_h5ad('private_test_input_sample.h5ad')

# print('X_train: ', X_train_p.shape)
print('X_test: ', X_test_p.X.shape)

## explainer = shap.KernelExplainer(model, shap.sample(X_train_p, 1000))
## explainer  # use same explainer trained above (?)

# shap_values = explainer.shap_values(X_test_p.X, nsamples=300)

# +
# np.save('shap_values_16_p.npy', np.array(shap_values, dtype=object), allow_pickle=True)
# -

shap_values_16_p = np.load('shap_values_16_p.npy', allow_pickle=True).astype('float')

X_test_p.var_names = new_columns[1:213]   # new_column already implemented for shap plots above

shap.initjs()
shap.summary_plot(shap_values_16_p[0], X_test_p.X, feature_names=X_test_p.var_names)


# #### backpropagate svd components

# +
attr_genes_only = np.zeros((shap_values_16_p.shape[0], len(X_test_p.X), 22001))  # Initialize the output array, 140x35x22001

for pred in range(shap_values_16_p.shape[0]):
    for cell in range(len(xtest)):
        attr_genes_only[pred, cell] = np.sum(shap_values_16_p[pred, cell, :128, None] * svd_comp_norm[:128], axis=0)

print(attr_genes_only.shape)
attr_genes_only[0]
# -

# create attr_all_22085_genes: pass gene and cell names -> for shap plot, "new shap values" with backpropagated svd weights
attr_all_22085_genes = np.hstack((attr_genes_only[0], shap_values_16_p[0][:,-84:]))
print(pd.DataFrame(attr_all_22085_genes, columns=list(all_genes)+gene_ids).shape)
pd.DataFrame(attr_all_22085_genes, columns=list(all_genes)+gene_ids).head()

# TODO private_test_input instead?
private_test_input_raw = pickle.load(open('/dss/dssfs02/lwp-dss-0001/pn36po/pn36po-dss-0001/di93zoj/kaggle/full_data/20220830_citeseq_rna_count_test_input_private_raw.pkl', 'rb'))
private_test_input_raw.head()

# cell ids used in X_test_p and SHAP:
sample_cells = np.array(X_test_p.obs_names)
sample_cells[:5]

# create xtest_all_genes: for shap plot, containing all 22085 genes instead of svd components
xtest_all_genes = private_test_input_raw.loc[sample_cells]
print(xtest_all_genes.shape)
xtest_all_genes.head()

shap.initjs()
shap.summary_plot(attr_all_22085_genes, xtest_all_genes, feature_names=xtest_all_genes.columns)
