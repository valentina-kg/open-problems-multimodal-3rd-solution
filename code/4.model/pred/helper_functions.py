# +
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import rc_context
import pandas as pd
import os
import random

import shap

import anndata as ad
import scipy
import scanpy as sc

from scipy import stats
# -



lrz_path = '/dss/dssfs02/lwp-dss-0001/pn36po/pn36po-dss-0001/di93zoj/'
index_path = lrz_path + 'open-problems-multimodal-3rd-solution/input/preprocess/cite/'

os.chdir('../..')

# ### load data

# all 22085 genes sorted totally alphabetically
train_column = np.load(index_path + "train_cite_raw_inputs_idxcol.npz", allow_pickle=True)["columns"]
train_column

xtest_16 = ad.read_h5ad('4.model/pred/private_test_input_128_svd_50_samples.h5ad')
private_test_input_raw = pd.read_pickle('/dss/dssfs02/lwp-dss-0001/pn36po/pn36po-dss-0001/di93zoj/kaggle/full_data/20220830_citeseq_rna_count_test_input_private_raw.pkl')
sample_cells = np.array(xtest_16.obs_names)
xtest_all_genes = private_test_input_raw.loc[sample_cells]

attr_all_22085_genes_16 = pd.read_pickle(lrz_path + 'large_preprocessed_files/attr_all_genes/attr_all_22085_genes_16_50_samples_p_ct_distr.pkl')
for i in range(len(attr_all_22085_genes_16)):
    attr_all_22085_genes_16[i] = attr_all_22085_genes_16[i].reindex(columns=train_column)

xtest_17 = ad.read_h5ad('4.model/pred/private_test_input_64_svd_50_samples.h5ad')
attr_all_22085_genes_17 = pd.read_pickle(lrz_path + 'large_preprocessed_files/attr_all_genes/attr_all_22085_genes_17_50_samples_p_ct_distr.pkl')
for i in range(len(attr_all_22085_genes_17)):
    attr_all_22085_genes_17[i] = attr_all_22085_genes_17[i].reindex(columns=train_column)


# ### general helper functions

def get_gene_name(last_chars):
    return [element for element in train_column if element.split('_')[1] == last_chars]


# top top_n features ordered by mean absolute shap_values
def get_top_features(shap_values, xtest, top_n):
    shap_sum = np.abs(shap_values).sum(axis=0)
    top_features_indices = np.argsort(shap_sum)[::-1][:top_n]  # Get the indices of the top n features
    top_feature_names_shap = xtest.var_names[top_features_indices]
    return top_feature_names_shap


# get elements that appear in at least min_percentage of the top features of all proteins
def get_common_elements(attr, xtest, top_features_considered, min_percentage):
    element_counts = {}

    for i in range(140):
        for elt in get_top_features(attr[i], xtest, top_features_considered):
            element_counts[elt] = element_counts.get(elt, 0) + 1

    # Get the elements that appear in at least min_percentage of the lists
    common_elements = {element for element, count in element_counts.items() if count / 140 >= min_percentage}
    return common_elements


proteins = np.load('4.model/pred/proteins.npy', allow_pickle=True)
def get_protein_idx(protein):
    return np.where(proteins == protein)[0][0]


# ### functions for plotting.ipynb

# +
# function for plotting shap values
def shap_beeswarm(shap_values, X_test, protein_idx=0):
    shap.initjs()
    shap.summary_plot(shap_values[protein_idx], X_test.to_df(), feature_names=X_test.var_names)

def shap_bar_plot(shap_values, X_test):
    shap.initjs()
    shap.summary_plot(list(shap_values), plot_type = 'bar', feature_names = X_test.var_names)
    # 140 classes = each regression output


# -

# function for plotting backpropagated attribution values (new function because of other column order)
def attr_beeswarm(attr_values, X_test, protein_idx):
    # change column order to have all columns sorted and in same order as X_test
    attr_values[protein_idx] = attr_values[protein_idx].reindex(columns=train_column)
    shap.initjs()
    shap.summary_plot(np.array(attr_values[protein_idx]), X_test, feature_names=[gene.split('_')[1] for gene in X_test.columns])


# +
# function for plot: plot top 10 features colouring datapoints by cell type
def get_plot_per_cell_type(shap_values, xtest, protein_idx=0, show=True, save=False, filename=None, all_attr=False):   # all_attr=True if attr_all_genes passed instead of svd shap values
    
    if all_attr:
        # in this case: have 22085 columns, not totally sorted but first 22001 columns sorted and next 84 columns sorted
        # => need total order (=train_column) to match xtest
        shap_values[protein_idx] = shap_values[protein_idx].reindex(columns=train_column)

    # get top 10 features (see shap beeswarm plot)
    top_feature_names_shap = get_top_features(shap_values[protein_idx], xtest, 10)
    
    # create df with shap values of top 10 features + cell type    
    shap_cell_types = {}
    for feature in top_feature_names_shap:
#         column_name = 'SHAP ' + feature.split('_')[-1]
        column_name = feature
        # column_values = shap_values[0, :, xtest.var_names.get_loc(feature)]   # use this if shap_values 3D array
        column_values = np.array(shap_values[protein_idx])[:, xtest.var_names.get_loc(feature)]   # use this if shap_values (or attr_all_22085...) dict of dataframes
        shap_cell_types[column_name] = column_values

    # Add 'Cell Type' column
    shap_cell_types['Cell Type'] = xtest.obs['cell_type']

    # Create DataFrame
    shap_cell_types = pd.DataFrame(shap_cell_types)

    # Get the mean per cell type using the mean_per_cell_type function
    mean_per_cell_type_df = mean_per_cell_type(shap_values[protein_idx], xtest, 10)
    # dict to map from cell type to index in mean_per_cell_type_df
    ct_idx_map = {'BP': 0, 'EryP': 1, 'HSC': 2, 'MasP': 3, 'MkP': 4, 'MoP': 5, 'NeuP': 6}
    
    # code for actual plot:

    # Assign different colors to each class
    # colourblind palette: ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
    colors = {'BP': '#a65628', 'EryP': '#e41a1c', 'HSC': '#4daf4a', 'MasP': '#f781bf', 
              'MkP': '#377eb8', 'MoP': '#984ea3', 'NeuP': '#ff7f00'}
    # plot shap values:
    fig, ax = plt.subplots(figsize=(8, 14))
    ax.grid(False)
    
    # x_limit = (shap_values[protein_idx].min()-0.05, shap_values[protein_idx].max()+0.05)  # this is min and max for shap_values[protein_idx]
    x_limit = (shap_values[protein_idx].min().min()-0.05, shap_values[protein_idx].max().max()+0.05)  # above if array, this row if dict of dataframes
    
    # Remove y-axis ticks and labels
    ax.yaxis.set_visible(False)

    class_offsets = {}   # add vertical offset per class
    y = 0
    # Iterate over the columns in shap_cell_types to create subplots
    for i, column in enumerate(shap_cell_types.columns[:-1], start=1):
        ax = fig.add_subplot(len(shap_cell_types.columns)-1, 1, i)

        # Set the x-axis limits and label
        ax.set_xlim(x_limit)
        ax.set_xlabel(column.split('_')[1]) #, fontsize=11)
        ax.grid(False)
        ax.yaxis.set_visible(False)
        
        # Get the mean value for the current column from mean_per_cell_type_df
        mean_value = mean_per_cell_type_df[column].values[:-1]  # [:-1] to exclude last row (Overall)

        # Plot the dots for the current column
        for index, row in shap_cell_types.iterrows():
            shap_value = row[column]
            class_label = row['Cell Type']
            color = colors[class_label]
            
            ##### add vertical offset per class ###
            # Check if the class_label is already in the class_offsets dictionary
            if class_label not in class_offsets:
                class_offsets[class_label] = y

            # Add the vertical offset to the y-coordinate of the dot
            y_offset = class_offsets[class_label]
            ##### add vertical offset per class ###
            
            # ax.plot(shap_value, 0, marker='o', color=color, markersize=2)  # without offset
            ax.plot(shap_value, y_offset, marker='o', color=color, markersize=2)

            # Increment the vertical offset for the next class_label
            y += 1
            
            # add vertical line showing mean per cell type
            ax.axvline(mean_value[ct_idx_map[class_label]], color=color, linestyle='solid', linewidth=2, alpha=0.8)
            
    # legend #
    legend_ax = fig.add_axes([0.45, 0.95, 0.1, 0.15])  # adjust the position and size of the legend axes

    for class_label, color in colors.items():
        legend_ax.scatter([], [], color=color, label=f'Cell type {class_label}')

    legend_ax.legend(loc='center', bbox_to_anchor=(0.5, 0.5), ncol=len(colors), frameon=False)
    legend_ax.axis('off')
    # legend #
    fig.tight_layout(rect=[0, 0.1, 0.85, 1])
            
#     fig.tight_layout(rect=[0, 0.1, 1, 1]) 
#     legend_fig.subplots_adjust(top=0.1, bottom=0.05)
    
    # save plot if save=True -> need to have filename as param
    if save:
        if filename is None:
            raise ValueError("A filename must be provided when save=True.")
        plt.savefig(f'4.model/pred/plots/{filename}.png')

    if show:
        plt.show()
    else:
        plt.close()



# +
# get cell type plot for a specific gene and compare across models

def get_ct_plot_compare(attr_values_16, attr_values_17, attr_values_ensemble, xtest, protein_idx, gene):
    attr_16 = attr_values_16[protein_idx][get_gene_name(gene)].rename(columns={get_gene_name(gene)[0]: 'Model 16'})
    attr_17 = attr_values_17[protein_idx][get_gene_name(gene)].rename(columns={get_gene_name(gene)[0]: 'Model 17'})
    attr_ensemble = attr_values_ensemble[protein_idx][get_gene_name(gene)].rename(columns={get_gene_name(gene)[0]: 'Ensemble'})
    attr_df = pd.concat([attr_16, attr_17, attr_ensemble], axis=1)
    attr_df['(buffer)'] = 0
    attr_df['Cell Type'] = xtest.obs['cell_type'].reset_index(drop=True)

    # Get the mean per cell type using the mean_per_cell_type function
    mean_per_cell_type_df = attr_df.groupby('Cell Type').apply(lambda x: x.mean())
    # dict to map from cell type to index in mean_per_cell_type_df
    ct_idx_map = {'BP': 0, 'EryP': 1, 'HSC': 2, 'MasP': 3, 'MkP': 4, 'MoP': 5, 'NeuP': 6}
    
    # code for actual plot:

    # Assign different colors to each class
    # colourblind palette: ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
    colors = {'BP': '#a65628', 'EryP': '#e41a1c', 'HSC': '#4daf4a', 'MasP': '#f781bf', 
              'MkP': '#377eb8', 'MoP': '#984ea3', 'NeuP': '#ff7f00'}
    # plot shap values:
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.grid(False)
    x_limit = (attr_df.drop('Cell Type', axis=1).min().min()-0.05, attr_df.drop('Cell Type', axis=1).max().max()+0.05)  # above if array, this row if dict of dataframes
    
    # Remove y-axis ticks and labels
    ax.yaxis.set_visible(False)

    class_offsets = {}   # add vertical offset per class
    y = 0
    # Iterate over the columns in shap_cell_types to create subplots
    for i, column in enumerate(attr_df.columns[:-1], start=1):
        ax = fig.add_subplot(len(attr_df.columns)-1, 1, i)

        # Set the x-axis limits and label
        ax.set_xlim(x_limit)
        ax.set_xlabel(column)
        ax.grid(False)
        ax.yaxis.set_visible(False)
        
        # Get the mean value for the current column from mean_per_cell_type_df
        mean_value = mean_per_cell_type_df[column]

        # Plot the dots for the current column
        for index, row in attr_df.iterrows():
            shap_value = row[column]
            class_label = row['Cell Type']
            color = colors[class_label]
            
            ##### add vertical offset per class ###
            # Check if the class_label is already in the class_offsets dictionary
            if class_label not in class_offsets:
                class_offsets[class_label] = y

            # Add the vertical offset to the y-coordinate of the dot
            y_offset = class_offsets[class_label]
            ##### add vertical offset per class ###
            
            # ax.plot(shap_value, 0, marker='o', color=color, markersize=2)  # without offset
            ax.plot(shap_value, y_offset, marker='o', color=color, markersize=2)
            
            # Increment the vertical offset for the next class_label
            y += 1
            
            # add vertical line showing mean per cell type
            ax.axvline(mean_value[ct_idx_map[class_label]], color=color, linestyle='solid', linewidth=2, alpha=0.8)
            
    # legend #
    legend_ax = fig.add_axes([0.45, 0.95, 0.1, 0.15])  # adjust the position and size of the legend axes

    for class_label, color in colors.items():
        legend_ax.scatter([], [], color=color, label=f'Cell type {class_label}')

    legend_ax.legend(loc='center', bbox_to_anchor=(0.5, 0.5), ncol=len(colors), frameon=False)
    legend_ax.axis('off')
    # legend #
    fig.tight_layout(rect=[0, 0.5, 0.85, 1])
    
    plt.show()


# -

def mean_per_cell_type(attr_all_genes, xtest, n_top):
    top_feature_names_shap = get_top_features(attr_all_genes, xtest, n_top)
    # create df with shap values of top n features + cell type    
    shap_cell_types = {}
    # Add 'Cell Type' column
    shap_cell_types['Cell Type'] = xtest.obs['cell_type']
    # Add attribution values of top n features
    for feature in top_feature_names_shap:
        column_values = np.array(attr_all_genes)[:, xtest.var_names.get_loc(feature)]
        shap_cell_types[feature] = column_values

    # Create DataFrame
    shap_cell_types = pd.DataFrame(shap_cell_types)
    
    mean_per_ct = shap_cell_types.groupby('Cell Type').apply(lambda x: x.mean())  # x.abs().mean()
    mean_per_ct.reset_index(inplace=True)
    # Calculate the mean across all rows (excluding the 'cell type' column)
    overall_mean = shap_cell_types.drop('Cell Type', axis=1).mean()    # .abs().mean()
    # Add the overall mean as a new row to the DataFrame
    mean_per_ct.loc[len(mean_per_ct)] = ['Overall'] + overall_mean.tolist()
    return mean_per_ct


# ### functions for ranking.ipynb

# +
# sample 20 features 

def sample_features_ranking(attr, protein, gene, top_or_random):
    '''function for sampling the features to be ranked using KL divergence
    top_or_random: 'top' for top20 + gene, 'random' for random 20 + gene
    '''
    df = attr.copy()
   
    # case 1: use top
    if top_or_random == 'top':
        top = get_top_features(df, ad.AnnData(xtest_all_genes, dtype=float), 20)
        if get_gene_name(gene)[0] not in top:  # if specific gene not among the top features but should still be plotted
            top = top.append(pd.Index(get_gene_name(gene)))
        return df[top]
    
    # case 2: use random
    else:
        random_columns = random.sample(df.columns.tolist(), 100)  # 20
        if get_gene_name(gene)[0] not in random_columns:
            random_columns = random_columns + get_gene_name(gene)
        return df[random_columns]


# -

def plot_distr_avg_gene(model_number, protein, gene, show=True):
    ''' plot avg distribution of top columns along with distribution of a specific gene for comparison
    model_number: 16 or 17
    show=True: show plot for specific gene
    show=False: don't show plot, instead print kl divergence for top 20 genes '''
    
    if model_number == 16:
        attr = attr_all_22085_genes_16[get_protein_idx(protein)]
    elif model_number == 17:
        attr = attr_all_22085_genes_17[get_protein_idx(protein)]
    
    # sample features to plot
    df = sample_features_ranking(attr, protein, gene, 'random')  # (top/)random
    
    # Initialize an array to store the sum of histograms
    hist_sum = np.zeros(40)  # 40 bins
    # Calculate the histogram for each column and sum them
    for col in df.columns:
        hist, edges = np.histogram(df[col], bins=40, range=(attr.min().min(), attr.max().max()), density=True)
        hist_sum += hist

    # Divide by the number of columns to average hist_sum
    avg_hist = hist_sum /len(df.columns)
    print(1)
    
    if show:
        # Plot the average histogram
        plt.figure(figsize=(10, 6))
        plt.bar(edges[:-1], avg_hist, width=(edges[1]-edges[0]), label='Avg')
        
        
        # Plot histogram for single gene to be compared
        hist, edges = np.histogram(df[get_gene_name(gene)], bins=40, range=(attr.min().min(), attr.max().max()), density=True)
        # Plot the histogram
        plt.bar(edges[:-1], hist, width=(edges[1]-edges[0]), color='orange', alpha=0.7, label=gene)
    
        # print kl divergence between both distributions
        # add 1e-10 to avoid dividing by 0
        print(np.sum(scipy.special.kl_div(avg_hist+1e-10, hist+1e-10)))
        
        plt.text(0.3, 0.95, f'KL Divergence: {np.sum(scipy.special.kl_div(avg_hist+1e-10, hist+1e-10)):.3f}', ha='center', color='turquoise', fontsize=12, transform=plt.gca().transAxes)
        
        plt.xlabel('Attribution value')
        plt.ylabel('Normalized frequency')
#         plt.title(f'Prediction of protein {protein}: Average distribution and distribution of gene {get_gene_name(gene)[0].split('_')[1]}')
        plt.title(f"Prediction of protein {protein}: Average attribution distribution and distribution of gene {get_gene_name(gene)[0].split('_')[1]}")
        plt.legend(loc='upper right')
        plt.grid(False)
        plt.show()


def plot_distr_avg_gene_for_slides(model_number, protein, gene, show=True):
    ''' plot avg distribution of top columns along with distribution of a specific gene for comparison
    model_number: 16 or 17
    show=True: show plot for specific gene
    show=False: don't show plot, instead print kl divergence for top 20 genes '''
    
    if model_number == 16:
        attr = attr_all_22085_genes_16[get_protein_idx(protein)]
    elif model_number == 17:
        attr = attr_all_22085_genes_17[get_protein_idx(protein)]
    
    # sample features to plot
    df = sample_features_ranking(attr, protein, gene, 'random')  # (top/)random
    
    # Initialize an array to store the sum of histograms
    hist_sum = np.zeros(40)  # 40 bins
    # Calculate the histogram for each column and sum them
    for col in df.columns:
        hist, edges = np.histogram(df[col], bins=40, range=(attr.min().min(), attr.max().max()), density=True)
        hist_sum += hist

    # Divide by the number of columns to average hist_sum
    avg_hist = hist_sum /len(df.columns)
    print(1)
    
    if show:
        # Plot the average histogram
        plt.figure(figsize=(4, 6))
        plt.bar(edges[:-1], avg_hist, width=(edges[1]-edges[0]), label='Avg')
        
        
        # Plot histogram for single gene to be compared
        hist, edges = np.histogram(df[get_gene_name(gene)], bins=40, range=(attr.min().min(), attr.max().max()), density=True)
        # Plot the histogram
        plt.bar(edges[:-1], hist, width=(edges[1]-edges[0]), color='orange', alpha=0.7, label=gene)
    
        # print kl divergence between both distributions
        # add 1e-10 to avoid dividing by 0
        print(np.sum(scipy.special.kl_div(avg_hist+1e-10, hist+1e-10)))
        
        # plt.text(0.3, 0.95, f'KL Divergence: {np.sum(scipy.special.kl_div(avg_hist+1e-10, hist+1e-10)):.3f}', ha='center', color='turquoise', fontsize=12, transform=plt.gca().transAxes)
        
        plt.xlabel('Attribution value')
        plt.xlim(-0.05, 0.1)
        plt.ylabel('Normalized frequency')
        plt.title(f"Prediction of protein {protein}: Average attribution distribution and distribution of gene {get_gene_name(gene)[0].split('_')[1]}")
        plt.legend(loc='upper right')
        plt.grid(False)
        plt.show()


# +
# 1. random 20
# 2. compute kl of all genes comparaed to random 20
# 3. rank all genes by kl

def ranking(model_number, protein, random_sample_number):
    ranking_dict = {}
    if model_number == 16:
        attr = attr_all_22085_genes_16[get_protein_idx(protein)]
    elif model_number == 17:
        attr = attr_all_22085_genes_17[get_protein_idx(protein)]
    
    range_attr=(attr.min().min(), attr.max().max())
    
    # average distr.:
    # sample 20 random features to compare to
    df = attr.copy()
    random_columns = random.sample(df.columns.tolist(), random_sample_number)
    df = df[random_columns]

    # Initialize an array to store the sum of histograms
    hist_sum = np.zeros(40)  # 40 bins
    # Calculate the histogram for each column and sum them
    for col in df.columns:
        hist, _ = np.histogram(df[col], bins=40, range=range_attr, density=True)
        hist_sum += hist

    # Divide by the number of columns to average hist_sum
    avg_hist = hist_sum /len(df.columns)
    
    for gene in train_column:
        # histogram for single gene to be compared
        hist, _ = np.histogram(attr[gene], bins=40, range=range_attr, density=True)
        
        # append this gene's kl divergence to ranking_dict
        ranking_dict[gene] = np.sum(scipy.special.kl_div(avg_hist+1e-10, hist+1e-10))  # hist+1e-10, avg_hist+1e-10
        # print(gene, ranking_dict[gene])
    
    sorted_dict = dict(sorted(ranking_dict.items(), key=lambda item: item[1], reverse=True))
    top_20_entries = dict(list(sorted_dict.items())[:19])
    
    # force ground truth gene if exists and not yet in top_20_entries
    try:
        if get_gene_name(protein)[0] not in list(top_20_entries.keys()):
            print(f'manually added gene {protein}')
            top_20_entries[get_gene_name(protein)[0]] = ranking_dict[get_gene_name(protein)[0]]
    except Exception as e:
        print(e)
        print('no')
    
    # plot ranking
    plt.figure(figsize=(10, 6))
    plt.barh([gene.split('_')[1] for gene in list(top_20_entries.keys())[::-1]], list(top_20_entries.values())[::-1])
    plt.ylabel('Top genes')
    plt.xlabel('KL divergence')
    plt.title('Top 20 attributing genes by KL divergence')
    plt.grid(False)
    plt.show()
    
    # plot corresponding beeswarm
    shap.initjs()
    shap.summary_plot(np.array(attr[list(top_20_entries.keys())]), 
                      xtest_all_genes[list(top_20_entries.keys())], 
                      feature_names=[gene.split('_')[1] for gene in list(top_20_entries.keys())])
    return sorted_dict


# -

# functions for correlation
def corr2_coeff(A, B):
    """Row-wise pearson correlation.
    """
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    return np.diagonal(np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None])))
def get_corr_df(attr_at_protein):
    corrs = corr2_coeff(attr_at_protein.values.T, xtest_all_genes.values.T)
    corrs_df = pd.DataFrame(corrs, index=xtest_all_genes.columns).sort_values(by=0, ascending=False).dropna()
    return corrs_df


xtest_all_genes


# +
# 1. random 20
# 2. compute kl of all genes compared to random 20
# 3. rank all genes by kl
# 4. filter by correlation

def ranking_corr(model_number, protein, random_sample_number):
    ranking_dict = {}
    if model_number == 16:
        attr = attr_all_22085_genes_16[get_protein_idx(protein)]
    elif model_number == 17:
        attr = attr_all_22085_genes_17[get_protein_idx(protein)]
    
    range_attr=(attr.min().min(), attr.max().max())
    
    # average distr.:
    # sample 20 random features to compare to
    df = attr.copy()
    random_columns = random.sample(df.columns.tolist(), random_sample_number)
    df = df[random_columns]

    print(1)
    
    # Initialize an array to store the sum of histograms
    hist_sum = np.zeros(40)  # 40 bins
    # Calculate the histogram for each column and sum them
    for col in df.columns:
        hist, _ = np.histogram(df[col], bins=40, range=range_attr, density=True)
        hist_sum += hist

    # Divide by the number of columns to average hist_sum
    avg_hist = hist_sum /len(df.columns)
    
    print(2)
    # only keep genes with abs. corr. >= 0.45 -> remove all 0 attributions
    corrs_df = get_corr_df(attr)
    
    for gene in train_column:
        # histogram for single gene to be compared
        hist, _ = np.histogram(attr[gene], bins=40, range=range_attr, density=True)
        
        # append this gene's kl divergence to ranking_dict: IF correlation >= threshold
        if gene in corrs_df.index:
            if np.abs(corrs_df.loc[gene][0]) >= 0.45:
                ranking_dict[gene] = np.sum(scipy.special.kl_div(avg_hist+1e-10, hist+1e-10))  # hist+1e-10, avg_hist+1e-10
        # print(gene, ranking_dict[gene])
    
    sorted_dict = dict(sorted(ranking_dict.items(), key=lambda item: item[1], reverse=True))
    
    print(3)
    # only keep genes with abs. corr. >= 0.45 -> remove all 0 attributions
    # corrs_df = get_corr_df(attr)
    # ranked_filtered_corr = {}
    # for gene in sorted_dict:
    #     if gene in corrs_df.index: 
    #         if np.abs(corrs_df.loc[gene][0]) >= 0.45:    # TODO try other thresholds, try np.abs() >= 0.5
    #             ranked_filtered_corr[gene] = sorted_dict[gene]
    
    # ranked_filtered_corr = dict(sorted(ranked_filtered_corr.items(), key=lambda item: item[1], reverse=True))
    # top_20_entries = dict(list(ranked_filtered_corr.items())[:19])
    top_20_entries = dict(list(sorted_dict.items())[:19])
    
    # force ground truth gene if exists and not yet in top_20_entries
    try:
        if get_gene_name(protein)[0] not in list(top_20_entries.keys()):
            print(f'manually added gene {protein}')
            top_20_entries[get_gene_name(protein)[0]] = ranking_dict[get_gene_name(protein)[0]]
    except Exception as e:
        print(e)
        print('no')
    print(4)
    # plot ranking
    plt.figure(figsize=(10, 6))
    plt.barh([gene.split('_')[1] for gene in list(top_20_entries.keys())[::-1]], list(top_20_entries.values())[::-1])
    plt.ylabel('Top genes')
    plt.xlabel('KL divergence')
    plt.title('Top 20 attributing genes by KL divergence, removed low absolute correlations')
    plt.grid(False)
    plt.show()
    
    # plot corresponding beeswarm
    shap.initjs()
    shap.summary_plot(np.array(attr[list(top_20_entries.keys())]), 
                      xtest_all_genes[list(top_20_entries.keys())], 
                      feature_names=[gene.split('_')[1] for gene in list(top_20_entries.keys())])
    return sorted_dict  # ranked_filtered_corr


# +
from matplotlib.colors import ListedColormap
# 1. random 20
# 2. compute kl of all genes compared to random 20
# 3. rank all genes by kl
# 4. filter by correlation
# 5. colour by mean attribution

def ranking_corr_new(model_number, protein, random_sample_number):
    ranking_dict = {}
    if model_number == 16:
        attr = attr_all_22085_genes_16[get_protein_idx(protein)]
    elif model_number == 17:
        attr = attr_all_22085_genes_17[get_protein_idx(protein)]
    
    range_attr=(attr.min().min(), attr.max().max())
    
    # average distr.:
    # sample 20 random features to compare to
    df = attr.copy()
    random_columns = random.sample(df.columns.tolist(), random_sample_number)
    df = df[random_columns]

    print(1)
    
    # Initialize an array to store the sum of histograms
    hist_sum = np.zeros(40)  # 40 bins
    # Calculate the histogram for each column and sum them
    for col in df.columns:
        hist, _ = np.histogram(df[col], bins=40, range=range_attr, density=True)
        hist_sum += hist

    # Divide by the number of columns to average hist_sum
    avg_hist = hist_sum /len(df.columns)
    
    print(2)
    # only keep genes with abs. corr. >= 0.45 -> remove all 0 attributions
    corrs_df = get_corr_df(attr)
    
    for gene in train_column:
        # histogram for single gene to be compared
        hist, _ = np.histogram(attr[gene], bins=40, range=range_attr, density=True)
        
        # append this gene's kl divergence to ranking_dict: IF correlation >= threshold
        if gene in corrs_df.index:
            if np.abs(corrs_df.loc[gene][0]) >= 0.45:
                ranking_dict[gene] = np.sum(scipy.special.kl_div(avg_hist+1e-10, hist+1e-10))  # hist+1e-10, avg_hist+1e-10
    
    sorted_dict = dict(sorted(ranking_dict.items(), key=lambda item: item[1], reverse=True))
    
    print(3)

    top_20_entries = dict(list(sorted_dict.items())[:19])
    
    # force ground truth gene if exists and not yet in top_20_entries
    try:
        if get_gene_name(protein)[0] not in list(top_20_entries.keys()):
            print(f'manually added gene {protein}')
            top_20_entries[get_gene_name(protein)[0]] = ranking_dict[get_gene_name(protein)[0]]
    except Exception as e:
        print(e)
        print('no')
    print(4)
    
    # plot ranking
    # remove ENSG for bar plot:
    top_20_entries_short = {key.split('_')[1]: value for key, value in top_20_entries.items()}
    
    
    # colour bars by mean absolute value
    # Get the means for the top genes
    top_means = [np.mean(np.abs(attr[gene])) for gene in top_20_entries.keys()]
    # Create a colormap based on the means
    norm = plt.Normalize(min(top_means), max(top_means))
    cmap = plt.cm.viridis
    colours_list = [cmap(norm(mean)) for mean in top_means]
    
    # Plot the bar chart with colors
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = plt.barh(list(top_20_entries_short.keys())[::-1], list(top_20_entries_short.values())[::-1], color=colours_list[::-1])

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([]) # this line may not be necessary depending on your matplotlib version
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Mean Absolute SHAP Value')

    plt.ylabel('Top genes')
    plt.xlabel('KL divergence')
    plt.title('Top 20 attributing genes by KL divergence, removed low absolute correlations')
    plt.grid(False)
    plt.show()
    
    # plot corresponding beeswarm
    shap.initjs()
    shap.summary_plot(np.array(attr[list(top_20_entries.keys())]), 
                      xtest_all_genes[list(top_20_entries.keys())], 
                      feature_names=list(top_20_entries_short.keys()))
    return sorted_dict  # ranked_filtered_corr
