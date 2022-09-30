#%%
from scrna_utils import *
import os
import scanpy as sc
import seaborn as sns
import numpy as np
import pandas as pd
#%%
sns.set_theme('paper','white',font_scale=2)
os.chdir('/home2/s190548/work_mu/xiaoling_scrna/')
if not os.path.exists('results'):
    os.mkdir('results')
samples = os.listdir('scrnaprep')
samples_design = pd.Series({
    'XL1': 'NTFS',

    'XL2': 'KDFS',
    'XL3': 'NTCE1',
    'XL4': 'KDCE1',
    'XL5': 'NTCE2',
    'XL6': 'KDCE2',
})
#%%
# Get minimal per batch cell readcount
min_total_counts = np.inf
for i, s in enumerate(sorted(samples)):
    adata = sc.read_10x_mtx(
        os.path.join(os.getcwd(),'scrnaprep',s,'cleaned counts'),
        )
    adata = cal_qc(adata)
    min_total_counts = np.min([min_total_counts,adata.obs['n_counts'].median()])

cm_genes = []
list_adata = [0]*6
variable_genes = []
for i, s in enumerate(sorted(samples)):
    adata = sc.read_10x_mtx(
        os.path.join(os.getcwd(),'scrnaprep',s,'cleaned counts'),
        )
    adata = cal_qc(adata)
    sc.pp.downsample_counts(adata, counts_per_cell = int(min_total_counts))
    # plot_qc(adata)
    # plt.savefig(os.path.join('results','{} scRNAseq QC.svg'.format(s)))
    # plt.close()
    adata = mito_qc(
        adata, min_genes=100, max_genes=8000, percent_mito_cutoff=0.2)
    adata = normalize_adata(adata)
    if cm_genes == []:
        cm_genes = adata.var_names.tolist()
    else:
        cm_genes = [x for x in cm_genes if x in adata.var_names]
    adata.obs['Sample'] = samples_design['XL'+str(i+1)]
    sc.pp.highly_variable_genes(adata, min_mean=0.01, min_disp=0.5)
    if variable_genes == []:
        variable_genes = adata.var_names[adata.var['highly_variable']].tolist()
    else:
        variable_genes = variable_genes + adata.var_names[adata.var['highly_variable']].tolist()
    list_adata[i] = adata

for i in range(6):
    list_adata[i] = list_adata[i][:,cm_genes]

variable_genes = list(set(variable_genes))
variable_genes = [x for x in variable_genes if x in cm_genes]

# MNN batch effect correction based on Flowcell
batch_1_adata = sc.concat(list_adata[:5],index_unique='-')
batch_2_adata = list_adata[5]
batch_2_adata.obs_names = [x+'-5' for x in batch_2_adata.obs_names]
merged_adata = sc.external.pp.mnn_correct(
    batch_1_adata, batch_2_adata,
    k = 30,
    var_subset = variable_genes)[0]

# clustering
merged_adata.var.loc[variable_genes,'highly_variable'] = True
merged_adata.var['highly_variable'].fillna(False, inplace=True)
merged_adata.obs_names = [x[:-2] for x in merged_adata.obs_names]
sc.pp.scale(merged_adata)
rps_genes = [x for x in merged_adata.var_names if x[:3] in ['RPS','RPL']]
merged_adata.var.loc[rps_genes,'highly_variable'] = False
sc.tl.pca(merged_adata,n_comps=50,random_state=0,use_highly_variable=True)
sc.pp.neighbors(merged_adata, n_pcs=25, n_neighbors=15)
sc.tl.leiden(merged_adata, resolution=0.5)
sc.tl.umap(merged_adata, min_dist=0.75)

# Cell cycle scores
cell_cycle_genes = [x.strip() for x in open('regev_lab_cell_cycle_genes.txt')]
s_genes = cell_cycle_genes[:43]
g2m_genes = cell_cycle_genes[43:]
cell_cycle_genes = [x for x in cell_cycle_genes if x in merged_adata.var_names]
sc.tl.score_genes_cell_cycle(merged_adata, s_genes=s_genes, g2m_genes=g2m_genes)

merged_adata.obs['batch'] = samples_design[
    ['XL'+str(int(x)+1) for x in merged_adata.obs.batch.astype(str).values]
].values
merged_adata.write_h5ad('MNN_BN_normed_data_v3_exp_batch_down_sampled.h5ad')