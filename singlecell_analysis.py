#%%
from scipy.stats import ttest_ind, ranksums
import os
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib as mpl
import holoviews as hv
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform, cdist
from itertools import combinations
from sklearn.preprocessing import minmax_scale
hv.extension('matplotlib')
#%%
sns.set_theme('notebook', 'white',font_scale=1.5)
os.chdir('/home2/s190548/work_mu/xiaoling_scrna/')
if not os.path.exists('results'):
    os.mkdir('results')
samples = os.listdir('scrnaprep')
output_path = '2022_new_scRNAseq_results/'
sc.settings.figdir = output_path
samples_design = pd.Series({
    'XL1': 'NTFS',
    'XL2': 'KDFS',
    'XL3': 'NTCE1',
    'XL4': 'KDCE1',
    'XL5': 'NTCE2',
    'XL6': 'KDCE2',
})
mpl.rcParams.update({'font.family':'Arial'})
# ITH by Sample
def cal_diffusion(adata, group_by):
    clusters = sorted(adata.obs[group_by].unique())
    cluster_avg = pd.DataFrame(
        adata.obsm['X_pca'][:,:25], index = adata.obs_names
        ).groupby(adata.obs[group_by]).mean()
    corr_matrix = cdist(adata.obsm['X_pca'][:,:25], cluster_avg, metric='correlation')
    corr_matrix = pd.DataFrame(
        corr_matrix, index=adata.obs_names, columns=clusters)
    corr_matrix[group_by] = adata.obs[group_by]
    corr_matrix = corr_matrix.apply(lambda x: x[x[-1]], axis=1)
    outliers = corr_matrix.groupby(
        adata.obs[group_by].astype(str)).apply(lambda x: x>x.quantile(0.50))
    corr_matrix = pd.DataFrame(
        corr_matrix, index=adata.obs_names, columns=['Distance'])
    corr_matrix[group_by] = adata.obs[group_by].astype(str)
    corr_matrix = corr_matrix[outliers]
    return corr_matrix

def cal_ith(adata, group_by):
    ith = pd.DataFrame(index=adata.obs_names, columns=['Distance'])
    for s in adata.obs[group_by].unique():
        s_df = adata[adata.obs[group_by]==s]
        dist = squareform(pdist(s_df.obsm['X_pca'][:,:25], metric='correlation'))
        dist = np.median(dist, axis=1)
        ith.loc[s_df.obs_names,'Distance'] = dist
        ith.loc[s_df.obs_names, group_by] = s
    ith['Distance'] = ith['Distance'].astype(float)
    return ith
#%%
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# Starting point 
os.chdir('/home2/s190548/work_mu/xiaoling_scrna/')
merged_adata = sc.read_h5ad('MNN_BN_normed_data_v3_exp_batch_down_sampled.h5ad')
merged_adata.obs['leiden'] = merged_adata.obs['leiden'].astype(str)
merged_adata.obs.loc[merged_adata.obs['leiden']=='9','leiden'] = '8'
merged_adata = merged_adata[~(merged_adata.obs.leiden=='8')]
# %%
# ==============================================================================
# ==============================================================================
# ==============================================================================
# Analyzing mutations
mut_anno = pd.read_csv('annotated_mutations_pooled.txt', sep='\t')
tcw_mut = mut_anno[mut_anno.mut_context.isin(['TCA','TCT'])].index.unique()
metadata = merged_adata.obs
cluster_mutations = pd.DataFrame()
cb_wise_mutations = pd.DataFrame()
tcw_cb_mutations = pd.DataFrame()
cb_mut_gene = pd.DataFrame()
df_coverage = pd.DataFrame()
for i, s in enumerate(sorted(samples)):
    cluster_meta = metadata[metadata['Sample'] == samples_design[s[-3:]]]
    cluster_sizes = cluster_meta['leiden'].value_counts()
    mutation_fn = os.path.join('results', s+ '_counts_CB.tsv')
    mutations = pd.read_csv(mutation_fn, index_col=7, sep='\t')
    mutations.index = mutations.index + '-' + str(i)
    total_coverage = mutations.groupby('gene').total_CB.sum()
    df_coverage = df_coverage.append(pd.DataFrame(total_coverage))
    valid_genes = total_coverage[total_coverage>100].index
    mutations = mutations[mutations.gene.isin(valid_genes)]
    mutations = mutations[mutations['type']!='synonymous SNV']
    mutations = mutations.loc[[x for x in cluster_meta.index if x in mutations.index]]
    mutations = mutations[mutations.alt_count>0]
    mutations['SNP'] = ('chr' + mutations.chrm.astype(str) + ':' + 
    mutations.start.astype(int).astype(str) + '_' + mutations.ref + '/' + mutations.alt
    )
    # total mutations
    _cb_mut = mutations.groupby(mutations.index).count().iloc[:,[0]]
    _cb_mut.columns = ['Mutation_counts']
    _cb_mut['Sample'] = s
    # CG mutations
    cg_mutations = mutations[mutations.ref.isin(['C','G'])]
    _cb_cg_mut = cg_mutations.groupby(cg_mutations.index).count().iloc[:,[0]]
    _cb_cg_mut.columns = ['CG_Mutation_counts']
    _cb_mut = _cb_mut.merge(
        _cb_cg_mut, left_index = True, right_index = True,how = 'left').fillna(0)
    # report number mutations
    cb_wise_mutations = cb_wise_mutations.append(_cb_mut)
    # TCW mutations
    tcw_mutations = mutations[mutations.SNP.isin(tcw_mut)]
    _tcw_cb_mut = tcw_mutations.groupby(tcw_mutations.index).count().iloc[:,[0]]
    _tcw_cb_mut.columns = ['Mutation_counts']
    _tcw_cb_mut['Sample'] = s
    tcw_cb_mutations = tcw_cb_mutations.append(_tcw_cb_mut)
    
    _cb_mut_g = cg_mutations.groupby(['barcode','gene'])['alt_count'].sum().reset_index()
    _cb_mut_g.columns = ['barcode','gene','mut']
    _cb_mut_g['mut'] = _cb_mut_g['mut']>0
    _cb_mut_g = _cb_mut_g.pivot(index='barcode',columns='gene', values='mut').fillna(False)
    cb_mut_gene = cb_mut_gene.append(_cb_mut_g)

    mutations['cluster'] = cluster_meta.loc[mutations.index, 'leiden'].values
    mut_stats = mutations.groupby(
        ['gene','cluster']).count().iloc[:,0].fillna(0)
    mut_stats = mut_stats.groupby(
        ['gene','cluster']).apply(lambda x: x/cluster_sizes[x.index[0][1]])
    mut_stats = mut_stats[mut_stats>0].reset_index()
    mut_stats.columns = ['Gene','Cluster','Variant_cluster_frequency']
    mut_stats['Trt'] = samples_design[s[-3:]]
    cluster_mutations = cluster_mutations.append(
        mut_stats, ignore_index=True
    )
#%%
# CG mutation counts
merged_adata.obs['CG_Mutations'] = cb_wise_mutations.reindex(
    index = merged_adata.obs_names)['CG_Mutation_counts'].fillna(0)
cmap = sns.color_palette("Reds", as_cmap=True)
sc.pl.umap(
    merged_adata,color=['CG_Mutations'],
    show=False,
    # cmap=cmap,
    save='_CG mutation counts.pdf', 
    title = 'Num. CG Mutations', alpha = 0.75,
    s = 15, vmin = 'p1', vmax = 'p99.9')

sc.pl.umap(
    merged_adata,color=['leiden'],
    # show=False,
    save='_Clusters.pdf', title = 'Clusters',alpha=0.8)

# sc.pl.umap(
#     merged_adata,color=['leiden'],
#     # show=False,
#     save='_Clusters.pdf', title = '_Cluster_6',alpha=0.8)
# %%
cb_wise_mutations = cb_wise_mutations.reindex(index = metadata.index)
cb_wise_mutations['Sample'] = metadata['Sample']
cb_wise_mutations.Mutation_counts.fillna(0, inplace=True)
cb_wise_mutations['Cluster'] = 'Cluster_' + metadata['leiden'].astype(str)
cb_wise_mutations.CG_Mutation_counts.fillna(0, inplace = True)
# outliers = cb_wise_mutations.groupby(['Cluster'])['CG_Mutation_counts'].apply(
#     lambda x: x>x.quantile(0.999)
# )
# cb_wise_mutations[outliers]

# Mutation counts by sample
ax = sns.boxplot(
    y='CG_Mutation_counts', x= 'Sample', data=cb_wise_mutations,
    order = ['NTFS', 'NTCE1', 'NTCE2', 'KDFS', 'KDCE1','KDCE2'],
    hue = 'Sample', palette = ['b', 'b','b', 'r','r','r'], dodge=False,
    # cut=0, bw=0.5, inner ='quartile',
    )
ax.get_legend().remove()
ax.set_ylabel('Number of mutations per cell')
ax.set_xlabel('Condition')
pvals_title = []
for a, b in zip(['NTFS', 'NTCE1', 'NTCE2'],['KDFS', 'KDCE1','KDCE2']):
    v1 = cb_wise_mutations.loc[cb_wise_mutations['Sample']==a, 'CG_Mutation_counts']
    v2 = cb_wise_mutations.loc[cb_wise_mutations['Sample']==b, 'CG_Mutation_counts']
    p = ttest_ind(v1.values,v2.values, equal_var=False)[1]
    pvals_title.append('{} vs {} : {:.3f}'.format(b, a, p))
plt.title('\n'.join(pvals_title))
plt.savefig(
    output_path + 'Single_cell_CG_mutation_load_by_Sample.pdf',
    bbox_inches = 'tight')
plt.close()
# Mutation counts by cluster

cluster_orders = [
    'Cluster_0','Cluster_2','Cluster_3','Cluster_6',
    'Cluster_1','Cluster_4','Cluster_5','Cluster_7']
ax = sns.boxplot(
    y='CG_Mutation_counts', x= 'Cluster', data=cb_wise_mutations,
    order = cluster_orders,
    # cut=0, bw=0.5, scale='width', inner ='quartile',
    )
ax.set_ylabel('Number of mutations per cell')
ax.set_xlabel('Clusters')
ax.set_xticklabels(cluster_orders,rotation=90)
winners = cluster_orders[:4]
losers_v = cb_wise_mutations[
    ~cb_wise_mutations.Cluster.isin(winners)].Mutation_counts.values
winner_v = cb_wise_mutations[
    cb_wise_mutations.Cluster.isin(winners)].Mutation_counts.values
pval = ttest_ind(losers_v, winner_v, equal_var=False)[1]
plt.title('Winner vs Lose Pval: {:.3e}'.format(pval))
plt.savefig(
    output_path + 'Single_cell_CG_mutation_load_cluster.pdf',
    bbox_inches = 'tight')
plt.close()

# Mutation counts by cluster and sample
#%%
# ==============================================================================
# ==============================================================================
# ==============================================================================
######## Cell cycle analysis ########
sc.pl.umap(
    merged_adata,
    color = 'phase', save='_cell cycle', show=False, palette ='Set3', s = 10,
    alpha=0.75)

phases = merged_adata.obs.groupby(['Sample','phase']).count()
phases = phases['percent_mito'].reset_index()
phases = phases.pivot(index = 'Sample', columns = 'phase', values = 'percent_mito')
phases = phases.rename_axis(None).rename_axis(None, axis=1)
phases = phases.apply(lambda x: x/x.sum(), axis=1)
x_ticks = ['NTFS', 'NTCE1', 'NTCE2', 'KDFS', 'KDCE1','KDCE2']
phases.loc[x_ticks].plot(
    kind = 'bar', stacked = True)
plt.legend(bbox_to_anchor = (1,0.5), loc= 'center left')
plt.xticks(rotation=30)
plt.ylabel('Fractions of all cells')
plt.savefig(output_path + '/cell_cycle_phases_by_sample.pdf', bbox_inches = 'tight')
plt.close()
#%%
# ==============================================================================
# ==============================================================================
# ==============================================================================
######### Evaluating heterogenity ################

ith = cal_ith(merged_adata, 'Sample')
ith['Distance'] = ith['Distance'].astype(float)
merged_adata.obs['ITH'] = ith.loc[merged_adata.obs_names,'Distance']
# sc.pl.umap(
#     merged_adata,color=['ITH'], show=False, cmap='viridis',
#     save='_ITH.pdf', title = 'ITH score', alpha = 0.75,
#     s = 15)
sns.boxplot(
    data=ith, y='Distance', x='Sample',hue='Sample',
    order = ['NTFS','NTCE1','NTCE2','KDFS','KDCE1','KDCE2'], 
    hue_order=['NTFS','NTCE1','NTCE2','KDFS','KDCE1','KDCE2'],
    dodge=False,
    # cut=0, bw=0.5, inner ='quartile',
    )
plt.legend(bbox_to_anchor=(1,.5),loc='center left')
plt.ylabel('ITH Score')
plt.xlabel('')
pvals_title = []
for c in combinations(['NTFS','NTCE1','NTCE2','KDFS','KDCE1','KDCE2'],2):
    c1, c2 = c
    v_c1 = ith[ith['Sample'] == c1].Distance.values
    v_c2 = ith[ith['Sample'] == c2].Distance.values
    p = ttest_ind(v_c1, v_c2, equal_var=False)[1]
    if p>0:
        pvals_title.append('{} vs {} : {:.3e}'.format(c1, c2, p))
plt.title('\n'.join(pvals_title))
plt.savefig(output_path + '/ITH scores by sample.pdf', bbox_inches='tight')
plt.close()

# ITH by Genotype and Treatment
ith['Genotype'] = [x[:2] for x in ith['Sample'].values]
ith['Treatment'] = [x[2:4] for x in ith['Sample'].values]
ith['Distance'] = ith['Distance'].astype(float)
sns.violinplot(
    y='Distance', x='Genotype', hue = 'Treatment', data = ith, dodge=False, 
    bw = 0.5, split=True, inner = 'quartile',
    )
plt.legend(bbox_to_anchor=(1,.5),loc='center left')
plt.ylabel('ITH Score')
plt.xlabel('')
pvals = []
for genotype in ['NT','KD']:
    ce_v = ith.loc[(ith.Genotype==genotype) & (ith.Treatment=='CE'), 'Distance'].values
    fs_v = ith.loc[(ith.Genotype==genotype) & (ith.Treatment=='FS'), 'Distance'].values
    pval = ranksums(ce_v, fs_v)[1]
    pvals.append(pval)
plt.title(
    'U-test CE vs FS in NT: {:.2e}\nU-test CE vs FS in KD: {:.2e}'.format(*pvals))
plt.savefig(
    output_path + '/ITH scores by Genotype and Treatment.pdf', bbox_inches='tight')
plt.close()

ith_ce = ith[ith.Treatment == 'CE'].copy()
ith_ce['Treatment_time'] = [x[2:] for x in ith_ce['Sample'].values]
sns.violinplot(
    y='Distance', x='Genotype', hue = 'Treatment_time',
    data = ith_ce, dodge=False, 
    bw = 0.5, split=True, inner = 'quartile')
plt.legend(bbox_to_anchor=(1,.5),loc='center left')
plt.ylabel('ITH Score')
plt.xlabel('')
pvals = []
for genotype in ['NT','KD']:
    ce1_v = ith_ce.loc[(ith_ce.Genotype==genotype) & (ith_ce.Treatment_time=='CE1'), 'Distance'].values
    ce2_v = ith_ce.loc[(ith_ce.Genotype==genotype) & (ith_ce.Treatment_time=='CE2'), 'Distance'].values
    pval = ranksums(ce_v, fs_v)[1]
    pvals.append(pval)
plt.title(
    'U-test in NT: {:.2e}\nU-test in KD: {:.2e}'.format(*pvals))
plt.savefig(
    output_path + '/ITH scores by Genotype and Treatment times.pdf', bbox_inches='tight')
plt.close()

# ITH by cluster 
ith = cal_ith(merged_adata, 'leiden')
ith['cluster'] = merged_adata.obs.leiden[ith.index].astype(str)
sns.boxplot(
    data=ith, y='Distance', x='cluster', dodge=False,
    order = ['0','2','3','6','1','4','5','7'],
    # bw = 0.5, split=True, inner = 'quartile',
    )
plt.legend(bbox_to_anchor=(1,.5),loc='center left')
plt.ylabel('ITH Score')
plt.xlabel('Cluster')
winners = ['0','2','3','6']
losers_v = ith[~ith.cluster.isin(winners)].Distance.values
winner_v = ith[ith.cluster.isin(winners)].Distance.values
pval = ttest_ind(losers_v, winner_v, equal_var=False)[1]
plt.title('Winner vs Lose Pval: {:.3e}'.format(pval))
plt.savefig(output_path + '/ITH scores by cluster.pdf', bbox_inches='tight')
plt.close()

# Hetergenity score by distance to centroid
# Diffusion score

diffusion_score = cal_diffusion(merged_adata, 'Sample')
sns.boxplot(
    data=diffusion_score, y='Distance', x='Sample',hue='Sample',
    order = ['NTFS','NTCE1','NTCE2','KDFS','KDCE1','KDCE2'], 
    hue_order=['NTFS','NTCE1','NTCE2','KDFS','KDCE1','KDCE2'],dodge=False,
    # bw = 0.5, inner = 'quartile',
    )
plt.legend(bbox_to_anchor=(1,.5),loc='center left')
plt.ylabel('Distance to sample centroid')
pvals_title = []
for c in combinations(['NTFS','NTCE1','NTCE2','KDFS','KDCE1','KDCE2'],2):
    c1, c2 = c
    v_c1 = diffusion_score[diffusion_score['Sample'] == c1].Distance.values
    v_c2 = diffusion_score[diffusion_score['Sample'] == c2].Distance.values
    p = ttest_ind(v_c1, v_c2, equal_var=False)[1]
    if p>0:
        pvals_title.append('{} vs {} : {:.3e}'.format(c1, c2, p))
plt.title('\n'.join(pvals_title))
plt.xlabel('')
plt.savefig(output_path + '/Diffusion scores by sample.pdf', bbox_inches='tight')
plt.close()

# Diffusion score by cluster

diffusion_score = cal_diffusion(merged_adata, 'leiden')
sns.boxplot(
    data=diffusion_score, y='Distance', x='leiden',hue='leiden',
    order = ['0','2','3','6','1','4','5','7'],
    hue_order = ['0','2','3','6','1','4','5','7'], 
    dodge=False,
    # bw = 0.5, inner = 'quartile',
    )
plt.legend(bbox_to_anchor=(1,.5),loc='center left')
plt.ylabel('Distance to sample centroid')
plt.xlabel('')
losers_v = diffusion_score[~diffusion_score.leiden.isin(winners)].Distance.values
winner_v = diffusion_score[diffusion_score.leiden.isin(winners)].Distance.values
pval = ttest_ind(losers_v, winner_v, equal_var=False)[1]
plt.title('Winner vs Lose Pval: {:.3e}'.format(pval))
plt.savefig(output_path + '/Diffusion scores by cluster.pdf', bbox_inches='tight')
plt.close()

# Diffusion score by samples
diffusion_score = cal_diffusion(merged_adata, 'Sample')
diffusion_score['Genotype'] = [x[:2] for x in diffusion_score['Sample'].values]
diffusion_score['Treatment'] = [x[2:4] for x in diffusion_score['Sample'].values]
diffusion_score['Distance'] = diffusion_score['Distance'].astype(float)
sns.violinplot(
    y='Distance', x='Genotype', hue = 'Treatment', data = diffusion_score, dodge=False, 
    bw = 0.5, split=True, inner = 'quartile')
plt.legend(bbox_to_anchor=(1,.5),loc='center left')
plt.ylabel('Diffusion Score')
plt.xlabel('')
pvals = []
for genotype in ['NT','KD']:
    ce_v = diffusion_score.loc[
        (diffusion_score.Genotype==genotype) & (diffusion_score.Treatment=='CE'), 'Distance'
        ].values
    fs_v = diffusion_score.loc[
        (diffusion_score.Genotype==genotype) & (diffusion_score.Treatment=='FS'), 'Distance'
        ].values
    pval = ranksums(ce_v, fs_v)[1]
    pvals.append(pval)
plt.title(
    'U-test CE vs FS in NT: {:.2e}\nU-test CE vs FS in KD: {:.2e}'.format(*pvals))
plt.savefig(
    output_path + '/Diffusion scores by Genotype and Treatment.pdf', bbox_inches='tight')
plt.close()

diff_ce = diffusion_score[diffusion_score.Treatment == 'CE'].copy()
diff_ce['Treatment_time'] = [x[2:] for x in diff_ce['Sample'].values]
sns.violinplot(
    y='Distance', x='Genotype', hue = 'Treatment_time',
    data = diff_ce, dodge=False, 
    bw = 0.5, split=True, inner = 'quartile')
plt.legend(bbox_to_anchor=(1,.5),loc='center left')
plt.ylabel('Diffusion Score')
plt.xlabel('')
pvals = []
for genotype in ['NT','KD']:
    ce1_v = diff_ce.loc[
        (diff_ce.Genotype==genotype) & (diff_ce.Treatment_time=='CE1'), 'Distance'
        ].values
    ce2_v = diff_ce.loc[
        (diff_ce.Genotype==genotype) & (diff_ce.Treatment_time=='CE2'), 'Distance'
        ].values
    pval = ranksums(ce_v, fs_v)[1]
    pvals.append(pval)
plt.title(
    'U-test in NT: {:.2e}\nU-test in KD: {:.2e}'.format(*pvals))
plt.savefig(
    output_path + '/Diffusion scores by Genotype and Treatment times.pdf', bbox_inches='tight')
plt.close()
#%%
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ======== Cluster composition ========
sample_order = ['NTFS','NTCE1','NTCE2','KDFS','KDCE1','KDCE2']
for genotype in ['NT','KD', 'All']:
    if genotype == 'All':
        merged_adata.obs['Treatment_time'] = [
            x[2:] for x in merged_adata.obs['Sample'].values
            ]
        cluster_comp = merged_adata.obs.groupby(
            ['leiden','Treatment_time']
            ).count().iloc[:,0].fillna(0)
    else:        
        cluster_comp = merged_adata[
            merged_adata.obs['Sample'].str[:2] == genotype
            ].obs.groupby(['leiden','Sample']).count().iloc[:,0].fillna(0)
    cluster_comp = cluster_comp.reset_index()
    cluster_comp.columns = ['cluster','trt','cell_count']
    cluster_by_trt = cluster_comp.groupby('cluster').cell_count.apply(
        lambda x: x/x.sum())
    trt_by_cluster = cluster_comp.groupby('trt').cell_count.apply(
        lambda x: x/x.sum())
    cluster_comp['fraction_cluster'] = cluster_by_trt.values
    cluster_comp['fraction_trt'] = trt_by_cluster.values
    plot_data = cluster_comp.pivot(index='cluster',columns='trt',values='fraction_trt')
    if genotype == 'All':
        plot_data = plot_data[['FS','CE1','CE2']]
    else:
        plot_data = plot_data[[x for x in sample_order if x in plot_data.columns]]
    colors = sns.color_palette('tab20',len(plot_data.index))
    ax = plt.stackplot(
        plot_data.columns, plot_data, labels=plot_data.index, colors=colors)
    plt.legend(bbox_to_anchor=(1,0.5), loc='center left')
    plt.savefig(
        output_path + '/{} cluster fractions.pdf'.format(genotype),
        bbox_inches = 'tight')
    plt.close()
#%%
cell_labels = []
cell_ori_id = []
leiden_label = []
ori_leiden = []
binned_df = pd.DataFrame()
for c in merged_adata.obs.leiden.unique():
    cells = (merged_adata.obs.leiden==c).values
    _adata = merged_adata[cells]
    for s in _adata.obs['Sample'].unique():
        s_cells = (_adata.obs['Sample'] == s).values
        _df_pca = _adata.obsm['X_pca'][s_cells]
        _df_umap = _adata.obsm['X_umap'][s_cells]
        n_clusters = _df_pca.shape[0]//10
        if n_clusters < 2:
            continue
            # clusters = ['c0']*len(_df_pca)
            # leiden_label += [s + '_' + c]
            # ori_leiden += [c]
        else:
            clusters = AgglomerativeClustering(
                n_clusters,linkage='complete', affinity='correlation').fit_predict(_df_pca)
            clusters = ['c{}'.format(x) for x in clusters]
            leiden_label += [s + '_' + c]*n_clusters
            ori_leiden += [c]*n_clusters
            cell_ori_id += _adata.obs_names.astype(str)[s_cells].tolist()
            cell_labels += [s+'_' + c + '_' + str(x) for x in clusters]
        cluster_df = _adata[s_cells].raw.to_adata().to_df().groupby(clusters).mean()
        cluster_pca = pd.DataFrame(_df_pca).groupby(clusters).mean()
        cluster_umap = pd.DataFrame(_df_umap).groupby(clusters).mean()
        if cluster_df.shape[0]!=cluster_pca.shape[0]:
            break
        binned_df = binned_df.append(cluster_df, ignore_index=True)
cell_clusters = pd.Series(cell_labels, index = cell_ori_id)
if not os.path.exists('monocle'):
    os.mkdir('monocle')
binned_meta = pd.DataFrame(index = binned_df.index)
binned_meta['leiden'] = ori_leiden
binned_meta['Sample'] = [x.split('_')[0] for x in leiden_label]
binned_meta.index = cell_clusters.unique()
merged_adata.obs.Mutation_counts = merged_adata.obs.Mutation_counts.fillna(0)
more_meta = merged_adata[
    cell_clusters.index].obs.groupby(cell_clusters, sort=False).mean()
more_meta['phase'] = [
    'G2M' if x>=y else 'S' for x, y in more_meta[['G2M_score','S_score']].values]
more_meta.loc[(more_meta[['G2M_score','S_score']]<=0).all(axis=1),'phase'] = 'G1'
binned_meta['phase'] = more_meta.phase
binned_meta['Mutation_counts'] = more_meta.Mutation_counts
ith = cal_ith(merged_adata, 'leiden')
ith['Distance'] = ith['Distance'].astype(float)
binned_meta['ITH'] = ith.loc[
    cell_clusters.index].groupby(cell_clusters.values, sort=False)['Distance'].mean()
binned_meta = binned_meta.reset_index().iloc[:,1:]
binned_meta.to_csv('monocle/cell_meta.txt', sep='\t')
binned_df.round(3).to_csv('monocle/expr.txt', sep='\t')

# %%
downstream_genes = pd.read_csv('downstream_genes_v2.csv')
geneset_dict = {}
sc.pp.scale(merged_adata)
for col in downstream_genes:
    genes = downstream_genes[col].dropna().values.tolist()
    if len(genes) == 0:
        continue
    geneset_dict[col] = genes
for geneset_name, geneset in geneset_dict.items():
    sc.tl.score_genes(merged_adata, geneset, score_name=geneset_name,
    ctrl_size=500, use_raw=True)

cell_scores = merged_adata.obs.loc[:,'AZARE_STAT3_Target':]
cluster_scores = cell_scores.groupby(merged_adata.obs['leiden']).mean()
cluster_scores.loc[:,:] = minmax_scale(cluster_scores, axis=0)
cluster_scores = cluster_scores.loc[['0', '2', '3', '6', '1', '4', '5', '7']]
sns.clustermap(
    cluster_scores.T, col_cluster=False)
plt.savefig(output_path + '/cluster_lineage_heatmap.svg')
# %%
merged_adata.obs = merged_adata.obs.iloc[:,:12]
downstream_genes = pd.read_csv('downstream_genes_c6.csv')
geneset_dict= {}
for col in downstream_genes:
    genes = downstream_genes[col].dropna().values.tolist()
    if len(genes) == 0:
        continue
    geneset_dict[col] = genes
for geneset_name, geneset in geneset_dict.items():
    sc.tl.score_genes(merged_adata, geneset, score_name=geneset_name,
    ctrl_size=500, use_raw=True)
c6 = merged_adata[merged_adata.obs.leiden == '6']
c6.obs.loc[:,'BRD7_gene_list':] = minmax_scale(c6.obs.loc[:,'BRD7_gene_list':])
sc.pl.umap(
    c6, color=geneset_dict.keys(), 
ncols=3, save='_Cluster6_Partial_Downstreams.pdf')