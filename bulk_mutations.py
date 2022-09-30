#%%
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib as mpl
from sklearn.preprocessing import scale
plt = mpl.pyplot
#%%
sns.set_theme('notebook', 'white',font_scale=1.5)
os.chdir('/home2/s190548/work_mu/xiaoling_scrna/')
if not os.path.exists('results'):
    os.mkdir('results')
output_path = '/home2/s190548/work_mu/xiaoling_scrna/2022_new_bulkRNAseq_results/'
mpl.rcParams.update({'font.family':'Arial'})
# %%
os.chdir(
    '/archive/shared/wang_mu/xiaoling_scrna/2022_new_bulkRNAseq_results')
cell_line_mut = pd.read_csv('annotated_mutations_pooled.txt', sep='\t')
cg_mut = cell_line_mut[
    (cell_line_mut.Ref.isin(['C','G'])) &
    (cell_line_mut.Alt.isin(['A','T','C','G']))]
mut_counts = pd.DataFrame(cg_mut['Sample'].value_counts())
mut_counts.columns = ['C and G mutations']
mut_counts = mut_counts.sort_index()
tcw_mut = cg_mut[
    cg_mut.mut_context.isin(['TCT','TCA'])
    ]['Sample'].value_counts()
mut_counts['TCW mutations'] = tcw_mut.reindex(mut_counts.index, fill_value=0)
apobec_mut = cg_mut[
    (cg_mut.mut_context.isin(['TCT','TCA'])) &
    (cg_mut.mut_type.isin(['C/T','C/G']))
    ]['Sample'].value_counts()
mut_counts['APOBEC mutations'] = apobec_mut.reindex(
    mut_counts.index, fill_value=0)
tc_mut = cg_mut[
    (cg_mut.mut_context.str[:2] == 'TC')
    ]['Sample'].value_counts()
mut_counts['TC mutations'] = tc_mut.reindex(
    mut_counts.index, fill_value=0)
# %%
genotype = pd.read_csv("SYNCRIP_genotype.csv", index_col=0)
su2c_muts = pd.read_csv("muatations_unfiltered.txt", sep="\t", index_col=0)
su2c_muts = su2c_muts.T.merge(genotype[["genotype"]], left_index=True, right_index=True)
# %%
cg_muts = [x for x in su2c_muts.columns if x[2] in (['C','G'])]
tcw_muts = [x for x in cg_muts if (x[0] == 'T') & (x[-1] in ['T','A'])]
apobec_muts = [x for x in tcw_muts if x[4] != 'A']
tc_muts = [x for x in su2c_muts.columns if (x[2] == 'C') & (x[0] == 'T')]
su2c_muts_counts = pd.DataFrame(index = su2c_muts.index, columns = mut_counts.columns)
i = 0
for mut_cat in [cg_muts, tcw_muts, apobec_muts, tc_muts]:
    _muts_counts = su2c_muts[mut_cat].sum(axis=1)
    su2c_muts_counts.iloc[:,i] = _muts_counts.loc[su2c_muts_counts.index]
    i +=1
su2c_muts_counts = su2c_muts_counts.merge(
    genotype[["genotype"]], left_index=True, right_index=True)
# %%
su2c_muts_counts.to_csv('Su2c_mutation_sum.csv')
mut_counts.to_csv('LNCaP_mutation_sum.csv')
counts = "/home2/s190548/work_mu/xiaoling_bulkmRNA/gene_counts_stringtie.csv"
downtream_genes = "/endosome/archive/shared/wang_mu/xiaoling_scrna/2022_new_bulkRNAseq_results/44_downstream_genes.txt"
hmap = pd.read_csv(counts,index_col=0)
genes = pd.read_csv(downtream_genes, header=None).iloc[:,0].values
cols = ['ntfs1','ntfs2','ntfs3','kdfs1','kdfs2','kdfs3','kdce1','kdce2','kdce3']
hmap = hmap.loc[genes, cols]
hmap.loc[:,:] = scale(np.log2(1+hmap), axis=1)
sns.clustermap(
    hmap.T,cmap='bwr', figsize=(20,6), col_cluster=False, vmax=2, vmin=-2)
plt.xlabel('log2(FPKM+1)')
plt.savefig(output_path + '44 gene heatmap.pdf', bbox_inches='tight')
# %%
from make_mut_transcriptional_figures import make_plot_data
mut_16 = [
    'AR', 'ARID1A', 'BRD7', 'CBX8', 'CUX1', 'EP300', 'FOXA1', 'FOXL1', 'HDAC5', 
    'HOXA10', 'HSF4', 'KLF11', 'NFATC2', 'RARG', 'SMARCA4', 'STAT3']
os.chdir('/home2/s190548/work_mu/ping_mu/')
filter_mutations = ['C/T','C/A','C/G','G/A','G/T','G/C']
kdce = make_plot_data('KDCE', filter_mutations=filter_mutations)
kdfs = make_plot_data('KDFS', filter_mutations=filter_mutations)
plot_genes = kdce.append(kdfs).index.unique()
plot_data = cg_mut[cg_mut['Gene.refGene'].isin(plot_genes)]
plot_data['VAF'] = plot_data.Tumor_alt / (plot_data.Tumor_alt + plot_data.Tumor_ref)
plot_data = plot_data[
    (plot_data['Func.refGene'].isin(['exonic','splicing'])) &
    (plot_data['ExonicFunc.refGene'] != 'synonymous SNV')
    ]
plot_data['Condition'] = [x[:4] for x in plot_data['Sample'].values]
plot_data = plot_data.groupby(['Condition','mut_name','Gene.refGene']).VAF.sum()/3
plot_data = plot_data.groupby(['Condition','Gene.refGene']).max()
plot_data = plot_data.reset_index().pivot(
    index = 'Condition', columns = 'Gene.refGene', values='VAF').fillna(0)
plot_data.loc['NTFS'] = 0
sns.clustermap(
    plot_data.loc[['NTFS','NTCE','KDFS','KDCE']].T, figsize=(8,20),
    col_cluster=False, cmap = 'YlGnBu_r')
mpl.pyplot.savefig(output_path + '/top_mutated_genes_bulk_by_GSEA_and_VAF.pdf')
mpl.pyplot.close()
# %%
plot_data['ExonicFunc.refGene'].unique()
# %%

# %%
