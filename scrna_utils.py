import os
import pandas as pd
import seaborn as sns
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind_from_stats
from statsmodels.stats.multitest import fdrcorrection
import gseapy as gp
sc.settings.verbosity = 0

def cal_qc(adata):
    adata.var_names_make_unique()
    mito_genes = adata.var_names.str.startswith("mt-")
    mito_genes |= adata.var_names.str.startswith("MT-")
    all_genes = adata.var_names.values
    rps_genes = [
        x for x in all_genes if 
        ('rps' == x.lower()[:3]) |
        ('rpl' == x.lower()[:3]) |
        ('rps' == x.lower()[:3]) |
        ('rpl' == x.lower()[:3])]
    # for each cell compute fraction of counts in mito genes vs. all genes
    # the `.A1` is only necessary as X is sparse (to transform to a dense array after summing)
    adata.obs["percent_mito"] = np.sum(
        adata[:, mito_genes].X, axis=1) / np.sum(adata.X, axis=1
    )
    adata.obs["percent_ribosomal"] = np.sum(
        adata[:, rps_genes].X, axis=1) / np.sum(adata.X, axis=1
    )
    # add the total counts per cell as observations-annotation to adata
    adata.obs["n_counts"] = adata.X.sum(axis=1)
    adata.obs["n_genes"] = np.sum(adata.X>0,axis=1)
    return adata

def plot_qc(adata, mito_high = None):
    fig = plt.figure(figsize=(18, 12))
    axes = fig.subplots(3, 2)
    axes = axes.ravel()
    sns.histplot(adata.obs["n_genes"], ax=axes[0], label="Number of genes")
    sns.histplot(adata.obs["n_counts"], ax=axes[1], label="Count depth")
    axes[0].set_title("Number of genes")
    axes[1].set_title("Count_depth")
    sns.histplot(adata.obs["percent_mito"], ax=axes[2])
    sns.histplot(adata.obs["percent_ribosomal"], ax=axes[3])
    if mito_high is None:
        norm = plt.Normalize(0, adata.obs["percent_mito"].max())
    else:
        norm = plt.Normalize(0, mito_high)
    sns.scatterplot(
        adata.obs["n_genes"],
        adata.obs["n_counts"],
        hue=adata.obs["percent_mito"],
        s=2,
        linewidth=0,
        alpha=0.6,
        ax=axes[4],
        legend=None,
        palette="gist_heat",
        hue_norm=norm,
    )
    sm = plt.cm.ScalarMappable(cmap="gist_heat", norm=norm)
    sm.set_array([])
    sns.scatterplot(
        adata.obs["percent_mito"],
        adata.obs["percent_ribosomal"],
        hue=adata.obs["percent_mito"],
        s=2,
        linewidth=0,
        alpha=0.6,
        ax=axes[5],
        legend=None,
        palette="gist_heat",
        hue_norm=norm,
    )
    axes[5].figure.colorbar(sm)
    plt.tight_layout()

def mito_qc(adata, min_genes=100, max_genes=4000, percent_mito_cutoff=0.5):
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_cells(adata, max_genes = max_genes)
    sc.pp.filter_genes(adata, min_cells=3)
    adata = adata[adata.obs["percent_mito"] < percent_mito_cutoff, :]
    return adata
    # plt.savefig(pathout+'/QC.png')


def normalize_adata(adata):
    # sc.pp.calculate_qc_metrics(adata, inplace=True)
    adata = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata
    return adata

def get_deg(adata, target, ref, groupby):
    if ref == 'rest':
        grouping_key = [
            x if x == target else 'rest' for x in adata.obs[groupby].values]
    else:
        grouping_key = adata.obs[groupby].values
    means = adata.to_df().groupby(grouping_key).mean()
    stds = adata.to_df().groupby(grouping_key).std()
    counts = adata.to_df().groupby(grouping_key).count().iloc[:,0]
    pvals = ttest_ind_from_stats(
        means.loc[target], stds.loc[target], counts.loc[target],
        means.loc[ref], stds.loc[ref], counts.loc[ref],
        False
    )[1]
    pvals[np.isnan(pvals)] = 1
    fdr_pval = fdrcorrection(pvals)[1]
    deg = pd.DataFrame(
        fdr_pval, index=means.columns, columns = [target + '_adjpval'])
    logfc = means.loc[target] - means.loc[ref]
    deg[target + '_logFC'] = logfc
    deg[target + '_pval'] = pvals
    deg = deg.sort_values(target + '_logFC', ascending=False)
    return deg

def get_deg_batched(adata, target, ref, groupby):
    pooled_deg = pd.DataFrame()
    for batch in adata.obs['batch'].unique():
        _batch_adata = adata[adata.obs['batch']==batch]
        if (_batch_adata.obs[groupby].value_counts()[target]<10).any():
            continue
        print(batch)
        _batch_deg = get_deg(_batch_adata, target, ref, groupby)
        _batch_deg.columns = [
            'batch_' + str(batch) + '_' + x for x in _batch_deg.columns]
        pooled_deg = pd.concat([pooled_deg, _batch_deg], axis=1)
    return pooled_deg

def get_scanpy_deg(adata, target, ref, groupby, use_raw=True, pval=0.1):
    sc.tl.rank_genes_groups(
        adata, groupby,
        groups=[target],
        reference=ref,
        n_genes = adata.raw.X.shape[1],
        use_raw=use_raw) 
    gene_names = pd.DataFrame(
        adata.uns['rank_genes_groups']['names'])[target]
    logfc = pd.DataFrame(
        adata.uns['rank_genes_groups']['logfoldchanges'])[target]
    pvals = pd.DataFrame(
        adata.uns['rank_genes_groups']['pvals_adj'])[target]
    deg = pd.DataFrame(logfc.values, index=gene_names.values, columns=['logFC'])
    deg['adj_pval'] = pvals.values
    deg = deg[~deg.index.isnull()].sort_values('logFC',ascending=False)
    deg = deg[deg.adj_pval<=pval]
    return deg

def get_scanpy_deg_batched(adata, target, ref, groupby, use_raw=True, pval=0.1):
    pooled_deg = pd.DataFrame()
    for batch in adata.obs['batch'].unique():
        _batch_adata = adata[adata.obs['batch']==batch]
        if (_batch_adata.obs[groupby].value_counts()[
            [target,ref]]<10).any():
            continue
        _batch_deg = get_scanpy_deg(_batch_adata, target, ref, groupby, use_raw, pval)
        _batch_deg.columns = [
            'batch_' + str(batch) + '_' + x for x in _batch_deg.columns]
        pooled_deg = pd.concat([pooled_deg, _batch_deg], axis=1)
    return pooled_deg

def make_deg_report(
    df_deg, libs = ['KEGG_2019_Mouse'], fn_prefix='', no_ribosomal=False,
    species = 'Human'):
    '''
    Interesting libaries: 
    'Cancer_Cell_Line_Encyclopedia',
    'Enrichr_Libraries_Most_Popular_Genes',
    'Enrichr_Submissions_TF-Gene_Coocurrence',
    'GO_Biological_Process_2018',
    'GO_Molecular_Function_2018',
    'Human_Gene_Atlas',
    'Human_Phenotype_Ontology',
    'KEGG_2019_Human',
    'KEGG_2019_Mouse',
    'MGI_Mammalian_Phenotype_Level_4_2019',
    'Mouse_Gene_Atlas',
    'OMIM_Disease',
    'Panther_2016',
    'Reactome_2016',
    'TRRUST_Transcription_Factors_2019',
    'WikiPathways_2019_Human',
    'WikiPathways_2019_Mouse',
    'dbGaP',
    '''
    output_fn = ' '.join(['DEG analysis report', '.xlsx'])
    writer = pd.ExcelWriter(output_fn)
    # Write each dataframe to a different worksheet.
    df_deg.to_excel(writer, sheet_name='DEG')
    pval_cols = [x for x in df_deg.columns if 'adjpval' in x]
    log_fc_cols = [x for x in df_deg.columns if 'logFC' in x]
    if no_ribosomal:
        all_genes = df_deg.index
        rps_genes = [
            x for x in all_genes if (
                (x.lower()[:4] in ['mrps','mrpl']) | 
                (x.lower()[:3] in ['rps','rpl'])
                )
            ]
        df_deg.drop(rps_genes, inplace=True)
    # Enrichment for each batch 
    for fc_col, pval_col in zip(log_fc_cols, pval_cols):
        _genes = df_deg[
            (df_deg[pval_col]<=0.05) & 
            (df_deg[fc_col]>0)
            ].sort_values(fc_col, ascending=False).index[:250].tolist()
        # mg = mygene.MyGeneInfo()
        # h_genes = mg.querymany(
        #     df_deg.index,species='human', scopes='symbol',as_dataframe=True
        #     )['symbol'].dropna().drop_duplicates()
        for lib in libs:
            enr = gp.enrichr(
                _genes, gene_sets=lib, organism=species, no_plot=True,
                )
            enr.res2d.to_excel(writer, sheet_name=lib+'_Enrichr')
            # GSEA
            # prerank_data = df_deg[[fc_col]]
            # prerank_data = prerank_data.merge(
            #     h_genes.to_frame(), left_index=True, right_index=True)
            # prerank_data = prerank_data.drop_duplicates().set_index('symbol')
            # gsea = gp.prerank(prerank_data,lib, no_plot=True)
            # gsea.res2d.to_excel(writer, sheet_name=lib+'_GSEA')
    writer.save()