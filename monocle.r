library('monocle3')
library('ggplot2')
library(devtools)
library('dplyr')
library('ggplot2')
library('circlize')
library('RColorBrewer')
library('ComplexHeatmap')

# devtools::install_github("SONGDONGYUAN1994/PseudotimeDE")
# remotes::install_github("scfurl/m3addon")

setwd('A:/xiaoling_scrna/monocle/')
expr = read.table('expr.txt',sep='\t', row.names = 1, header=T)
meta = read.table('cell_meta.txt', sep = '\t', row.names = 1, header=T)
# cds = align_cds(cds, alignment_group = 'Sample')

########## Calculating pseudotime ##########
  geneanno = data.frame(colnames(expr),row.names=colnames(expr))
  colnames(geneanno)=c('gene_short_name')
  cds = new_cell_data_set(
    2^as.matrix(t(expr))-1,
    cell_metadata = meta,
    gene_metadata = geneanno)
  cds = preprocess_cds(cds, num_dim = 100, norm_method = 'none')
  
  cds = reduce_dimension(
    cds, max_components = 50, reduction_method = 'UMAP', umap.min_dist = 0.3)
  cds = cluster_cells(cds, k=25)
  cds = learn_graph(
    cds, close_loop = T, verbose = T, use_partition = T,learn_graph_control = list(ncenter=20))
  
  pdf(file='Monocle_leiden.pdf')
  plot_cells(cds, color_cells_by = "leiden", group_label_size = 5,
label_cell_groups = F,
label_groups_by_cluster = F,
show_trajectory_graph = F,
cell_size = 1)
  dev.off()
  
  pdf(file='Monocle_sample.pdf')
  plot_cells(cds, color_cells_by = "Sample", group_label_size = 5,
label_cell_groups = F,
label_groups_by_cluster = F,
show_trajectory_graph = F,
cell_size = 1)
  dev.off()
  
  cds = order_cells(cds)
  saveRDS(cds, file = 'monocle.Rdata')
