# Ex1

#The fastcluster package is not necessary.  By loading the
# fastcluster package, the fastcluster::hclust() function 
# automatically replaces the slower stats::hclust() function
# whenever hclust() is called.
library(TDAmapper)
# install.packages("fastcluster") 
require(fastcluster) 

m1 <- mapper1D(
  distance_matrix = dist(data.frame( x=2*cos(0.5*(1:100)), y=sin(1:100) )),
  filter_values = 2*cos(0.5*(1:100)),
  num_intervals = 10,
  percent_overlap = 55,
  num_bins_when_clustering = 10)

# filter valueには好きな値が使えるみたい

# The igraph package is necessary to view simplicial complexes
# (undirected graph) resulting from mapper1D().
# install.packages("igraph") 
library(igraph)

g1 <- graph.adjacency(m1$adjacency, mode="undirected")
plot(g1, layout = layout.auto(g1) )
