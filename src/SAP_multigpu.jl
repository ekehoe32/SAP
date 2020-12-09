# test script for multi gpu SAP

# imports
using Pkg
env_path = "/s/a/home/ekehoe/.julia/mypackages/SAP/";
cd(env_path)
Pkg.activate(".")
using Revise, LinearAlgebra, SAP, BenchmarkTools, Distributed, Plots, JLD, CSV, DataFrames, StatsPlots

# ---- Testing ----#
# params
#m = 20_000;
#n = 100;

# initialize test data set
#X = gen_trig_moment_curve(Float64, m, n);
# ---------------- #

# specific labeling info for Indian Pines
labels =  DataFrame(CSV.File("/home/apete41/indian_pines_labels.csv", header=false));
label_dict = Dict(0 => "Unknown",
                  1 => "Alfalfa", 
                  2 => "Corn-notill",
                  3 => "Corn-mintill",
                  4 => "Corn",
                  5 => "Grass-pasture",
                  6 => "Grass-trees",
                  7 => "Grass-pasture-mowed",
                  8 => "Hay-windrowed",
                  9 => "Oats",
                  10 => "Soybean-notill",
                  11 => "Soybean-mintill",
                  12 => "Soybean-clean",
                  13 => "Wheat",
                  14 => "Woods",
                  15 => "Building-Grass-Trees-Drives",
                  16 => "Stone-Steel-Towers")

label_dict_q = Dict(0 => "Unknown",
                  1 => "Alfalfa", 
                  2 => "Corn",
                  3 => "Corn",
                  4 => "Corn",
                  5 => "Grass",
                  6 => "Grass",
                  7 => "Grass",
                  8 => "Hay-windrowed",
                  9 => "Oats",
                  10 => "Soybean",
                  11 => "Soybean",
                  12 => "Soybean",
                  13 => "Wheat",
                  14 => "Woods",
                  15 => "Building-Grass-Trees-Drives",
                  16 => "Stone-Steel-Towers")
labels[!, :object] = [label_dict[obj] for obj in labels[!, :Column1]]
labels[!, :object_q] = [label_dict_q[obj] for obj in labels[!, :Column1]]

# specific labeling info for word2vec
labels = df[:, [:words]];

# Load data onto main thread to estimate problem_size
dataset_name = "COVID Word2Vec Dataset";
df = DataFrame(CSV.File("/home/apete41/word2vec_covid.csv"; delim=',', header=true));
X = Array{Float32, 2}(df[:,2:end-1]);
#X = Array(X');
#min_val = minimum(X);
#tf = X .≈ min_val;
#bad_cols = prod(tf, dims=1);
#X = X[:, .!bad_cols[:]];
m, n = size(X);

# ---- Run SAP on gpus ---- #
# params
iterations = 100;
problem_size = 2*binomial(m, 2)*n;
golden_number = 4_031_615_000;
ideal_batches = ((problem_size - 1) ÷ golden_number) + 1;
parts = ceil(Int64, inv_binomial(ideal_batches));
device_list = collect(0:15);
num_batches = pairwiseNumBatches(parts);
dims = collect(1:1:20);

# setup cluster
addprocs(min(length(device_list),num_batches))
@everywhere env_path = "/s/a/home/ekehoe/.julia/mypackages/SAP/";
@everywhere cd(env_path)
@everywhere using Pkg
@everywhere Pkg.activate(".")
@everywhere using CUDA, SAP, CSV, DataFrames

# SAP algorithm
#@time projections, min_secants, min_norms = SAP_multigpu_new(X, dims, .01, 2, 0, parts, device_list, random=true);
@time projections, min_secants, min_norms = SAP_multigpu(X, dims, .01, iterations, 0, parts, device_list; random=false, subsample=.01);
timing = 58171.395116

# save data
save(string(dataset_name, "_SAP.jld"),
 "projections",
  projections,
  "min_secants",
  min_secants,
  "min_norms",
  min_norms,
  "timing",
  timing,
  "dims",
  dims,
  "iterations",
  iterations,
  "ngpus",
  length(workers()),
  )

# load data
projections = load(string(dataset_name, "_SAP.jld"), "projections");
min_secants = load(string(dataset_name, "_SAP.jld"), "min_secants");
min_norms = load(string(dataset_name, "_SAP.jld"), "min_norms");
timing = load(string(dataset_name, "_SAP.jld"), "timing");
dims = load(string(dataset_name, "_SAP.jld"), "dims");
iterations = load(string(dataset_name, "_SAP.jld"), "iterations");
ngpus = load(string(dataset_name, "_SAP.jld"), "ngpus");

# plot minimum secant norms across dimension
gr()
p = scatter(dims, min_norms[:,end], yscale=:log10, xticks=0:1:20, legend=false)
#title!("Norm of minimum projected secant")
xlabel!("Dimension")
ylabel!("2-norm")
title!(string("SAP Minimum Projected Norms for \n", dataset_name))
savefig(p, string(dataset_name, "_SAP_min_norms.png"))

# plot heatmap of minimum min_secants
gr()
iter_scale = 1:iterations;
p = heatmap(iter_scale, dims, min_norms, size = (750, 600))
xticks!(0:10:100)
yticks!(dims)
xlabel!("Iteration")
ylabel!("Dimension")
title!(string("SAP Minimum Projected Norms for \n", dataset_name))
savefig(p, string(dataset_name, "_SAP_min_norms_heatmap.png"))

# plot embedding
Z = X*projections[3];
dim = dims[3];
#U = svd(Z).U;
U = Z;
plotly()
p = scatter(U[:,1],U[:,2], U[:,3],
            title= string("SAP Embedding of ", dataset_name,  " <br>in $dim-Dimensional Space"),
            linewidth = 2, 
            legend=false,
            size = (1000, 1000),
            markersize = 1)
savefig(p, string(dataset_name,"_SAP_$dim.html"))


# plot embedding /w labels
Z = X*projections[3];
dim = dims[3];
#U = svd(Z).U;
df = DataFrame(Z);
df = hcat(df, labels, makeunique=true);
plotly()
@df df scatter(:x1, :x2, :x3,
            title= string("SAP Embedding of ", dataset_name,  "<br> in $dim-Dimensional Space"),
            group = :object_q,
            size = (1000, 1000),
            legend=:right,
            markersize = 3)
savefig(string(dataset_name,"_SAP_type_q_$dim.html"))

for worker in workers()
    remotecall(device_reset!, worker)
end

# free cluster workers
rmprocs(workers())