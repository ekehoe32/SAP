# test script for multi gpu SAP

# imports
using Pkg
env_path = "/your/SAP/path/here/";
cd(env_path)
Pkg.activate(".")
using Revise, LinearAlgebra, SAP, BenchmarkTools, Distributed, Plots, JLD


# ---- Test Data Set ---- #
m = 20_000;
n = 100;
X = gen_trig_moment_curve(Float64, m, n);
dataset_name = "Trigonometric_Moment_Curve_100"
# ----------------------- #


# ---- User parameters ---- #
dims = collect(1:1:20); # specify your embedding dimensions
iterations = 100;
alpha = .01;
seed = 0;
gamma = .01;
# ------------------------- #


# ---- Run SAP on gpus ---- #

# calculate batching info
problem_size = 2*binomial(m, 2)*n;
golden_number = 4_031_615_000;
ideal_batches = ((problem_size - 1) ÷ golden_number) + 1;
parts = ceil(Int64, inv_binomial(ideal_batches));
device_list = collect(0:15);
num_batches = pairwiseNumBatches(parts);

# setup cluster
addprocs(min(length(device_list),num_batches))
@everywhere env_path = "/your/SAP/path/here/";
@everywhere cd(env_path)
@everywhere using Pkg
@everywhere Pkg.activate(".")
@everywhere using CUDA, SAP, CSV, DataFrames

# run SAP algorithm
@time projections, min_secants, min_norms = SAP_multigpu(X, dims, alpha, iterations, seed, parts, device_list; random=false, subsample=gamma);
timing = 12345 # store the time here and save all below

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
            length(device_list),
    )

# load data (only if exited)
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
xlabel!("Dimension")
ylabel!("2-norm")
title!(string("SAP Minimum Projected Norms for \n", dataset_name))
savefig(p, string(dataset_name, "_SAP_min_norms.png"))

# plot heatmap of minimum min_secants
gr()
iter_scale = 1:iterations;
p = heatmap(iter_scale, dims, min_norms, size = (750, 600))
xticks!(1:(iterations÷10):iterations)
yticks!(dims)
xlabel!("Iteration")
ylabel!("Dimension")
title!(string("SAP Minimum Projected Norms for \n", dataset_name))
savefig(p, string(dataset_name, "_SAP_min_norms_heatmap.png"))

# plot interative embedding
Z = X*projections[3];
dim = dims[3];
U = Z;
plotly()
p = scatter(U[:,1],U[:,2], U[:,3],
            title= string("SAP Embedding of ", dataset_name,  " <br>in $dim-Dimensional Space"),
            linewidth = 2, 
            legend=false,
            size = (1000, 1000),
            markersize = 1)
savefig(p, string(dataset_name,"_SAP_$dim.html"))

# clear remaining device resources
for worker in workers()
    remotecall(device_reset!, worker)
end

# free cluster workers
rmprocs(workers())