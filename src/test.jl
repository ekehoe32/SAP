# test script for SAP on gpu

using Revise, LinearAlgebra, CUDA, SAP


m_X = 3000;
m_Y = 3000;
n = 30;
k = 10;
X = rand(m_X, n);
Y = rand(m_Y, n);
N = binomial(m, 2);
S = secants(X, Y)
P, R = qr(rand(n, n));
P = P[:,1:k];


# test secants
function test_secants(S)
    c = 1;
    for j = 1:m_Y
        for i = 1:m_X
            s = X[i, :] - Y[j, :];
            s /= norm(s);
            println(S[c, :] ≈ s);
            #println(norm(S[I, :])≈ 1);
            c += 1;
        end
    end
end

test_secants(S)


# test indexing
function tri(n::Int)
    return (n * (n + 1)) ÷ 2
end

for i = 1:N
    J = ceil(Int, (-1 + sqrt(1 + 8 * i)) / 2) + 1;
    I = J + i - 1 - tri(J - 1);

    println((I,J))
end

# test projections
for i = 1:100
    result = min_project(P, S);
end

result = remotecall_fetch(min_project, 4, P, S);

results = Array{Any, 1}(undef, 20);

for i = 1:20
    results[i] = remotecall(min_project, 4,P, S);
end

for i = 1:20
    results[i] = fetch(results[i]);
end

remotecall_fetch(device, 4)


# test multivec norm
dims = [10, 5, 3, 2, 1]
P = nothing;
seed=0;
rng = MersenneTwister(seed);
for i = 1:length(dims)
    Q = qr(rand(rng, n, n)).Q;
    Q = Q[:,1:dims[i]];
    Q = Array(Q);
    if i == 1
        P = Q;
    else
        P = [P Q];
    end
end

N = 9000000;
n= 30;
num_dims = length(dims);
proj = S*P;
proj = CuArray(proj);
nrms = typeof(proj)(undef,N, num_dims);
# generate dimension ticks
dim_ticks = Array{Int64, 1}(undef, length(dims)+1);
for i = 1:(length(dims)+1)
    dim_ticks[i] = sum(dims[1:(i-1)]) + 1;
end
dim_ticks = CuArray(dim_ticks);

threads_per_block_x = 512;
threads_per_block_y = 2;
num_blocks_x = (N - 1) ÷ threads_per_block_x + 1;
num_blocks_y = (num_dims - 1) ÷ threads_per_block_y + 1;

@cuda blocks=(num_blocks_x, num_blocks_y) threads=(threads_per_block_x, threads_per_block_y) multi_vecnorm!(proj, N, nrms, dim_ticks, num_dims);

S = CuArray(S);
T = typeof(S);
s = T(undef, N, 1);

threads_per_block = 1024;
num_blocks = (N - 1) ÷ threads_per_block + 1;

@cuda blocks=num_blocks threads=threads_per_block vecnorm!(S, N, n, s);

nrms_d = Array(nrms);
dim_ticks = Array(dim_ticks);
proj = Array(proj);
s_d = Array(s);
S_d = Array(S);

all(s_d .≈ f1(S_d,2))


# create dimension slices
dim_slices = Array{UnitRange{Int64}}(undef, num_dims);

for i = 1:length(dims)
    dim_slices[i] = (sum(dims[1:(i-1)])+1):sum(dims[1:i]);
end

for i = 1:num_dims
    println(all(nrms_d[:, i] .≈ f1(proj[:,dim_slices[i]], 2))) 
end


# compute minimum projection
I₀ = Array(argmin(nrms, dims=1));
indices = Array{Int64,1}(undef, num_dims);
for i = 1:num_dims
    indices[i] = I₀[i][1]
end
s = Array(S[indices, :]);
nrm = Array(nrms[indices]);