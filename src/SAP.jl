module SAP

using LinearAlgebra, CUDA, Random, Distributed, Combinatorics, ProgressMeter

export secant_kernel!, secants, min_project, SAP_multigpu, gen_trig_moment_curve, pairwiseNumBatches, multi_vecnorm!, vecnorm!, inv_binomial, SAP_multigpu_new


# helper gpu functions
function tri(n::Int)
    return (n * (n + 1)) ÷ 2
end

function inv_binomial(n)
    r = (1 + sqrt(1+8*n))/2;
    return r;
end

function arr2cuarray(::Union{Array{Float64, 2}, Array{Float32, 2}})
    return CuArray(X);
end

function pairwiseNumBatches(parts::Int64)
    return binomial(parts+1, 2)
end
#

# gpu kernel for computing secants
function secant_kernel!(
    X::Union{CuDeviceArray{Float64,2}, CuDeviceArray{Float32,2}},
    n::Int,
    S::Union{CuDeviceArray{Float64,2}, CuDeviceArray{Float32,2}},
    N::Int,
    )

    # set kernel indices and dimensions
    stride_x = gridDim().x * blockDim().x;
    stride_y = gridDim().y * blockDim().y;
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x;
    idy = (blockIdx().y - 1) * blockDim().y + threadIdx().y;

    for ii = idx:stride_x:N
        # find 2d index
        J = ceil(Int, (-1 + sqrt(1 + 8 * ii)) / 2) + 1;
        I = J + ii - 1 - tri(J - 1);

        for jj = idy:stride_y:n
            # set secant value
            @inbounds S[ii, jj] = X[I, jj] - X[J, jj];
        end
    end
    


    return nothing
end

# gpu kernel for computing secants
function secant_kernel!(
    X::Union{CuDeviceArray{Float64,2}, CuDeviceArray{Float32,2}},
    m::Int,
    n::Int,
    Y::Union{CuDeviceArray{Float64,2}, CuDeviceArray{Float32,2}},
    S::Union{CuDeviceArray{Float64,2}, CuDeviceArray{Float32,2}},
    N::Int,
    )

    # set kernel indices and dimensions
    stride_x = gridDim().x * blockDim().x;
    stride_y = gridDim().y * blockDim().y;
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x;
    idy = (blockIdx().y - 1) * blockDim().y + threadIdx().y;

    for ii = idx:stride_x:N
        # find 2d index
        r = ii % m
        i = (r == 0) ? m : r
        j = ((ii - i) ÷ m) + 1;

        for jj = idy:stride_y:n
            # set secant value
            @inbounds S[ii, jj] = X[i, jj] - Y[j, jj];
        end
    end
    return nothing
end

# gpu kernel for powers
function pow!(x::CuDeviceArray, n::Int)
    stride = gridDim().x * blockDim().x
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    for k = id:stride:length(x)
        @inbounds x[k] = CUDA.pow(x[k], n)
    end
end

# gpu kernel for row-wise norms of matrix
function vecnorm!(
    x::Union{CuDeviceArray{Float64, 2}, CuDeviceArray{Float32, 2}},
    m::Int,
    n::Int,
    z::Union{CuDeviceArray{Float64,2}, CuDeviceArray{Float32,2}},
)
    stridex = gridDim().x * blockDim().x;
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x;
    for i = idx:stridex:m
            s = 0.0;
            c = 0.0;
            for k = 1:n
                y = CUDA.pow(x[i, k], 2) - c;
                t = s + y;
                c = (t - s) - y;
                s = t;
            end
            s = sqrt(s);
            @inbounds z[i] = s;
    end

    return nothing
end

# gpu kernel for row-wise norms of sub matrices given by column slices
function multi_vecnorm!(
    x::Union{CuDeviceArray{Float64, 2}, CuDeviceArray{Float32, 2}},
    m::Int,
    z::Union{CuDeviceArray{Float64, 2}, CuDeviceArray{Float32, 2}},
    dim_ticks::CuDeviceArray{Int64, 1},
    l::Int,
)
    stridex = gridDim().x * blockDim().x;
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x;
    idy = (blockIdx().y - 1) * blockDim().y + threadIdx().y;
    for i = idx:stridex:m
        if idy <= l
            s = 0.0;
            c = 0.0;

            p = dim_ticks[idy];
            q = dim_ticks[idy+1]-1;

            for k = p:q
                y = CUDA.pow(x[i, k], 2) - c;
                t = s + y;
                c = (t - s) - y;
                s = t;
            end
            s = sqrt(s);
            @inbounds z[i, idy] = s;
        end
    end

    return nothing
end

# gpu kernel for normalize rows by a vector
function vecnormalize!(
    x::Union{CuDeviceArray{Float64, 2}, CuDeviceArray{Float32, 2}},
    m::Int64,
    n::Int64,
    y::Union{CuDeviceArray{Float64, 2}, CuDeviceArray{Float32, 2}},
)
    stridex = gridDim().x * blockDim().x;
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x;
    stridey = gridDim().y * blockDim().y;
    idy = (blockIdx().y - 1) * blockDim().y + threadIdx().y;

    for i = idx:stridex:m
        for j = idy:stridey:n
            @inbounds x[i, j] /= y[i, 1];
        end
    end

    return nothing
end

"""
    secants(X::Union{Array{Float64, 2}, Array{Float32, 2}})

Returns the pairwise unit length secants from the row vectors in X. e.g. sᵢⱼ = (X⁽ⁱ⁾ - X⁽ʲ⁾) / ||X⁽ⁱ⁾ - X⁽ʲ⁾|| for i ≠ j
"""
function secants(X::Union{Array{Float64, 2}, Array{Float32, 2}})

    # grab dimension of data
    m, n = size(X);
    N = binomial(m, 2);
    tt = typeof(X[1])

    # assign to GPU
    X_cu = CuArray(X);
    S_cu = CuArray{tt, 2}(undef, N, n);
    s = CuArray{tt, 2}(undef, N, 1)

    # calculate secants
    threads_per_block_x = 256;
    threads_per_block_y = 4;
    num_blocks_x = (N - 1) ÷ threads_per_block_x + 1;
    num_blocks_y = (n - 1) ÷ threads_per_block_y + 1;

    @cuda blocks=(num_blocks_x, num_blocks_y) threads=(threads_per_block_x, threads_per_block_y) secant_kernel!(X_cu, n, S_cu, N);
    
    CUDA.synchronize()

    # normalize secants
    threads_per_block = 1024;
    num_blocks = (N - 1) ÷ threads_per_block + 1;

    @cuda blocks=num_blocks threads=threads_per_block vecnorm!(S_cu, N, n, s);

    CUDA.synchronize()
    
    @cuda blocks=(num_blocks_x, num_blocks_y) threads=(threads_per_block_x, threads_per_block_y) vecnormalize!(S_cu, N, n, s);

    CUDA.synchronize()

    # transfer to host
    S = Array(S_cu);

    # free device resources
    X_cu = nothing;
    S_cu = nothing;
    s = nothing

    return S
end

"""
    secants(X::Union{Array{Float64, 2}, Array{Float32, 2}}, Y::Union{Array{Float64, 2}, Array{Float32, 2}})

Returns the pairwise unit length secants between the row vectors in X and Y. e.g. sᵢⱼ = (X⁽ⁱ⁾ - Y⁽ʲ⁾) / ||X⁽ⁱ⁾ - Y⁽ʲ⁾|| ∀ i,j
"""
function secants(X::Union{Array{Float64, 2}, Array{Float32, 2}}, Y::Union{Array{Float64, 2}, Array{Float32, 2}})

    # grab dimension of data
    m_X, n = size(X);
    m_Y = size(Y)[1];
    N = m_X*m_Y;
    tt = typeof(X[1])

    # assign to GPU
    X_cu = CuArray(X);
    Y_cu = CuArray(Y);
    S_cu = CuArray{tt, 2}(undef, N, n);
    s = CuArray{tt, 2}(undef, N, 1)

    # calculate secants
    threads_per_block_x = 256;
    threads_per_block_y = 4;
    num_blocks_x = (N - 1) ÷ threads_per_block_x + 1;
    num_blocks_y = (n - 1) ÷ threads_per_block_y + 1;

    @cuda blocks=(num_blocks_x, num_blocks_y) threads=(threads_per_block_x, threads_per_block_y) secant_kernel!(X_cu, m_X, n, Y_cu, S_cu, N);
    
    CUDA.synchronize()

    # normalize secants
    threads_per_block = 1024;
    num_blocks = (N - 1) ÷ threads_per_block + 1;

    @cuda blocks=num_blocks threads=threads_per_block vecnorm!(S_cu, N, n, s);

    CUDA.synchronize()
    
    @cuda blocks=(num_blocks_x, num_blocks_y) threads=(threads_per_block_x, threads_per_block_y) vecnormalize!(S_cu, N, n, s);

    CUDA.synchronize()

    # transfer to host
    S = Array(S_cu);

    # free gpu vars
    X_cu = nothing;
    Y_cu = nothing;
    S_cu = nothing;
    s = nothing;

    return S
end

"""
min_project(P::Union{CuArray{Float64, 2}, CuArray{Float32, 2}}, S::Union{Array{Float64, 2}, Array{Float32, 2}})

Returns the minimum 2-norm projection P*S⁽ⁱ⁾ and the associated norm over rows S⁽ⁱ⁾ in S.
"""
function min_project(P::Union{Array{Float64, 2}, Array{Float32, 2}}, S::Union{Array{Float64, 2}, Array{Float32, 2}})
    # transfer projection to device
    P_cu = CuArray(P);
    
    # calculate dimensions
    m, n = size(S);
    k = size(P)[2];
    
    # load S onto gpu
    S_cu = CuArray(S);

    # compute projections
    proj = S_cu*P_cu;
    nrms = typeof(proj)(undef,m, 1);

    # free device memory
    S_cu = nothing
    P_cu = nothing
    
    # compute norms
    threads_per_block = 1024;
    num_blocks = (m - 1) ÷ threads_per_block + 1;

    @cuda blocks=num_blocks threads=threads_per_block vecnorm!(proj, m, k, nrms);

    CUDA.synchronize();

    # compute minimum projection
    nrms = Array(nrms);
    i₀ = argmin(nrms)[1]
    s = S[i₀, :];
    nrm = nrms[i₀];

    # free projections and norms
    proj = nothing
    nrms = nothing

    return [s, nrm]
end

function secant_project(X::Union{Array{Float64, 2}, Array{Float32, 2}}, P::Union{Array{Float64, 2}, Array{Float32, 2}}, slice::UnitRange{Int64}, dim_ticks::Array{Int64})
    
    # slice data
    x = X[slice, :];
    x = CuArray(x);
    P = CuArray(P);
    dim_ticks = CuArray(dim_ticks);

    # grab type
    T = typeof(x);

    # grab dimension of data
    n = size(X)[2];
    m = length(slice);
    N = binomial(m, 2);
    num_dims = length(dim_ticks) - 1;

    # assign to GPU
    S = T(undef, N, n);
    s = T(undef, N, 1)

    # calculate secants
    threads_per_block_x = 256;
    threads_per_block_y = 4;
    num_blocks_x = (N - 1) ÷ threads_per_block_x + 1;
    num_blocks_y = (n - 1) ÷ threads_per_block_y + 1;

    @cuda blocks=(num_blocks_x, num_blocks_y) threads=(threads_per_block_x, threads_per_block_y) secant_kernel!(x, n, S, N);
    
    CUDA.synchronize()

    # normalize secants
    threads_per_block = 1024;
    num_blocks = (N - 1) ÷ threads_per_block + 1;

    @cuda blocks=num_blocks threads=threads_per_block vecnorm!(S, N, n, s);

    CUDA.synchronize()
    
    @cuda blocks=(num_blocks_x, num_blocks_y) threads=(threads_per_block_x, threads_per_block_y) vecnormalize!(S, N, n, s);

    CUDA.synchronize()

    # free device resources
    s = nothing;
    x = nothing;
    
    # calculate dimensions
    m, n = size(S);
    k = size(P)[2];

    # compute projections
    proj = S*P;
    nrms = typeof(proj)(undef,N, num_dims);

    # compute norms
    threads_per_block_x = 512;
    threads_per_block_y = 2;
    num_blocks_x = (N - 1) ÷ threads_per_block_x + 1;
    num_blocks_y = (num_dims - 1) ÷ threads_per_block_y + 1;

    @cuda blocks=(num_blocks_x, num_blocks_y) threads=(threads_per_block_x, threads_per_block_y) multi_vecnorm!(proj, N, nrms, dim_ticks, num_dims);

    CUDA.synchronize();

    # compute minimum projection
    I₀ = Array(argmin(nrms, dims=1));
    indices = Array{Int64,1}(undef, num_dims);
    for i = 1:num_dims
        indices[i] = I₀[i][1]
    end
    s = S[indices, :];
    s = Array(s);
    nrm = nrms[indices, :];
    nrm = Array(nrm);
    nrm = diag(nrm);

    # free projections and norms
    proj = nothing
    nrms = nothing
    S = nothing

    return [s, nrm]
end

function secant_project(X::Union{CuArray{Float64, 2}, CuArray{Float32, 2}}, P::Union{CuArray{Float64, 2}, CuArray{Float32, 2}}, slice::UnitRange{Int64}, dim_ticks::CuArray{Int64})
    
    # slice data
    x = X[slice, :];
    #x = CuArray(x);
    #P = CuArray(P);
    #dim_ticks = CuArray(dim_ticks);

    # grab type
    T = typeof(x);

    # grab dimension of data
    n = size(X)[2];
    m = length(slice);
    N = binomial(m, 2);
    num_dims = length(dim_ticks) - 1;

    # assign to GPU
    S = T(undef, N, n);
    s = T(undef, N, 1)

    # calculate secants
    threads_per_block_x = 256;
    threads_per_block_y = 4;
    num_blocks_x = (N - 1) ÷ threads_per_block_x + 1;
    num_blocks_y = (n - 1) ÷ threads_per_block_y + 1;

    @cuda blocks=(num_blocks_x, num_blocks_y) threads=(threads_per_block_x, threads_per_block_y) secant_kernel!(x, n, S, N);
    
    CUDA.synchronize()

    # normalize secants
    threads_per_block = 1024;
    num_blocks = (N - 1) ÷ threads_per_block + 1;

    @cuda blocks=num_blocks threads=threads_per_block vecnorm!(S, N, n, s);

    CUDA.synchronize()
    
    @cuda blocks=(num_blocks_x, num_blocks_y) threads=(threads_per_block_x, threads_per_block_y) vecnormalize!(S, N, n, s);

    CUDA.synchronize()

    # free device resources
    s = nothing;
    x = nothing;
    
    # calculate dimensions
    m, n = size(S);
    k = size(P)[2];

    # compute projections
    proj = S*P;
    nrms = typeof(proj)(undef,N, num_dims);

    # compute norms
    threads_per_block_x = 512;
    threads_per_block_y = 2;
    num_blocks_x = (N - 1) ÷ threads_per_block_x + 1;
    num_blocks_y = (num_dims - 1) ÷ threads_per_block_y + 1;

    @cuda blocks=(num_blocks_x, num_blocks_y) threads=(threads_per_block_x, threads_per_block_y) multi_vecnorm!(proj, N, nrms, dim_ticks, num_dims);

    CUDA.synchronize();

    # compute minimum projection
    I₀ = Array(argmin(nrms, dims=1));
    indices = Array{Int64,1}(undef, num_dims);
    for i = 1:num_dims
        indices[i] = I₀[i][1]
    end
    s = S[indices, :];
    s = Array(s);
    nrm = nrms[indices, :];
    nrm = Array(nrm);
    nrm = diag(nrm);

    # free projections and norms
    proj = nothing
    nrms = nothing
    S = nothing

    return [s, nrm]
end

function secant_project(X::Union{Array{Float64, 2}, Array{Float32, 2}}, P::Union{Array{Float64, 2}, Array{Float32, 2}}, slice_x::UnitRange{Int64}, slice_y::UnitRange{Int64}, dim_ticks::Array{Int64})
    
    # slice data
    x = X[slice_x, :];
    y = X[slice_y, :];
    x = CuArray(x);
    y = CuArray(y);
    P = CuArray(P);
    dim_ticks = CuArray(dim_ticks);

    # grab type
    T = typeof(x);

    # grab dimension of data
    n = size(X)[2];
    m_x = length(slice_x);
    m_y = length(slice_y);
    N = m_x*m_y;
    num_dims = length(dim_ticks) - 1;

    # assign to GPU
    S = T(undef, N, n);
    s = T(undef, N, 1)

    # calculate secants
    threads_per_block_x = 256;
    threads_per_block_y = 4;
    num_blocks_x = (N - 1) ÷ threads_per_block_x + 1;
    num_blocks_y = (n - 1) ÷ threads_per_block_y + 1;

    @cuda blocks=(num_blocks_x, num_blocks_y) threads=(threads_per_block_x, threads_per_block_y) secant_kernel!( x, m_x, n, y, S, N);
    
    CUDA.synchronize()

    # normalize secants
    threads_per_block = 1024;
    num_blocks = (N - 1) ÷ threads_per_block + 1;

    @cuda blocks=num_blocks threads=threads_per_block vecnorm!(S, N, n, s);

    CUDA.synchronize()
    
    @cuda blocks=(num_blocks_x, num_blocks_y) threads=(threads_per_block_x, threads_per_block_y) vecnormalize!(S, N, n, s);

    CUDA.synchronize()

    # free device resources
    s = nothing;
    x = nothing;
    y = nothing;
    
    # calculate dimensions
    m, n = size(S);
    k = size(P)[2];

    # compute projections
    proj = S*P;
    nrms = typeof(proj)(undef,N, num_dims);

    # compute norms
    threads_per_block_x = 512;
    threads_per_block_y = 2;
    num_blocks_x = (N - 1) ÷ threads_per_block_x + 1;
    num_blocks_y = (num_dims - 1) ÷ threads_per_block_y + 1;

    @cuda blocks=(num_blocks_x, num_blocks_y) threads=(threads_per_block_x, threads_per_block_y) multi_vecnorm!(proj, N, nrms, dim_ticks, num_dims);

    CUDA.synchronize();

    # compute minimum projection
    I₀ = Array(argmin(nrms, dims=1));
    indices = Array{Int64,1}(undef, num_dims);
    for i = 1:num_dims
        indices[i] = I₀[i][1]
    end
    s = S[indices, :];
    s = Array(s);
    nrm = nrms[indices, :];
    nrm = Array(nrm);
    nrm = diag(nrm);

    # free projections and norms
    proj = nothing
    nrms = nothing
    S = nothing

    return [s, nrm]
end

function secant_project(X::Union{CuArray{Float64, 2}, CuArray{Float32, 2}}, P::Union{CuArray{Float64, 2}, CuArray{Float32, 2}}, slice_x::UnitRange{Int64}, slice_y::UnitRange{Int64}, dim_ticks::CuArray{Int64})
    
    # slice data
    x = X[slice_x, :];
    y = X[slice_y, :];
    #x = CuArray(x);
    #y = CuArray(y);
    #P = CuArray(P);
    #dim_ticks = CuArray(dim_ticks);

    # grab type
    T = typeof(x);

    # grab dimension of data
    n = size(X)[2];
    m_x = length(slice_x);
    m_y = length(slice_y);
    N = m_x*m_y;
    num_dims = length(dim_ticks) - 1;

    # assign to GPU
    S = T(undef, N, n);
    s = T(undef, N, 1)

    # calculate secants
    threads_per_block_x = 256;
    threads_per_block_y = 4;
    num_blocks_x = (N - 1) ÷ threads_per_block_x + 1;
    num_blocks_y = (n - 1) ÷ threads_per_block_y + 1;

    @cuda blocks=(num_blocks_x, num_blocks_y) threads=(threads_per_block_x, threads_per_block_y) secant_kernel!( x, m_x, n, y, S, N);
    
    CUDA.synchronize()

    # normalize secants
    threads_per_block = 1024;
    num_blocks = (N - 1) ÷ threads_per_block + 1;

    @cuda blocks=num_blocks threads=threads_per_block vecnorm!(S, N, n, s);

    CUDA.synchronize()
    
    @cuda blocks=(num_blocks_x, num_blocks_y) threads=(threads_per_block_x, threads_per_block_y) vecnormalize!(S, N, n, s);

    CUDA.synchronize()

    # free device resources
    s = nothing;
    x = nothing;
    y = nothing;
    
    # calculate dimensions
    m, n = size(S);
    k = size(P)[2];

    # compute projections
    proj = S*P;
    nrms = typeof(proj)(undef,N, num_dims);

    # compute norms
    threads_per_block_x = 512;
    threads_per_block_y = 2;
    num_blocks_x = (N - 1) ÷ threads_per_block_x + 1;
    num_blocks_y = (num_dims - 1) ÷ threads_per_block_y + 1;

    @cuda blocks=(num_blocks_x, num_blocks_y) threads=(threads_per_block_x, threads_per_block_y) multi_vecnorm!(proj, N, nrms, dim_ticks, num_dims);

    CUDA.synchronize();

    # compute minimum projection
    I₀ = Array(argmin(nrms, dims=1));
    indices = Array{Int64,1}(undef, num_dims);
    for i = 1:num_dims
        indices[i] = I₀[i][1]
    end
    s = S[indices, :];
    s = Array(s);
    nrm = nrms[indices, :];
    nrm = Array(nrm);
    nrm = diag(nrm);

    # free projections and norms
    proj = nothing
    nrms = nothing
    S = nothing

    return [s, nrm]
end

function SAP_multigpu_old(X::Union{Array{Float64, 2}, Array{Float32, 2}}, dims::Array{Int64,1}, alpha::Float64, max_iter::Int64, seed::Int64, parts::Int64, device_list::Array{Int64,1})
    # random number generator
    rng = MersenneTwister(seed);
    
    # get dimension and type of data
    T = typeof(X);
    m, n  = size(X);

    # params
    part_size = ceil(Int64, m/parts);

    # partion samples for batching
    num_batches = binomial(parts+1, 2);
    batches = 1:num_batches;
    sample_partitions = Iterators.partition(1:m, part_size);
    diagonal_partitions = zip(batches[1:parts], sample_partitions);
    pairwise_partitions = zip(batches[parts+1:end], combinations(collect(sample_partitions), 2));
    X_remote = Array{Any, 1}(undef, length(workers()));
    dim_slices_remote = Array{Any, 1}(undef, length(workers()));
    secant_set = Array{Any, 1}(undef, num_batches);
    min_set = Array{Any, 1}(undef, num_batches);
    min_vec = Array{Float64, 1}(undef, num_batches);

    # init outputs
    min_secants = T(undef, length(dims), n);
    min_norms = Array{Float64, 1}(undef, length(dims));
    projections = Array{T, 1}(undef, length(dims));

    # assign devices
    for (idx, worker) in enumerate(workers())
        d = device_list[idx];
        remotecall(device!, worker, d);
        @info "Worker $worker uses device $d."
    end

    # generate worker sequence
    worker_cycle = collect(zip(1:num_batches, Base.Iterators.cycle(workers())))

    # calculate secant set
    for (batch, slice) in diagonal_partitions
        x = X[slice, :];
        worker = worker_cycle[batch][2];
        @info "Sending secant batch $batch to worker $worker..."
        secant_set[batch] = remotecall(secants, worker,  x);
    end

    for (batch, slices) in pairwise_partitions
        x = X[slices[1], :];
        y = X[slices[2], :];
        worker = worker_cycle[batch][2];
        @info "Sending secant batch $batch to worker $worker..."
        secant_set[batch] = remotecall(secants, worker, x, y);
    end

    for batch = 1:num_batches
        @info "Fetching secant batch $batch..."
        secant_set[batch] = fetch(secant_set[batch]);
    end
    
    for (id, dim) = enumerate(dims)
        
        # intialize projection matrix
        @info "Initializing projection matrix for dimension = $dim..."
        P = qr(rand(rng, n, n)).Q;
        P = P[:,1:dim];

        # iterate projection alg
        for i = 1:max_iter
            @info "Starting iteration $i..."
            # calculate projections
            for batch = 1:num_batches
                worker = worker_cycle[batch][2];
                @info "Sending projection computation $batch to worker $worker..."
                min_set[batch] = remotecall(min_project, worker, P, secant_set[batch])
            end

            for batch = 1:num_batches
                @info "Fetching projection computation $batch..."
                result = fetch(min_set[batch]);
                min_vec[batch] = Float64(result[2]);
            end

            # calculate minimum of minimums
            @info "Finding minimum 2-norm projected secant..."
            i₀ = argmin(min_vec)
            min_norm = min_vec[i₀];
            @info "Minimum 2-norm = $min_norm."
            s₀ = min_set[i₀][1];
            
            if i < max_iter
                @info "Updating projection operator..."
                # update projection matrix
                ps = s₀'*P;
                pps = ((s₀'*P)*P')';
                j₀ = argmax(abs.(ps))[2];
                P[:, 2:j₀] = P[:, 1:(j₀ - 1)]
                P[:, 1] = pps;
                P = qr(P).Q;
                p₁ = (1 - alpha)*pps + alpha*(s₀ - pps);
                p₁ /= norm(p₁);
                P = Array(P);
                P[:, 1] = p₁;
            else
                @info "Storing results..."
                # store results
                projections[id] = P;
                min_secants[id, :] = s₀';
                min_norms[id] = min_norm;
            end
        end
    end

    return [projections, min_secants, min_norms]
end

function SAP_multigpu(X::Union{Array{Float64, 2}, Array{Float32, 2}}, dims::Array{Int64,1}, alpha::Float64, max_iter::Int64, seed::Int64, parts::Int64, device_list::Array{Int64,1}; random = false, subsample=.01)
    # random number generator
    rng = MersenneTwister(seed);
    
    # get dimension and type of data
    T = typeof(X);
    t = typeof(X[1]);
    m, n  = size(X);
    num_dims = length(dims);

    # create dimension slices
    dim_slices = Array{UnitRange{Int64}}(undef, num_dims);
    
    for i = 1:length(dims)
        dim_slices[i] = (sum(dims[1:(i-1)])+1):sum(dims[1:i]);
    end

    # generate dimension ticks
    dim_ticks = Array{Int64, 1}(undef, length(dims)+1);
    for i = 1:(length(dims)+1)
        dim_ticks[i] = sum(dims[1:(i-1)]) + 1;
    end

    # params
    part_size = ceil(Int64, m/parts);

    # partion samples for batching
    num_batches = binomial(parts+1, 2);
    batches = 1:num_batches;
    sample_partitions = Iterators.partition(1:m, part_size);
    diagonal_partitions = zip(batches[1:parts], sample_partitions);
    pairwise_partitions = zip(batches[parts+1:end], combinations(collect(sample_partitions), 2));
    #X_remote = Array{Any, 1}(undef, length(workers()));
    #P_remote = Array{Any, 1}(undef, length(workers()));
    #dim_ticks_remote = Array{Any, 1}(undef, length(workers()));
    #secant_set = Array{Any, 1}(undef, num_batches);
    min_set = Array{Any, 1}(undef, num_batches);
    min_vec = Array{Float64, 2}(undef, num_batches, num_dims);

    # init outputs
    min_secants = Array{t, 3}(undef, length(dims), max_iter, n);
    min_norms = Array{Float64, 2}(undef, length(dims), max_iter);
    projections = Array{T, 1}(undef, length(dims));

    # assign devices
    for (idx, worker) in enumerate(workers())
        d = device_list[idx];
        remotecall(device!, worker, d);
        @info "Worker $worker uses device $d."
    end

    # generate worker sequence
    #worker_cycle = collect(zip(1:num_batches, Base.Iterators.cycle(workers())))

    # load data set and slices onto each worker
    #for (i, worker) in enumerate(workers())
    #    @info "Storing dataset and slices onto worker $worker..."
    #    X_remote[i] = remotecall(arr2cuarray, worker, X);
    #    dim_ticks_remote[i] = remotecall(arr2cuarray, worker, dim_ticks);
    #end

    # intialize projection matrix
    if random
        @info "Initializing random projection matrices for dimensions: $dims..."
        P = nothing;
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
    else
        sp = 100*subsample;
        @info "Initializing with $sp% secant PCA projection for dimensions: $dims..."
        p = ceil(Int64, subsample*m);
        sid = randperm(rng, m)[1:p];
        x = X[sid, :];
        S = secants(x);
        U = svd(S').U;
        P = nothing;
        for i = 1:length(dims)
            Q = U[:,1:dims[i]];
            Q = Array(Q);
            if i == 1
                P = Q;
            else
                P = [P Q];
            end
        end
    end

    # send projections to workers on device
    #for (i, worker) in enumerate(workers())
    #    @info "Storing projections onto worker $worker..."
    #    P_remote[i] = remotecall(arr2cuarray, worker, P);
    #end

    # iterate projection alg
    for i = 1:max_iter
        
        @info "Starting iteration $i..."
        @info "Computing minimum projected secants..."
        # calculate minimum secants
        for (batch, slice) in diagonal_partitions
            #worker = worker_cycle[batch][2];
            #@info "Computing minimum secants for batch $batch on worker $worker..."
            min_set[batch] = @spawnat :any secant_project(X, P, slice, dim_ticks);
        end

        for (batch, slices) in pairwise_partitions
            #worker = worker_cycle[batch][2];
            #@info "Computing minimum secants for batch $batch on worker $worker..."
            min_set[batch] = @spawnat :any secant_project(X, P, slices[1], slices[2], dim_ticks);
        end

        @info "Fetching minimum projected secant results..."
        for batch = 1:num_batches
            #@info "Fetching minimum secant results for batch $batch..."
            min_set[batch] = fetch(min_set[batch]);
            min_vec[batch, :] = min_set[batch][2];
        end


        # calculate minimum of minimums
        @info "Finding minimum 2-norm projected secant..."
        I₀ = argmin(min_vec, dims=1)
        for (l, dim) in enumerate(dims)
            i₀ = I₀[l];
            min_norm = min_vec[i₀];
            min_norms[l, i] = min_norm;
            @info "Minimum 2-norm for dimension $dim = $min_norm."
            s₀ = min_set[i₀[1]][1][l,:];
            min_secants[l, i, :] = s₀';
            
            if i < max_iter
                Q = P[:,dim_slices[l]];
                @info "Updating projection operator for dimension: $dim..."
                # update projection matrix
                ps = s₀'*Q;
                pps = ((s₀'*Q)*Q')';
                j₀ = argmax(abs.(ps))[2];
                Q[:, 2:j₀] = Q[:, 1:(j₀ - 1)]
                Q[:, 1] = pps;
                Q = qr(Q).Q;
                p₁ = (1 - alpha)*pps + alpha*(s₀ - pps);
                p₁ /= norm(p₁);
                Q = Array(Q);
                Q[:, 1] = p₁;
                P[:,dim_slices[l]] = Q;
            else
                @info "Storing results..."
                # store results
                projections[l] = P[:,dim_slices[l]];
            end
        end
    end

    return [projections, min_secants, min_norms]
end

function SAP_multigpu_new(X::Union{Array{Float64, 2}, Array{Float32, 2}}, dims::Array{Int64,1}, alpha::Float64, max_iter::Int64, seed::Int64, parts::Int64, device_list::Array{Int64,1}; random=false)
    # get dimension and type of data
    T = typeof(X);
    m, n  = size(X);
    num_dims = length(dims);

    # create dimension slices
    dim_slices = Array{UnitRange{Int64}}(undef, num_dims);
    
    for i = 1:length(dims)
        dim_slices[i] = (sum(dims[1:(i-1)])+1):sum(dims[1:i]);
    end

    # generate dimension ticks
    dim_ticks = Array{Int64, 1}(undef, length(dims)+1);
    for i = 1:(length(dims)+1)
        dim_ticks[i] = sum(dims[1:(i-1)]) + 1;
    end

    # params
    part_size = ceil(Int64, m/parts);

    # partion samples for batching
    num_batches = binomial(parts+1, 2);
    batches = 1:num_batches;
    sample_partitions = Iterators.partition(1:m, part_size);
    diagonal_partitions = zip(batches[1:parts], sample_partitions);
    pairwise_partitions = zip(batches[parts+1:end], combinations(collect(sample_partitions), 2));
    X_remote = Array{Any, 1}(undef, length(workers()));
    P_remote = Array{Any, 1}(undef, length(workers()));
    dim_ticks_remote = Array{Any, 1}(undef, length(workers()));
    #secant_set = Array{Any, 1}(undef, num_batches);
    min_set = Array{Any, 1}(undef, num_batches);
    min_vec = Array{Float64, 2}(undef, num_batches, num_dims);

    # init outputs
    min_secants = T(undef, length(dims), n);
    min_norms = Array{Float64, 1}(undef, length(dims));
    projections = Array{T, 1}(undef, length(dims));

    # assign devices
    for (idx, worker) in enumerate(workers())
        d = device_list[idx];
        remotecall(device!, worker, d);
        @info "Worker $worker uses device $d."
    end

    # generate worker sequence
    worker_cycle = collect(zip(1:num_batches, Base.Iterators.cycle(enumerate(workers()))))

    # load data set and slices onto each worker
    for (i, worker) in enumerate(workers())
        @info "Storing dataset and slices onto worker $worker..."
        X_remote[i] = @spawnat worker X;
        dim_ticks_remote[i] = @spawnat worker dim_ticks;
    end

    # intialize projection matrix
    if random
        @info "Initializing projection matrices for dimensions: $dims..."
        rng = MersenneTwister(seed);
        P = nothing;
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
    else
    end

    # send projections to workers on device
    for (i, worker) in enumerate(workers())
        @info "Storing projections onto worker $worker..."
        P_remote[i] = @spawnat worker P;
    end

    # iterate projection alg
    for i = 1:max_iter
        
        @info "Starting iteration $i..."
        @info "Computing minimum projected secants..."
        # calculate minimum secants
        for (batch, slice) in diagonal_partitions
            (l, worker) = worker_cycle[batch][2];
            @info "Computing minimum secants for batch $batch on worker $worker..."
            min_set[batch] = @spawnat worker secant_project(fetch(X_remote[l]), fetch(P_remote[l]), slice, fetch(dim_ticks_remote[l]));
        end

        for (batch, slices) in pairwise_partitions
            (l, worker) = worker_cycle[batch][2];
            @info "Computing minimum secants for batch $batch on worker $worker..."
            min_set[batch] = @spawnat worker secant_project(fetch(X_remote[l]), fetch(P_remote[l]), slices[1], slices[2], fetch(dim_ticks_remote[l]));
        end

        @info "Fetching minimum projected secant results..."
        for batch = 1:num_batches
            #@info "Fetching minimum secant results for batch $batch..."
            min_set[batch] = fetch(min_set[batch]);
            min_vec[batch, :] = min_set[batch][2];
        end


        # calculate minimum of minimums
        @info "Finding minimum 2-norm projected secant..."
        I₀ = argmin(min_vec, dims=1)
        for (l, dim) in enumerate(dims)
            i₀ = I₀[l];
            min_norm = min_vec[i₀];
            @info "Minimum 2-norm for dimension $dim = $min_norm."
            s₀ = min_set[i₀[1]][1][l,:];
            
            if i < max_iter
                Q = P[:,dim_slices[l]];
                @info "Updating projection operator for dimension: $dim..."
                # update projection matrix
                ps = s₀'*Q;
                pps = ((s₀'*Q)*Q')';
                j₀ = argmax(abs.(ps))[2];
                Q[:, 2:j₀] = Q[:, 1:(j₀ - 1)]
                Q[:, 1] = pps;
                Q = qr(Q).Q;
                p₁ = (1 - alpha)*pps + alpha*(s₀ - pps);
                p₁ /= norm(p₁);
                Q = Array(Q);
                Q[:, 1] = p₁;
                P[:,dim_slices[l]] = Q;
            else
                @info "Storing results..."
                # store results
                projections[l] = P[:,dim_slices[l]];
                min_secants[l, :] = s₀';
                min_norms[l] = min_norm;
            end
        end
        
        if i < max_iter
            # send projections to workers on device
            for (l, worker) in enumerate(workers())
                @info "Storing projections onto worker $worker..."
                finalize(P_remote[l])
                P_remote[l] = @spawnat worker P;
            end
        end
    end

    return [projections, min_secants, min_norms]
end

function gen_trig_moment_curve(T::DataType, m::Int64, dim::Int64)
    X = Array{T, 2}(undef, m, dim);
    for (i,t)  = enumerate((2π/m):(2π/m):2π)
        r = t*(1:(dim÷2))
        c = cos.(r);
        s = sin.(r);
        X[i, 1:2:dim] = c;
        X[i, 2:2:dim] = s;
    end

    return X;
end
end # module


