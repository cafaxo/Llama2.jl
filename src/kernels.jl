
@kernel function rmsnorm_kernel_optimized!(o, x, weight, length_x)
    local_idx = @index(Local, Linear)
    global_idx = @index(Global, Linear)
    group_size = @groupsize()[1]

    # Shared memory for partial sums and final ss value
    shared_mem = @localmem Float32 (group_size)
    
    ss = 0.0f0
    
    # Only the first workgroup calculates the normalization factor
    # Calculate partial sum of squares
    @inbounds for j in local_idx:group_size:length_x
        ss += x[j] * x[j]
    end
    
    shared_mem[local_idx] = ss
    @synchronize
    
    # Parallel reduction
    s = group_size ÷ 2
    while s > 0
        if local_idx <= s
            shared_mem[local_idx] += shared_mem[local_idx + s]
        end
        @synchronize
        s ÷= 2
    end
    
    # Final calculation
    if local_idx == 1
        ss = shared_mem[1]
        ss /= length_x
        ss += 1f-6
        ss = 1f0 / sqrt(ss)
    end
    # Broadcast ss to all workgroups
    if local_idx == 1
        @inbounds shared_mem[1] = ss
    end

    @synchronize

    # All threads read the broadcasted ss value
    sall = shared_mem[1]
    @synchronize

    # Each thread calculates its corresponding output
    if global_idx <= length_x
        @inbounds o[global_idx] = weight[global_idx] * (sall * x[global_idx])
    end
end

function rmsnorm!(o::AbstractVector, x::AbstractVector, weight::AbstractVector)
    length_x = length(x)
    backend = KernelAbstractions.get_backend(o)
    
    # Choose an appropriate group size (e.g., 256)
    group_size = 256
    
    kernel! = rmsnorm_kernel_optimized!(backend, group_size)
    kernel!(o, x, weight, length_x, ndrange=length_x, )
end

@kernel function rope_kernel_v2!(x, @Const(pos), @Const(head_size_div2), @Const(n_heads), @Const(theta_scale), @Const(freq_scale))
    i, head = @index(Global, NTuple)
    
    if i <= head_size_div2 && head <= n_heads
        theta_base = freq_scale * (pos - 1)
        
        idx = 2 * (i - 1)
        real_part = x[idx + 1, head]
        imag_part = x[idx + 2, head]
        
        theta = theta_base * (theta_scale ^ (i - 1))
        c, s = cos(theta), sin(theta)

        new_real = muladd(real_part, c, -imag_part * s)
        new_imag = muladd(real_part, s, imag_part * c)
        
        x[idx + 1, head] = new_real
        x[idx + 2, head] = new_imag
    end
end

function rope!(x::AbstractMatrix{Float32}, pos::Int, freq_base::Float32)
    head_size, n_heads = size(x)
    head_size_div2 = head_size ÷ 2
    freq_scale = 1.0f0

    theta_scale = freq_base ^ (-inv(Float32(head_size_div2)))

    workgroup_size = (16, 16)  # Adjust these values based on your hardware
    kernel! = rope_kernel_v2!(KernelAbstractions.get_backend(x), workgroup_size)
    
    kernel!(x, pos, head_size_div2, n_heads, theta_scale, freq_scale, ndrange=(head_size_div2, n_heads))
end

@kernel function attention_weights_kernel!(att, @Const(key_cache), @Const(q), n_gqa)
    t, h = @index(Global, NTuple)

    if t <= size(att, 1) && h <= size(att, 2)
        key_h = (h - 1) ÷ n_gqa + 1
        s = 0f0

        @inbounds for j in 1:size(q, 1)
            s += q[j, h] * key_cache[j, key_h, t]
        end
        @inbounds att[t, h] = s
    end
end

function attention_weights!(att::AbstractArray, key_cache::AbstractArray, q::AbstractArray)
    n_gqa = size(q, 2) ÷ size(key_cache, 2)

    kernel! = attention_weights_kernel!(KernelAbstractions.get_backend(att))
    kernel!(att, key_cache, q, n_gqa, ndrange=size(att))
end

@kernel function combine_values_kernel!(xb, @Const(value_cache), @Const(att), n_gqa)
    i, h = @index(Global, NTuple)
  
    if i <= size(xb, 1) && h <= size(xb, 2)
        s = 0.0f0
        value_h = 1 + (h - 1) ÷ n_gqa
        
        for t in 1:size(att, 1)
            s += att[t, h] * value_cache[t, i, value_h]
        end
        
        xb[i, h] = s
    end
end

function combine_values!(xb::AbstractMatrix, value_cache::AbstractArray, att::AbstractMatrix)
    n_gqa = size(att, 2) ÷ size(value_cache, 3)
  
    kernel! = combine_values_kernel!(KernelAbstractions.get_backend(xb))
    kernel!(xb, value_cache, att, n_gqa, ndrange=size(xb))
end

@kernel function softmax_kernel_v2!(att, @Const(attention_maximum))
    i, h = @index(Global, NTuple)
    local_idx = @index(Local)
    group_size = @groupsize()[1]

    if h <= size(att, 2)
        max_val = attention_maximum[h]
        exp_sum = 0.0f0
        
        # Shared memory for partial sums
        shared_mem = @localmem Float32 (group_size)

        # Calculate partial exp sum
        for t in local_idx:group_size:size(att, 1)
            exp_val = exp(att[t, h] - max_val)
            exp_sum += exp_val
            att[t, h] = exp_val
        end

        shared_mem[local_idx] = exp_sum
        @synchronize

        # Parallel reduction for exp_sum
        s = group_size ÷ 2
        while s > 0
            if local_idx <= s
                shared_mem[local_idx] += shared_mem[local_idx + s]
            end
            @synchronize
            s ÷= 2
        end

        @synchronize
        exp_sum = shared_mem[1]

        # Normalize
        for t in local_idx:group_size:size(att, 1)
            att[t, h] /= exp_sum
        end
    end
end

@views function softmax_for!(att::AbstractMatrix)
    pos_size, n_heads = size(att) 
    backend = KernelAbstractions.get_backend(att)
    att_max = reshape(maximum(att, dims=1), :)

    group_size = 32  # Adjust based on your hardware
    kernel! = softmax_kernel_v2!(backend, (group_size, 1))
    kernel!(att, att_max, ndrange=(group_size, n_heads), )
end

silu(x) = x*σ(x) # Basically: x * (1f0 / (1f0 + exp(-x)))
