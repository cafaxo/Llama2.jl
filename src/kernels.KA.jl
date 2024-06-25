using KernelAbstractions

@kernel function rope_kernel!(x, pos, head_size_div2, n_heads, theta_scale, freq_scale)
    head = @index(Global, Linear)
    
    if head <= n_heads
        theta = freq_scale * (pos - 1)
        
        for i in 1:head_size_div2
            real_part = x[2 * (i - 1) + 1, head]
            imag_part = x[2 * (i - 1) + 2, head]
            c = cos(theta)
            s = sin(theta)

            new_real = real_part * c - imag_part * s
            new_imag = real_part * s + imag_part * c
            
            x[2 * (i - 1) + 1, head] = new_real
            x[2 * (i - 1) + 2, head] = new_imag
            
            theta *= theta_scale
        end
    end
end

function rope!(x::AbstractMatrix{Float32}, pos::Int, freq_base::Float32)
    head_size, n_heads = size(x)
    head_size_div2 = head_size ÷ 2
    freq_scale = 1.0f0

    theta_scale = freq_base ^ (-inv(Float32(head_size_div2)))

    kernel! = rope_kernel!(KernelAbstractions.get_backend(x))
    kernel!(x, pos, head_size_div2, n_heads, theta_scale, freq_scale, ndrange=n_heads)
    KernelAbstractions.synchronize(KernelAbstractions.get_backend(x))
end

@kernel function attention_weights_kernel!(att, key_cache, q, n_gqa)
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
    KernelAbstractions.synchronize(KernelAbstractions.get_backend(att))

    return att
end

@kernel function combine_values_kernel!(xb, value_cache, att, n_gqa)
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
    KernelAbstractions.synchronize(KernelAbstractions.get_backend(xb))
end

@kernel function softmax_kernel!(att, attention_maximum)
    h = @index(Global)

    if h <= size(att, 2)
        max_val = attention_maximum[h]

        exp_sum = 0.0f0
        for t in 1:size(att, 1)
            exp_val = exp(att[t, h] - max_val)
            exp_sum += exp_val
            att[t, h] = exp_val
        end

        for t in 1:size(att, 1)
            att[t, h] /= exp_sum
        end
    end
end

@views function softmax_for!(att::AbstractMatrix, n_heads::Int)
    att_max = reshape(maximum(att, dims=1), 1, :)

    kernel! = softmax_kernel!(KernelAbstractions.get_backend(att))
    kernel!(att, att_max, ndrange=n_heads)
    KernelAbstractions.synchronize(KernelAbstractions.get_backend(att))
end

@kernel function rmsnorm_kernel!(o, x, weight, length_x)
    i = @index(Global)

    if i <= length_x
        ss = 0.0f0
        for j in 1:length_x
            ss += x[j] * x[j]
        end

        ss /= length_x
        ss += 1f-6
        ss = 1f0 / sqrt(ss)
        o[i] = weight[i] * (ss * x[i])
    end
end

function rmsnorm!(o::AbstractVector, x::AbstractVector, weight::AbstractVector)
    length_x = length(x)

    kernel! = rmsnorm_kernel!(KernelAbstractions.get_backend(o))
    kernel!(o, x, weight, length_x, ndrange=length_x)
    KernelAbstractions.synchronize(KernelAbstractions.get_backend(o))
end
