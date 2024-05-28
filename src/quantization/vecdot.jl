struct BlockF16Sums16
    x::NTuple{256,Float16}
    sums::NTuple{16,Float16} # sum of 16 element blocks
end

function BlockF16Sums16(x::NTuple{256,Float16})
    return BlockF16Sums16(x, ntuple(i -> sum(ntuple(j -> x[16*(i-1) + j], Val(16))), Val(16)))
end

function to_block_f16_sums16(x::Vector{<:Real})
    return [BlockF16Sums16(ntuple(j -> Float16(x[256*(i-1) + j]), Val(256))) for i in 1:length(x)÷256]
end

struct BlockF16Sums32
    x::NTuple{256,Float16}
    sums::NTuple{8,Float16} # sum of 32 element blocks
end

function BlockF16Sums32(x::NTuple{256,Float16})
    return BlockF16Sums32(x, ntuple(i -> sum(ntuple(j -> x[32*(i-1) + j], Val(32))), Val(8)))
end

function to_block_f16_sums32(x::Vector{<:Real})
    return [BlockF16Sums32(ntuple(j -> Float16(x[256*(i-1) + j]), Val(256))) for i in 1:length(x)÷256]
end

function vecdot(x::AbstractVector{block_q4_K}, y::AbstractVector{BlockF16Sums32})
    @assert length(x) == length(y)
    nb = length(x)

    sumf = zero(Float32)

    GC.@preserve x y begin
    @inbounds for i in 1:nb
        d = unsafe_load(unsafe_pointer_to_field(x, i, :d))
        dmin = unsafe_load(unsafe_pointer_to_field(x, i, :dmin))
        q8sums = unsafe_pointer_to_field(y, i, :sums)

        xi_scales = convert(Ptr{UInt32}, unsafe_pointer_to_field(x, i, :scales))
        utmp0, utmp1, utmp2 = unsafe_load(xi_scales, 1), unsafe_load(xi_scales, 2), unsafe_load(xi_scales, 3)

        kmask1 = 0x3f3f3f3f
        kmask2 = 0x0f0f0f0f
        kmask3 = 0x03030303

        mins8 = (utmp1 & kmask1, ((utmp2 >> 4) & kmask2) | (((utmp1 >> 6) & kmask3) << 4))
        utmp1 = (utmp2 & kmask2) | (((utmp0 >> 6) & kmask3) << 4)
        utmp0 &= kmask1

        mins = @inline reinterpret(NTuple{8,UInt8}, mins8)

        s = zero(Float32)

        @fastmath @inbounds for k in 1:8
            s += (dmin * mins[k]) * unsafe_load(q8sums, k)
        end

        sumf -= s

        scales = @inline reinterpret(NTuple{8,UInt8}, (utmp0, utmp1))

        sumi1 = zero(Float32)
        sumi2 = zero(Float32)

        q4 = unsafe_pointer_to_field(x, i, :qs)
        q8 = unsafe_pointer_to_field(y, i, :x)

        for j in 1:(QK_K÷64)
            s = zero(Float32)
    
            @fastmath @inbounds for k in 1:32
                s += (d * reinterpret(Int8, unsafe_load(q4, k) & 0xf)) * unsafe_load(q8, k)
            end

            sumi1 += s * scales[2*(j-1)+1]

            q8 += 32 * sizeof(Float16)

            s = zero(Float32)
    
            @fastmath @inbounds for k in 1:32
                s += (d * reinterpret(Int8, unsafe_load(q4, k) >> 4)) * unsafe_load(q8, k)
            end

            sumi2 += s * scales[2*(j-1)+2]

            q4 += 32
            q8 += 32 * sizeof(Float16)
        end

        sumf += sumi1 + sumi2
    end
    end

    return sumf
end
# CUDA equvalent of the above, for reproducing the same results...
using CUDA

# function cuda_dot!(result, x_d, x_dmin, y_d, y_bsums, x_scales, x_qs, y_qs, nb)

function cuda_dot_kernel(result::CuVector{Float32}, v::CuVector{Int32}, scales::CuVector{Int32}, d4::Float32, ds8::CuVector{Float32},  vdr::Int)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if idx > length(result)
        return
    end

    sumi = 0
    QI4_0 = 4  # Assuming QI4_0 is defined somewhere, adjust as needed

    for i in 1:nb
        vi0 = (v[i] >> 0) & 0x0F0F0F0F
        vi1 = (v[i] >> 4) & 0x0F0F0F0F

        # Manually compute the dot product for 4-bit values
        for bit_idx in 0:((QK_K÷64)-1)  # 8 4-bit values in each 32-bit integer
            v0 = (vi0 >> (4 * bit_idx)) & 0xF
            v1 = (vi1 >> (4 * bit_idx)) & 0xF
            u0 = (scales[2*i - 1] >> (4 * bit_idx)) & 0xF
            u1 = (scales[2*i] >> (4 * bit_idx)) & 0xF

            sumi += v0 * u0 + v1 * u1
        end
    end

    ds8f_x, ds8f_y = ds8[1], ds8[2]  # Assuming ds8 contains two elements for scaling

    # Compute final result
    result[idx] = d4 * (sumi * ds8f_x - (8 * nb / QI4_0) * ds8f_y)
end
using CUDA
function dp4a!(a::Int32, b::Int32, c::Int32)
    va = reinterpret(Vec{4,Int8}, a)
    vb = reinterpret(Vec{4,Int8}, b)
    c += va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2] + va[3] * vb[3]
    nothing
end
# function cuda_dot_kernel(x_qs::CuVector{UInt8}, y_qs::CuVector{UInt8}, x_d::CuVector{Float32}, y_d::CuVector{Float32}, x_dmin::CuVector{Float32}, y_bsums::CuVector{Int16}, x_scales::CuVector{UInt32}, result::CuVector{Float32}, nb::Int)
function cuda_dot_kernel(x::CuVector{block_q4_K}, y::CuVector{block_q8_K})
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    nb = length(x)

    if idx > nb
        return
    end

    x_d = x[idx].d
    x_dmin = x[idx].dmin
    x_scales = x[idx].scales
    x_qs = x[idx].qs
    y_d = y[idx].d
    y_dmin = y[idx].dmin
    y_scales = y[idx].scales
    y_qs = y[idx].qs
    sumf = 0f0  # Corresponds to sumf in the CPU version
    d = y_d[idx] * x_d[idx]  # Similar to d = yi_d * Float32(xi_d) in the CPU version
    dmin = y_d[idx] * x_dmin[idx]  # Similar to dmin = yi_d * Float32(xi_dmin) in the CPU version

    # The following loop is an adaptation of the loop in the CPU version
    # that iterates over quantized values and performs dot product calculations
    # Extract 4-bit values from x_qs and y_qs, similar to q4 and q8 in the CPU version
    # vi0 and vi1 correspond to the unpacked 4-bit values from x_qs (similar to q4bits in the CPU version)
    vi0 = (x_qs[idx] >> 0) & 0x0F0F0F0F
    vi1 = (x_qs[idx] >> 4) & 0x0F0F0F0F

    sumi = 0  # Placeholder for the first half of dot product calculations
    sumi2 = 0  # Placeholder for the second half of dot product calculations

    # Placeholder loop for dot product calculation, similar to the loop over q4bits and q8bytes in the CPU version
    for bit_idx in 0:(QK_K÷64)-1
        v0 = (vi0 >> (4 * bit_idx)) & 0xF
        v1 = (vi1 >> (4 * bit_idx)) & 0xF
        u0 = (x_scales[2*idx - 1] >> (4 * bit_idx)) & 0xF
        u1 = (x_scales[2*idx] >> (4 * bit_idx)) & 0xF

        # Perform the dot product operation for the unpacked 4-bit values
        sumi += v0 * u0 + v1 * u1
    end

    # Incorporate the dot product result into sumf, adjusting with d and dmin
    sumf += d * sumi - dmin

    # Adjust the result with dmin and scaling factors from y_bsums and x_scales
    # This part needs to be adapted to match the CPU version's final adjustments with dmin, q8sums, mins, and scales
    result[idx] = sumf - dmin  # Placeholder for final result calculation

    return
end

# function launch_cuda_dot_kernel!(v, u, d4, ds8, result, vdr)
function LinearAlgebra.dot(x::CuVector{block_q4_K}, y::CuVector{block_q8_K})
    @assert length(x) == length(y)
    nb = length(x)

    threads_per_block = 256  # Adjust as needed
    # num_blocks = ceil(Int, length(result) / threads_per_block)
    blocks = ceil(Int, nb / threads_per_block)
    # @cuda threads=threads_per_block blocks=num_blocks cuda_dot_kernel(v, u, d4, ds8, result, vdr)
    # @cuda blocks=blocks threads=threads_per_block cuda_dot_kernel(result, x_d, x_dmin, y_d, y_bsums, x_scales, x_qs, y_qs, nb)
    @cuda threads=threads_per_block blocks=num_blocks cuda_dot_kernel(x_qs, y_qs, x_d, y_d, x_dmin, y_bsums, x_scales, result, nb)
end

function vecdot(x::AbstractVector{block_q5_K}, y::AbstractVector{BlockF16Sums32})
    @assert length(x) == length(y)
    nb = length(x)

    sumf = zero(Float32)

    GC.@preserve x y begin
    @inbounds for i in 1:nb
        d = unsafe_load(unsafe_pointer_to_field(x, i, :d))
        dmin = unsafe_load(unsafe_pointer_to_field(x, i, :dmin))
        q8sums = unsafe_pointer_to_field(y, i, :sums)

        xi_scales = convert(Ptr{UInt32}, unsafe_pointer_to_field(x, i, :scales))
        utmp0, utmp1, utmp2 = unsafe_load(xi_scales, 1), unsafe_load(xi_scales, 2), unsafe_load(xi_scales, 3)

        kmask1 = 0x3f3f3f3f
        kmask2 = 0x0f0f0f0f
        kmask3 = 0x03030303

        mins8 = (utmp1 & kmask1, ((utmp2 >> 4) & kmask2) | (((utmp1 >> 6) & kmask3) << 4))
        utmp1 = (utmp2 & kmask2) | (((utmp0 >> 6) & kmask3) << 4)
        utmp0 &= kmask1

        mins = @inline reinterpret(NTuple{8,UInt8}, mins8)

        s = zero(Float32)

        @fastmath @inbounds for k in 1:8
            s += (dmin * mins[k]) * unsafe_load(q8sums, k)
        end

        sumf -= s

        scales = @inline reinterpret(NTuple{8,UInt8}, (utmp0, utmp1))

        q5 = unsafe_pointer_to_field(x, i, :qs)
        qh = unsafe_pointer_to_field(x, i, :qh)
        q8 = unsafe_pointer_to_field(y, i, :x)

        qhbits = unsafe_load(convert(Ptr{NTuple{32,UInt8}}, qh))

        sumi = zero(Float32)

        for j in 1:(QK_K÷64)
            s = zero(Float32)

            @fastmath @inbounds for k in 1:32
                q5h0 = (qhbits[k] & 0x1) << 4
                q5bytes0 = (unsafe_load(q5, k) & 0x0f) | q5h0
                s += (d * reinterpret(Int8, q5bytes0)) * unsafe_load(q8, k)
            end

            sumi += s * scales[2*(j-1)+1]

            q8 += 32 * sizeof(Float16)
            s = zero(Float32)

            @fastmath @inbounds for k in 1:32
                q5h1 = (qhbits[k] & 0x2) << 3
                q5bytes1 = (unsafe_load(q5, k) >> 4) | q5h1
                s += (d * reinterpret(Int8, q5bytes1)) * unsafe_load(q8, k)
            end

            sumi += s * scales[2*(j-1)+2]

            qhbits = qhbits .>> 2
            q5 += 32
            q8 += 32 * sizeof(Float16)
        end

        sumf += sumi
    end
    end

    return sumf
end

@noinline function _vecdot_hack(x, y, i, d_all)
    scale = unsafe_pointer_to_field(x, i, :scales)
    q8sums = unsafe_pointer_to_field(y, i, :sums)

    s = zero(Float32)

    @fastmath @inbounds for k in 1:16
        s += (d_all*unsafe_load(scale, k)) * unsafe_load(q8sums, k)
    end

    return 32 * s
end

function vecdot(x::AbstractVector{block_q6_K}, y::AbstractVector{BlockF16Sums16})
    @assert length(x) == length(y)
    nb = length(x)

    sumf = zero(Float32)

    GC.@preserve x y begin
    @inbounds for i in 1:nb
        d_all = unsafe_load(unsafe_pointer_to_field(x, i, :d))
        scale = unsafe_pointer_to_field(x, i, :scales)

        # for some reason LLVM does not auto-vectorize this code if it is inlined
        # workaround by manually outlining
        isum_mins = _vecdot_hack(x, y, i, d_all)

        isum = zero(Float32)

        qh = unsafe_pointer_to_field(x, i, :qh)
        q6 = unsafe_pointer_to_field(x, i, :ql)
        q8 = unsafe_pointer_to_field(y, i, :x)

        for j in 1:(QK_K÷128)
            s1 = zero(Float32)

            @fastmath @inbounds for k in 1:16
                qhbits0 = unsafe_load(qh, k)
                q6h0 = (qhbits0 & 0x03) << 4
                q6bits0 = unsafe_load(q6, k)
                q6bytes0 = d_all * reinterpret(Int8, (q6bits0 & 0x0f) | q6h0)

                s1 += q6bytes0 * unsafe_load(q8, k)
            end

            q8 += 16 * sizeof(Float16)
            s2 = zero(Float32)

            @fastmath @inbounds for k in 1:16
                qhbits1 = unsafe_load(qh, 16 + k)
                q6h1 = (qhbits1 & 0x03) << 4
                q6bits1 = unsafe_load(q6, 16 + k)
                q6bytes1 = d_all * reinterpret(Int8, (q6bits1 & 0x0f) | q6h1)

                s2 += q6bytes1 * unsafe_load(q8, k)
            end

            isum += s1 * unsafe_load(scale, 1) + s2 * unsafe_load(scale, 2)
            scale += 2

            q8 += 16 * sizeof(Float16)
            s1 = zero(Float32)

            @fastmath @inbounds for k in 1:16
                qhbits0 = unsafe_load(qh, k)
                q6h2 = ((qhbits0 >> 2) & 0x03) << 4
                q6bits2 = unsafe_load(q6, 2*16 + k)
                q6bytes2 = d_all * reinterpret(Int8, (q6bits2 & 0x0f) | q6h2)

                s1 += q6bytes2 * unsafe_load(q8, k)
            end

            q8 += 16 * sizeof(Float16)
            s2 = zero(Float32)

            @fastmath @inbounds for k in 1:16
                qhbits1 = unsafe_load(qh, 16 + k)
                q6h3 = ((qhbits1 >> 2) & 0x03) << 4
                q6bits3 = unsafe_load(q6, 3*16 + k)
                q6bytes3 = d_all * reinterpret(Int8, (q6bits3 & 0x0f) | q6h3)

                s2 += q6bytes3 * unsafe_load(q8, k)
            end

            isum += s1 * unsafe_load(scale, 1) + s2 * unsafe_load(scale, 2)
            scale += 2
            
            q8 += 16 * sizeof(Float16)
            s1 = zero(Float32)

            @fastmath @inbounds for k in 1:16
                qhbits0 = unsafe_load(qh, k)
                q6h0 = ((qhbits0 >> 4) & 0x03) << 4
                q6bits0 = unsafe_load(q6, k)
                q6bytes0 = d_all * reinterpret(Int8, (q6bits0 >> 4) | q6h0)

                s1 += q6bytes0 * unsafe_load(q8, k)
            end

            q8 += 16 * sizeof(Float16)
            s2 = zero(Float32)

            @fastmath @inbounds for k in 1:16
                qhbits1 = unsafe_load(qh, 16 + k)
                q6h1 = ((qhbits1 >> 4) & 0x03) << 4
                q6bits1 = unsafe_load(q6, 16 + k)
                q6bytes1 = d_all * reinterpret(Int8, (q6bits1 >> 4) | q6h1)

                s2 += q6bytes1 * unsafe_load(q8, k)
            end

            isum += s1 * unsafe_load(scale, 1) + s2 * unsafe_load(scale, 2)
            scale += 2

            q8 += 16 * sizeof(Float16)
            s1 = zero(Float32)

            @fastmath @inbounds for k in 1:16
                qhbits0 = unsafe_load(qh, k)
                q6h2 = ((qhbits0 >> 6) & 0x03) << 4
                q6bits2 = unsafe_load(q6, 2*16 + k)
                q6bytes2 = d_all * reinterpret(Int8, (q6bits2 >> 4) | q6h2)

                s1 += q6bytes2 * unsafe_load(q8, k)
            end

            q8 += 16 * sizeof(Float16)
            s2 = zero(Float32)

            @fastmath @inbounds for k in 1:16
                qhbits1 = unsafe_load(qh, 16 + k)
                q6h3 = ((qhbits1 >> 6) & 0x03) << 4
                q6bits3 = unsafe_load(q6, 3*16 + k)
                q6bytes3 = d_all * reinterpret(Int8, (q6bits3 >> 4) | q6h3)

                s2 += q6bytes3 * unsafe_load(q8, k)
            end

            isum += s1 * unsafe_load(scale, 1) + s2 * unsafe_load(scale, 2)
            scale += 2

            qh += 32
            q6 += 64
            q8 += 16 * sizeof(Float16)
        end

        sumf += isum - isum_mins
    end
    end

    return sumf
end
