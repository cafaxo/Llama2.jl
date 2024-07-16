
using KernelAbstractions
using KernelAbstractions.Extras: @unroll

function matmul!(
    y::AbstractVector{Float32},
    A::AbstractMatrix{T},
    x::AbstractVector{Float32},
) where {T<:Union{block_q4_K,block_q5_K,block_q6_K}}
    x_sums = block_sums(x, (T <: Union{block_q4_K,block_q5_K}) ? 32 : 16)

    vecdot!(y, A, x, x_sums)
    return nothing
end

@kernel function block_sums_kernel_v2(@Const(x), sums)
    i = @index(Global, Linear)
    li = @index(Local, Linear)
    N = @uniform @groupsize()[1]
    
    shared_mem = @localmem Float32 N
    
    @inbounds if i <= length(x)
        shared_mem[li] = x[i]
        @synchronize
        
        s = N ÷ 2
        while s > 0
            if li <= s
                shared_mem[li] += shared_mem[li + s]
            end
            s ÷= 2
            @synchronize
        end
        
        if li == 1
            block_id = (i - 1) ÷ N + 1
            sums[block_id] = Float16(shared_mem[1])
        end
    end
end

function block_sums!(sums, x::AbstractVector{Float32}, block_size::Int=32)
    num_blocks = cld(length(x), block_size)
    backend = KernelAbstractions.get_backend(x)
    kernel! = block_sums_kernel_v2(backend, (block_size,))
    kernel!(x, sums, ndrange=length(x))
    return sums
end

function block_sums(x::AbstractVector{Float32}, block_size::Int=32)
    num_blocks = cld(length(x), block_size)
    sums = KernelAbstractions.zeros(get_backend(x), Float16, num_blocks)  # FIXME: preallocate this, or fuse into the next kernel
    block_sums!(sums, x, block_size)
  
    return sums
end

@inline function vecdot_q4_v2(A, idx, x, x_sums)
    # @assert size(x, 1) == length(x) ÷ 256
    nb = size(A, 1)
  
    kmask1 = 0x3f3f3f3f
    kmask2 = 0x0f0f0f0f
    kmask3 = 0x03030303
  
    sumf = zero(Float32)
    @inbounds for i in 1:nb
        block = A[i, idx]
        d = Float32(block.d)
        dmin = Float32(block.dmin)
        scales = block.scales
        qs = block.qs
    
        scales_uint32 = reinterpret_contiguous(NTuple{3, UInt32}, scales) # THIS would be the best.
    
        utmp0, utmp1, utmp2 = scales_uint32[1], scales_uint32[2], scales_uint32[3]
    
    
        mins8 = (utmp1 & kmask1, ((utmp2 >> 4) & kmask2) | (((utmp1 >> 6) & kmask3) << 4))
        utmp1 = (utmp2 & kmask2) | (((utmp0 >> 6) & kmask3) << 4)
        utmp0 &= kmask1
        
        q8sums_offset = (i-1)*8 # 8 = 256 ÷ 32(sum size)
        
        mins = reinterpret_contiguous(NTuple{8, UInt8}, mins8)
    
        s = zero(Float32)
        @fastmath @unroll for k in 1:8
            s += (dmin * mins[k]) * x_sums[q8sums_offset + k]
        end
        sumf -= s
        
        scales_new = reinterpret_contiguous(NTuple{8,UInt8}, (utmp0, utmp1))
        sumi1 = zero(Float32)
        sumi2 = zero(Float32)
    
        qs = block.qs
        qs_offset = 0
        q8_offset = (i - 1) * 256
    
        @unroll for j in 1:4  # QK_K ÷ 64 = 4
            s1 = zero(Float32)
            s2 = zero(Float32)
            @fastmath @unroll for k in 1:32
                q = qs[qs_offset + k]
                s1 += (q & 0xf) * x[q8_offset + k]
                s2 += (q >> 4) * x[q8_offset + k + 32]
            end
    
            sumi1 += s1 * scales_new[2j - 1]
            sumi2 += s2 * scales_new[2j]
    
            qs_offset += 32
            q8_offset += 64
        end
    
        sumf += d * sumi1 + d * sumi2
    end
    sumf
end

@kernel function vecdot_q4_kernel!(y, @Const(A), @Const(x), @Const(x_sums))
    idx = @index(Global)
    
    if idx <= length(y)
        y[idx] = vecdot_q4_v2(A, idx, x, x_sums)
    end
end

function vecdot!(y::AbstractVector{Float32}, A::AbstractMatrix{block_q4_K}, x, x_sums::AbstractVector{Float16})
    N = length(y)

    kernel! = vecdot_q4_kernel!(KernelAbstractions.get_backend(y), 32)
    kernel!(y, A, x, x_sums, ndrange=N)

    return y
end

@inline function vecdot_q5_v2(A, idx, x, x_sums)
    nb = size(A, 1)
    sumf = zero(Float32)

    @inbounds for i in 1:nb
        block = A[i, idx]
        d = Float32(block.d)
        dmin = Float32(block.dmin)
        scales = block.scales
        qs = block.qs
        qh = block.qh

        scales_uint32 = reinterpret_contiguous(NTuple{3, UInt32}, scales)
        utmp0, utmp1, utmp2 = scales_uint32[1], scales_uint32[2], scales_uint32[3]

        mins8 = (utmp1 & 0x3f3f3f3f, ((utmp2 >> 4) & 0x0f0f0f0f) | (((utmp1 >> 6) & 0x03030303) << 4))
        utmp1 = (utmp2 & 0x0f0f0f0f) | (((utmp0 >> 6) & 0x03030303) << 4)
        utmp0 &= 0x3f3f3f3f
        
        q8sums_offset = (i-1) * 8
        mins = reinterpret_contiguous(NTuple{8, UInt8}, mins8)

        s = zero(Float32)
        @fastmath @unroll for k in 1:8
            s += (dmin * mins[k]) * x_sums[q8sums_offset + k]
        end
        sumf -= s
        
        scales_new = reinterpret_contiguous(NTuple{8,UInt8}, (utmp0, utmp1))
        
        sumi = zero(Float32)
        qhbits = qh
        qs_offset = 0
        q8_offset = (i - 1) * 256

        @unroll for j in 1:4  # QK_K ÷ 64 = 4
            s1 = zero(Float32)
            s2 = zero(Float32)
            @fastmath @unroll for k in 1:32
                q5h0 = (qhbits[k] & 0x1) << 4
                q5h1 = (qhbits[k] & 0x2) << 3
                q5bytes0 = (qs[qs_offset + k] & 0x0f) | q5h0
                q5bytes1 = (qs[qs_offset + k] >> 4) | q5h1
                x1 = x[q8_offset + k]
                x2 = x[q8_offset + k + 32]
                s1 += (d * reinterpret(Int8, q5bytes0)) * x1
                s2 += (d * reinterpret(Int8, q5bytes1)) * x2
            end
            sumi += s1 * scales_new[2j - 1] + s2 * scales_new[2j]

            qhbits = qhbits .>> 2
            qs_offset += 32
            q8_offset += 64
        end

        sumf += sumi
    end
    sumf
end

@kernel function vecdot_q5_kernel!(y, @Const(A), @Const(x), @Const(x_sums))
    idx = @index(Global)
    
    if idx <= length(y)
        y[idx] = vecdot_q5_v2(A, idx, x, x_sums)
    end
end

function vecdot!(y::AbstractVector{Float32}, A::AbstractMatrix{block_q5_K}, x, x_sums::AbstractVector{Float16})
    N = length(y)

    kernel! = vecdot_q5_kernel!(KernelAbstractions.get_backend(y), 32)
    kernel!(y, A, x, x_sums, ndrange=N)

    return y
end

@inline function _vecdot_hack(scale, sums::AbstractVector, i, d_all)
    q8sums_offset = (i-1)*16
    s = zero(Float32)

    @fastmath @inbounds for k in 1:16
        s += (d_all * scale[k]) * sums[q8sums_offset + k]
    end

    return 32 * s
end

@inline function vecdot_q6_v2(A, idx, x, x_sums)
    nb = size(A, 1)
    sumf = zero(Float32)

    @inbounds for i in 1:nb
        block = A[i, idx]
        d_all = block.d
        scale = block.scales
        qh = block.qh
        q6 = block.ql
        
        isum_mins = _vecdot_hack(scale, x_sums, i, d_all)
        isum = zero(Float32)
        
        scale_offset = 0
        qh_offset = 0
        q6_offset = 0
        q8_offset = (i-1) * 256

        @unroll for j in 1:2  # QK_K ÷ 128 = 4
            # First half (quarters 1 and 2)
            s = (zero(Float32), zero(Float32), zero(Float32), zero(Float32))
            @fastmath @unroll for k in 1:16
                qhbits0 = qh[qh_offset + k]
                qhbits1 = qh[qh_offset + 16 + k]
                
                q6h0 = (qhbits0 & 0x03) << 4
                q6h1 = (qhbits1 & 0x03) << 4
                q6h2 = ((qhbits0 >> 2) & 0x03) << 4
                q6h3 = ((qhbits1 >> 2) & 0x03) << 4
                
                q6bits0 = q6[q6_offset + k]
                q6bits1 = q6[q6_offset + 16 + k]
                q6bits2 = q6[q6_offset + 2*16 + k]
                q6bits3 = q6[q6_offset + 3*16 + k]
                
                q6bytes0 = d_all * reinterpret(Int8, (q6bits0 & 0x0f) | q6h0)
                q6bytes1 = d_all * reinterpret(Int8, (q6bits1 & 0x0f) | q6h1)
                q6bytes2 = d_all * reinterpret(Int8, (q6bits2 & 0x0f) | q6h2)
                q6bytes3 = d_all * reinterpret(Int8, (q6bits3 & 0x0f) | q6h3)
                
                s = (s[1] + q6bytes0 * x[q8_offset + k],
                     s[2] + q6bytes1 * x[q8_offset + 16 + k],
                     s[3] + q6bytes2 * x[q8_offset + 32 + k],
                     s[4] + q6bytes3 * x[q8_offset + 48 + k])
            end
            
            isum += s[1] * scale[scale_offset + 1] + s[2] * scale[scale_offset + 2]
            isum += s[3] * scale[scale_offset + 3] + s[4] * scale[scale_offset + 4]
            scale_offset += 4
            q8_offset += 64

            # Second half (quarters 3 and 4)
            s = (zero(Float32), zero(Float32), zero(Float32), zero(Float32))
            @fastmath @unroll for k in 1:16
                qhbits0 = qh[qh_offset + k]
                qhbits1 = qh[qh_offset + 16 + k]
                
                q6h0 = ((qhbits0 >> 4) & 0x03) << 4
                q6h1 = ((qhbits1 >> 4) & 0x03) << 4
                q6h2 = ((qhbits0 >> 6) & 0x03) << 4
                q6h3 = ((qhbits1 >> 6) & 0x03) << 4
                
                q6bits0 = q6[q6_offset + k]
                q6bits1 = q6[q6_offset + 16 + k]
                q6bits2 = q6[q6_offset + 2*16 + k]
                q6bits3 = q6[q6_offset + 3*16 + k]
                
                q6bytes0 = d_all * reinterpret(Int8, (q6bits0 >> 4) | q6h0)
                q6bytes1 = d_all * reinterpret(Int8, (q6bits1 >> 4) | q6h1)
                q6bytes2 = d_all * reinterpret(Int8, (q6bits2 >> 4) | q6h2)
                q6bytes3 = d_all * reinterpret(Int8, (q6bits3 >> 4) | q6h3)
                
                s = (s[1] + q6bytes0 * x[q8_offset + k],
                     s[2] + q6bytes1 * x[q8_offset + 16 + k],
                     s[3] + q6bytes2 * x[q8_offset + 32 + k],
                     s[4] + q6bytes3 * x[q8_offset + 48 + k])
            end
            
            isum += s[1] * scale[scale_offset + 1] + s[2] * scale[scale_offset + 2]
            isum += s[3] * scale[scale_offset + 3] + s[4] * scale[scale_offset + 4]
            scale_offset += 4
            q8_offset += 64

            qh_offset += 32
            q6_offset += 64
        end

        sumf += isum - isum_mins
    end

    return sumf
end

@kernel function vecdot_q6_kernel!(y, A, x, x_sums)
    idx = @index(Global)
    
    if idx <= length(y)
        y[idx] = vecdot_q6_v2(A, idx, x, x_sums)
    end
end

function vecdot!(y::AbstractVector{Float32}, A::AbstractMatrix{block_q6_K}, x, x_sums::AbstractVector{Float16})
    N = length(y)

    kernel! = vecdot_q6_kernel!(KernelAbstractions.get_backend(y), 16)
    kernel!(y, A, x, x_sums, ndrange=N)
end

# https://github.com/JuliaLang/julia/pull/43065 suggestion by Tim Besard @maleadt
@inline function reinterpret_contiguous(::Type{T}, val::U) where {T,U}
  box = Ref(val)
  ptr = Base.unsafe_convert(Ptr{U}, box)
  return unsafe_load(convert(Ptr{T}, ptr))
end