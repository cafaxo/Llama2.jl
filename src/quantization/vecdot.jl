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
