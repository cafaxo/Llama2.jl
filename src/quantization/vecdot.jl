function LinearAlgebra.dot(x::AbstractVector{block_q4_K}, y::AbstractVector{block_q8_K})
    @assert length(x) == length(y)
    nb = length(x)

    sumf = 0f0

    GC.@preserve x y begin
    @inbounds for i in 1:nb
        yi_d = unsafe_load(unsafe_pointer_to_field(y, i, :d))
        xi_d = unsafe_load(unsafe_pointer_to_field(x, i, :d))
        xi_dmin = unsafe_load(unsafe_pointer_to_field(x, i, :dmin))
        yi_bsums = unsafe_pointer_to_field(y, i, :bsums)

        d = yi_d * Float32(xi_d)
        dmin = yi_d * Float32(xi_dmin)

        q8sums = vpaddq(vload(Vec{8,Int16}, yi_bsums), vload(Vec{8,Int16}, yi_bsums + 8*sizeof(Int16)))

        xi_scales = convert(Ptr{UInt32}, unsafe_pointer_to_field(x, i, :scales))
        utmp0, utmp1, utmp2 = unsafe_load(xi_scales, 1), unsafe_load(xi_scales, 2), unsafe_load(xi_scales, 3)

        kmask1 = 0x3f3f3f3f
        kmask2 = 0x0f0f0f0f
        kmask3 = 0x03030303

        mins8 = Vec((utmp1 & kmask1, ((utmp2 >> 4) & kmask2) | (((utmp1 >> 6) & kmask3) << 4)))
        utmp1 = (utmp2 & kmask2) | (((utmp0 >> 6) & kmask3) << 4)
        utmp0 &= kmask1

        mins = reinterpret(Vec{8,Int16}, Vec{8,UInt16}(reinterpret(Vec{8,UInt8}, mins8)))

        sumf -= dmin * sum(vwidemul(q8sums, mins))

        scales = reinterpret_nonprimitive(NTuple{8,UInt8}, (utmp0, utmp1))

        sumi1 = Int32(0)
        sumi2 = Int32(0)

        q4 = unsafe_pointer_to_field(x, i, :qs)
        q8 = unsafe_pointer_to_field(y, i, :qs)

        for j in 1:(QK_K÷64)
            q4bits = vload(Vec{32,UInt8}, q4)
            q4 += 32

            q8bytes = vload(Vec{32,Int8}, q8)
            q8 += 32

            q4bytes = reinterpret(Vec{32,Int8}, q4bits & 0xf)

            sumi1 += widemul(sum(vwidemul(q4bytes, q8bytes)), scales[2*(j-1)+1])

            q8bytes = vload(Vec{32,Int8}, q8)
            q8 += 32

            q4bytes = reinterpret(Vec{32,Int8}, q4bits >> 4)

            sumi2 += widemul(sum(vwidemul(q4bytes, q8bytes)), scales[2*(j-1)+2])
        end

        sumf += d * (sumi1 + sumi2)
    end
    end

    return sumf
end

function LinearAlgebra.dot(x::AbstractVector{block_q6_K}, y::AbstractVector{block_q8_K})
    @assert length(x) == length(y)
    nb = length(x)

    sumf = 0f0

    GC.@preserve x y begin
    @inbounds for i in 1:nb
        yi_d = unsafe_load(unsafe_pointer_to_field(y, i, :d))
        d_all = Float32(unsafe_load(unsafe_pointer_to_field(x, i, :d)))
        yi_bsums = unsafe_pointer_to_field(y, i, :bsums)
        scale = unsafe_pointer_to_field(x, i, :scales)

        q8sums = vload(Vec{16,Int16}, yi_bsums)
        q6scales = Vec{16,Int16}(vload(Vec{16,Int8}, scale))

        isum_mins = sum(vwidemul(q8sums, q6scales))

        isum = 0

        qh = unsafe_pointer_to_field(x, i, :qh)
        q6 = unsafe_pointer_to_field(x, i, :ql)
        q8 = unsafe_pointer_to_field(y, i, :qs)

        for j in 1:(QK_K÷128)
            qhbits0 = vload(Vec{16,UInt8}, qh)
            qhbits1 = vload(Vec{16,UInt8}, qh + 16)
            qh += 32

            q6bits0 = vload(Vec{16,UInt8}, q6)
            q6bits1 = vload(Vec{16,UInt8}, q6 + 16)
            q6bits2 = vload(Vec{16,UInt8}, q6 + 16*2)
            q6bits3 = vload(Vec{16,UInt8}, q6 + 16*3)
            q6 += 64

            q8bytes0 = vload(Vec{16,Int8}, q8)
            q8bytes1 = vload(Vec{16,Int8}, q8 + 16)
            q8bytes2 = vload(Vec{16,Int8}, q8 + 16*2)
            q8bytes3 = vload(Vec{16,Int8}, q8 + 16*3)
            q8 += 64

            q6h0 = (qhbits0 & 0x03) << 4
            q6h1 = (qhbits1 & 0x03) << 4
            q6h2 = ((qhbits0 >> 2) & 0x03) << 4
            q6h3 = ((qhbits1 >> 2) & 0x03) << 4

            m4b = 0x0f

            q6bytes0 = reinterpret(Vec{16,Int8}, (q6bits0 & 0x0f) | q6h0)
            q6bytes1 = reinterpret(Vec{16,Int8}, (q6bits1 & 0x0f) | q6h1)
            q6bytes2 = reinterpret(Vec{16,Int8}, (q6bits2 & 0x0f) | q6h2)
            q6bytes3 = reinterpret(Vec{16,Int8}, (q6bits3 & 0x0f) | q6h3)

            isum += widemul(sum(vwidemul(q6bytes0, q8bytes0)), unsafe_load(scale, 1)) +
                    widemul(sum(vwidemul(q6bytes1, q8bytes1)), unsafe_load(scale, 2))
            scale += 2

            isum += widemul(sum(vwidemul(q6bytes2, q8bytes2)), unsafe_load(scale, 1)) +
                    widemul(sum(vwidemul(q6bytes3, q8bytes3)), unsafe_load(scale, 2))
            scale += 2

            q8bytes0 = vload(Vec{16,Int8}, q8)
            q8bytes1 = vload(Vec{16,Int8}, q8 + 16)
            q8bytes2 = vload(Vec{16,Int8}, q8 + 16*2)
            q8bytes3 = vload(Vec{16,Int8}, q8 + 16*3)
            q8 += 64

            q6h0 = ((qhbits0 >> 4) & 0x03) << 4
            q6h1 = ((qhbits1 >> 4) & 0x03) << 4
            q6h2 = ((qhbits0 >> 6) & 0x03) << 4
            q6h3 = ((qhbits1 >> 6) & 0x03) << 4

            q6bytes0 = reinterpret(Vec{16,Int8}, (q6bits0 >> 4) | q6h0)
            q6bytes1 = reinterpret(Vec{16,Int8}, (q6bits1 >> 4) | q6h1)
            q6bytes2 = reinterpret(Vec{16,Int8}, (q6bits2 >> 4) | q6h2)
            q6bytes3 = reinterpret(Vec{16,Int8}, (q6bits3 >> 4) | q6h3)

            isum += widemul(sum(vwidemul(q6bytes0, q8bytes0)), unsafe_load(scale, 1)) +
                    widemul(sum(vwidemul(q6bytes1, q8bytes1)), unsafe_load(scale, 2))
            scale += 2

            isum += widemul(sum(vwidemul(q6bytes2, q8bytes2)), unsafe_load(scale, 1)) +
                    widemul(sum(vwidemul(q6bytes3, q8bytes3)), unsafe_load(scale, 2))
            scale += 2
        end

        sumf += d_all * yi_d * (isum - 32 * isum_mins)
    end
    end

    return sumf
end
