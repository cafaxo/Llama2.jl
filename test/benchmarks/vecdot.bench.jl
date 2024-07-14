using Llama2: reinterpret_contiguous

function matmul_v2!(
  y::AbstractVector{Float32},
  A::AbstractMatrix{T},
  x::AbstractVector{Float32},
) where {T<:Union{block_q4_K,block_q5_K}}
  x_sums = sum_blocks_ka(x, (T <: Union{block_q4_K,block_q5_K}) ? 32 : 16) # FIXME: preallocate this
  vecdot_ka_v2_try!(y, A, x, x_sums)
  return nothing
end


@inline function vecdot_q4_ka_v2_try(A, idx, x, x_sums)
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

    @unroll for j in 1:4
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

function vecdot_ka_v2_try!(y::AbstractVector{Float32}, A::AbstractMatrix{block_q4_K}, x, x_sums::AbstractVector{Float16})
N = length(y)

kernel! = vecdot_q4_kernel_v2_try!(KernelAbstractions.get_backend(y), 32)
kernel!(y, A, x, x_sums, ndrange=N)

return y
end

@kernel function vecdot_q4_kernel_v2_try!(y, @Const(A), @Const(x), @Const(x_sums))
idx = @index(Global)

if idx <= length(y)
    y[idx] = vecdot_q4_ka_v2_try(A, idx, x, x_sums)
end
end
