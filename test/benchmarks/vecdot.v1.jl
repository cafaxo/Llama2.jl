
function matmul_v2!(
  y::AbstractVector{Float32},
  A::AbstractMatrix{T},
  x::AbstractVector{Float32},
) where {T<:Union{block_q4_K,block_q5_K}}
  x_sums = sum_blocks_ka(x, (T <: Union{block_q4_K,block_q5_K}) ? 32 : 16) # FIXME: preallocate this
  vecdot_ka_v1!(y, A, x, x_sums)
  return nothing
end

function vecdot_ka_v1!(y::AbstractVector{Float32}, A::AbstractMatrix{block_q4_K}, x, x_sums::AbstractVector{Float16})
  N = length(y)

  kernel! = vecdot_q4_kernel_v1!(KernelAbstractions.get_backend(y))
  kernel!(y, A, x, x_sums, ndrange=N)

  return y
end

@kernel function vecdot_q4_kernel_v1!(y, @Const(A), @Const(x), @Const(x_sums))
  idx = @index(Global)
  
  if idx <= length(y)
      y[idx] = vecdot_q4_ka_v1(A, idx, x, x_sums)
  end
end
@inline function vecdot_q4_ka_v1(A, idx, x, x_sums)
  # @assert size(x, 1) == length(x) ÷ 256
nb = size(A, 1)

kmask1 = 0x3f3f3f3f
kmask2 = 0x0f0f0f0f
kmask3 = 0x03030303

sumf = zero(Float32)
@inbounds for i in 1:nb
  d = Float32(A[i, idx].d)
  dmin = Float32(A[i, idx].dmin)

  scales = A[i, idx].scales
  # Different tries to convert the 4 byte to 1 UInt32
  # V1 try
#   scales_uint32_t = reinterpret(NTuple{3, UInt32}, scales) # THIS would be the best.
  # V2 try
  # utmp0 = reinterpret(UInt32, (scales[1], scales[2], scales[3], scales[4]))
  # V3 try
  # utmp1, utmp2 = reinterpret(UInt32, scales[5:8]), reinterpret(UInt32, scales[9:12])
  # v4 try
  # FIXME This works... but it is for sure not optimal.
  scales_uint32 = (UInt32(scales[1]) | UInt32(scales[2])<<8 | UInt32(scales[3])<<16 | UInt32(scales[4])<<24,
  UInt32(scales[5]) | UInt32(scales[6])<<8 | UInt32(scales[7])<<16 | UInt32(scales[8])<<24,
  UInt32(scales[9]) | UInt32(scales[10])<<8 | UInt32(scales[11])<<16 | UInt32(scales[12])<<24)

  utmp0, utmp1, utmp2 = scales_uint32[1], scales_uint32[2], scales_uint32[3]


  mins8 = (utmp1 & kmask1, ((utmp2 >> 4) & kmask2) | (((utmp1 >> 6) & kmask3) << 4))
  utmp1 = (utmp2 & kmask2) | (((utmp0 >> 6) & kmask3) << 4)
  utmp0 &= kmask1
  
  q8sums_offset = (i-1)*8 # 8 = 256 ÷ 32(sum size)
  # @cushow typeof(mins8)
  # FIXME This works... but it is for sure not optimal.
  
  mins = (extract_bytes(mins8[1])..., extract_bytes(mins8[2])...)
  # mins8[1] & (mask<<16), mins8[1] & (mask<<24), mins8[2] & mask, mins8[2] & (mask<<8), mins8[2] & (mask<<16), mins8[2] & (mask<<24))   
  # Previously:
  # mins = reinterpret(NTuple{8,UInt8}, mins8)
  s = zero(Float32)
  @fastmath @inbounds for k in 1:8
      s += (dmin * mins[k]) * x_sums[q8sums_offset + k]
  end
  sumf -= s
  
  # FIXME This works... but it is for sure not optimal.
  scales_new = (extract_bytes(utmp0)..., extract_bytes(utmp1)...)
  # Previously:
  # scales_new = @inline reinterpret(NTuple{8,UInt8}, (utmp0, utmp1))
  sumi1 = zero(Float32)
  sumi2 = zero(Float32)

  qs = A[i, idx].qs
  qs_offset = 0
  q8_offset = (i - 1) * 256

  for j in 1:(QK_K ÷ 64)
      s = zero(Float32)
      @fastmath @inbounds for k in 1:32
          s += (d * reinterpret(Int8, qs[qs_offset + k] & 0xf)) * x[q8_offset + k]
      end
      sumi1 += s * scales_new[2 * (j - 1) + 1]
      q8_offset += 32

      s = zero(Float32)
      @fastmath @inbounds for k in 1:32
          s += (d * reinterpret(Int8, qs[qs_offset + k] >> 4)) * x[q8_offset + k]
      end
      sumi2 += s * scales_new[2 * (j - 1) + 2]

      qs_offset += 32
      q8_offset += 32
  end

  sumf += sumi1 + sumi2
end
sumf
end
