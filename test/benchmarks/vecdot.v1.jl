using KernelAbstractions
using Llama2: block_q4_K, block_q5_K, block_q6_K, QK_K, extract_bytes, sum_blocks_ka, reinterpret_contiguous, _vecdot_hack
function matmul_v1!(
  y::AbstractVector{Float32},
  A::AbstractMatrix{T},
  x::AbstractVector{Float32},
) where {T<:Union{block_q4_K,block_q5_K,block_q6_K}}
  x_sums = block_sums_v1(x, (T <: Union{block_q4_K,block_q5_K}) ? 32 : 16) # FIXME: preallocate this
  vecdot_ka_v1!(y, A, x, x_sums)
  return nothing
end

@kernel function block_sums_kernel_v2_forperf(@Const(x), sums)
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
# this simple solution is pretty much the same speed as the above LMEM solution.
@kernel function block_sums_kernel_v1(@Const(x), sums, num_blocks, sum_size)
  block_id = @index(Global)

  if block_id <= num_blocks
      sum = 0.0f0
      start_idx = (block_id-1) * sum_size
      for i in 1:sum_size
          sum += x[start_idx + i]
      end
      sums[block_id] = sum
  end
end

function block_sums_v1!(sums, x::AbstractVector{Float32}, block_size::Int=32)
  num_blocks = cld(length(x), block_size)
  backend = KernelAbstractions.get_backend(x)
  # kernel! = block_sums_kernel_v1(backend, (block_size,))
  # kernel!(x, sums, num_blocks, block_size, ndrange=num_blocks)
  kernel! = block_sums_kernel_v2_forperf(backend, (block_size,)) # we use v2 version because this way we have the same solution for speed comparison.
  kernel!(x, sums, ndrange=length(x))
  return sums
end

function block_sums_v1(x::AbstractVector{Float32}, block_size::Int=32)
  num_blocks = cld(length(x), block_size)
  sums = KernelAbstractions.zeros(get_backend(x), Float16, num_blocks)  # FIXME: preallocate this, or fuse into the next kernel
  # sums = similar(x, Float16, num_blocks)
  block_sums_v1!(sums, x, block_size)

  return sums
end

include("archived.v1.jl")

function vecdot_ka_v1!(y::AbstractVector{Float32}, A::AbstractMatrix{block_q4_K}, x, x_sums::AbstractVector{Float16})
  N = length(y)

  kernel! = vecdot_q4_kernel_v1!(KernelAbstractions.get_backend(y), 32)
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


function vecdot_ka_v1!(y::AbstractVector{Float32}, A::AbstractMatrix{block_q5_K}, x, x_sums::AbstractVector{Float16})
  N = length(y)

  kernel! = vecdot_q5_kernel_v1!(KernelAbstractions.get_backend(y), 8)
  kernel!(y, A, x, x_sums, ndrange=N)

  return y
end

@kernel function vecdot_q5_kernel_v1!(y, A, x, x_sums)
  idx = @index(Global)
  
  if idx <= length(y)
      y[idx] = vecdot_q5_ka_v1(A, idx, x, x_sums)
  end
end

@inline function vecdot_q5_ka_v1(A, idx, x, x_sums)
  nb = size(A, 1)

  kmask1 = 0x3f3f3f3f
  kmask2 = 0x0f0f0f0f
  kmask3 = 0x03030303

  sumf = zero(Float32)
  for i in 1:nb
      d = Float32(A[i, idx].d)
      dmin = Float32(A[i, idx].dmin)

      scales = A[i, idx].scales
      
      scales_uint32 = reinterpret_contiguous(NTuple{3, UInt32}, scales) # THIS would be the best.

      utmp0, utmp1, utmp2 = scales_uint32[1], scales_uint32[2], scales_uint32[3]

      mins8 = (utmp1 & kmask1, ((utmp2 >> 4) & kmask2) | (((utmp1 >> 6) & kmask3) << 4))
      utmp1 = (utmp2 & kmask2) | (((utmp0 >> 6) & kmask3) << 4)
      utmp0 &= kmask1
      
      q8sums_offset = (i-1)*8
      
      mins = reinterpret_contiguous(NTuple{8, UInt8}, mins8)
      
      s = zero(Float32)
      @fastmath @inbounds for k in 1:8
          s += (dmin * mins[k]) * x_sums[q8sums_offset + k]
      end
      sumf -= s

      scales_new = reinterpret_contiguous(NTuple{8,UInt8}, (utmp0, utmp1))
      
      sumi = zero(Float32)

      qs = A[i, idx].qs
      qh = A[i, idx].qh
      qhbits = qh
      qs_offset = 0
      q8_offset = (i - 1) * 256

      for j in 1:(QK_K ÷ 64)
          s = zero(Float32)
          @fastmath @inbounds for k in 1:32
              q5h0 = (qhbits[k] & 0x1) << 4
              q5bytes0 = (qs[qs_offset + k] & 0x0f) | q5h0
              s += (d * reinterpret(Int8, q5bytes0)) * x[q8_offset + k]
          end
          sumi += s * scales_new[2 * (j - 1) + 1]
          q8_offset += 32

          s = zero(Float32)
          @fastmath @inbounds for k in 1:32
              q5h1 = (qhbits[k] & 0x2) << 3
              q5bytes1 = (qs[qs_offset + k] >> 4) | q5h1
              s += (d * reinterpret(Int8, q5bytes1)) * x[q8_offset + k]
          end
          sumi += s * scales_new[2 * (j - 1) + 2]

          qhbits = qhbits .>> 2
          qs_offset += 32
          q8_offset += 32
      end

      sumf += sumi
  end
  sumf
end


function vecdot_ka_v1!(y::AbstractVector{Float32}, A::AbstractMatrix{block_q6_K}, x, x_sums::AbstractVector{Float16})
  N = length(y)

  kernel! = vecdot_q6_kernel_v1!(KernelAbstractions.get_backend(y))
  kernel!(y, A, x, x_sums, ndrange=N)
end

@kernel function vecdot_q6_kernel_v1!(y, A, x, x_sums)
  idx = @index(Global)
  
  if idx <= length(y)
      y[idx] = vecdot_q6_ka_v1(A, idx, x, x_sums)
  end
end

@inline function vecdot_q6_ka_v1(A, idx, x, x_sums)
  nb = size(A, 1)

 sumf = zero(Float32)

 for i in 1:nb
     d_all = A[i, idx].d
     scale = A[i, idx].scales
     
     isum_mins = _vecdot_hack(scale, x_sums, i, d_all)
     # @show isum_mins

     isum = zero(Float32)
     
     qh = A[i, idx].qh
     q6 = A[i, idx].ql
     scale_offset = 0
     qh_offset = 0
     q6_offset = 0
     q8_offset = (i-1) * 256

     for j in 1:(256 ÷ 128)
         s1 = zero(Float32)

         @fastmath @inbounds for k in 1:16
             qhbits0 = qh[qh_offset + k]
             q6h0 = (qhbits0 & 0x03) << 4
             q6bits0 = q6[q6_offset + k]
             q6bytes0 = d_all * reinterpret(Int8, (q6bits0 & 0x0f) | q6h0)
             
             s1 += q6bytes0 * x[q8_offset + k]
         end

         q8_offset += 16  # Adjusted offset step size
         s2 = zero(Float32)

         @fastmath @inbounds for k in 1:16
             qhbits1 = qh[qh_offset + 16 + k]
             q6h1 = (qhbits1 & 0x03) << 4
             q6bits1 = q6[q6_offset + 16 + k]
             q6bytes1 = d_all * reinterpret(Int8, (q6bits1 & 0x0f) | q6h1)

             s2 += q6bytes1 * x[q8_offset + k]
             # i<3 && j<2 && k>14&& @show q6bytes1, s2
         end

         isum += s1 * scale[scale_offset + 1] + s2 * scale[scale_offset + 2]
         scale_offset += 2
         q8_offset += 16

         s1 = zero(Float32)

         @fastmath @inbounds for k in 1:16
             qhbits0 = qh[qh_offset + k]
             q6h2 = ((qhbits0 >> 2) & 0x03) << 4
             q6bits2 = q6[q6_offset + 2*16 + k]
             q6bytes2 = d_all * reinterpret(Int8, (q6bits2 & 0x0f) | q6h2)

             s1 += q6bytes2 * x[q8_offset + k]
         end

         q8_offset += 16
         s2 = zero(Float32)

         @fastmath @inbounds for k in 1:16
             qhbits1 = qh[qh_offset + 16 + k]
             q6h3 = ((qhbits1 >> 2) & 0x03) << 4
             q6bits3 = q6[q6_offset + 3*16 + k]
             q6bytes3 = d_all * reinterpret(Int8, (q6bits3 & 0x0f) | q6h3)

             s2 += q6bytes3 * x[q8_offset + k]
         end

         isum += s1 * scale[scale_offset + 1] + s2 * scale[scale_offset + 2]
         # i<3 &&@show isum, s1, s2

         scale_offset += 2
         q8_offset += 16

         s1 = zero(Float32)

         @fastmath @inbounds for k in 1:16
             qhbits0 = qh[qh_offset + k]
             q6h0 = ((qhbits0 >> 4) & 0x03) << 4
             q6bits0 = q6[q6_offset + k]
             q6bytes0 = d_all * reinterpret(Int8, (q6bits0 >> 4) | q6h0)

             s1 += q6bytes0 * x[q8_offset + k]
         end

         q8_offset += 16
         s2 = zero(Float32)

         @fastmath @inbounds for k in 1:16
             qhbits1 = qh[qh_offset + 16 + k]
             q6h1 = ((qhbits1 >> 4) & 0x03) << 4
             q6bits1 = q6[q6_offset + 16 + k]
             q6bytes1 = d_all * reinterpret(Int8, (q6bits1 >> 4) | q6h1)

             s2 += q6bytes1 * x[q8_offset + k]
             # i<3 && j<2 && k>14&& @show q6bytes1, s2
         end

         isum += s1 * scale[scale_offset + 1] + s2 * scale[scale_offset + 2]
         scale_offset += 2
         q8_offset += 16

         s1 = zero(Float32)

         @fastmath @inbounds for k in 1:16
             qhbits0 = qh[qh_offset + k]
             q6h2 = ((qhbits0 >> 6) & 0x03) << 4
             q6bits2 = q6[q6_offset + 2*16 + k]
             q6bytes2 = d_all * reinterpret(Int8, (q6bits2 >> 4) | q6h2)

             s1 += q6bytes2 * x[q8_offset + k]
         end

         q8_offset += 16
         s2 = zero(Float32)

         @fastmath @inbounds for k in 1:16
             qhbits1 = qh[qh_offset + 16 + k]
             q6h3 = ((qhbits1 >> 6) & 0x03) << 4
             q6bits3 = q6[q6_offset + 3*16 + k]
             q6bytes3 = d_all * reinterpret(Int8, (q6bits3 >> 4) | q6h3)

             s2 += q6bytes3 * x[q8_offset + k]
         end

         isum += s1 * scale[scale_offset + 1] + s2 * scale[scale_offset + 2]

         scale_offset += 2
         qh_offset += 32
         q6_offset += 64
         q8_offset += 16
     end

     sumf += isum - isum_mins
 end

 return sumf
end
