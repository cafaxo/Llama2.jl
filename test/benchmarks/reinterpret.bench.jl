

function main()
    scales = UInt8.((1,2,3,4, 2,3,4,5, 3,4,5,6)) # NTuple{12, UInt8}
    @cuda threads=1 blocks=1 reinterpret_kernel_MVP(scales)
end

#%%
@inline function reinterpret_contiguous(::Type{T}, val::U) where {T,U}
  box = Ref(val)
  ptr = Base.unsafe_convert(Ptr{U}, box)
  return unsafe_load(convert(Ptr{T}, ptr))
end
@inline function extract_bytes(x::UInt32)
  return (x & 0xff, (x >> 8) & 0xff, (x >> 16) & 0xff, (x >> 24) & 0xff)
end

using CUDA
function reinterpret_kernel_MVP(scales_vec)
  idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
  
  kmask1, kmask2, kmask3 = 0x3f3f3f3f, 0x0f0f0f0f, 0x03030303
  if idx <= length(scales_vec)
    scales = scales_vec[idx]
    scales_uint32 = (UInt32(scales[1]) | UInt32(scales[2])<<8 | UInt32(scales[3])<<16 | UInt32(scales[4])<<24,
    UInt32(scales[5]) | UInt32(scales[6])<<8 | UInt32(scales[7])<<16 | UInt32(scales[8])<<24,
    UInt32(scales[9]) | UInt32(scales[10])<<8 | UInt32(scales[11])<<16 | UInt32(scales[12])<<24)
    # scales_uint32 = reinterpret(NTuple{3, UInt32}, scales)
    utmp0, utmp1, utmp2 = scales_uint32[1], scales_uint32[2], scales_uint32[3]
    mins8 = (utmp1 & kmask1, ((utmp2 >> 4) & kmask2) | (((utmp1 >> 6) & kmask3) << 4))
    utmp1 = (utmp2 & kmask2) | (((utmp0 >> 6) & kmask3) << 4)
    utmp0 &= kmask1

    mins = (extract_bytes(mins8[1])..., extract_bytes(mins8[2])...)
  end
  nothing
end
function reinterpret_kernel_MVP_pro(scales_vec)
  idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x

  kmask1, kmask2, kmask3 = 0x3f3f3f3f, 0x0f0f0f0f, 0x03030303
  if idx <= length(scales_vec)
    scales = scales_vec[idx]
    scales_uint32 = reinterpret_contiguous(NTuple{3, UInt32}, scales)
    utmp0, utmp1, utmp2 = scales[1], scales[2], scales[3]
    # scales = reinterpret(NTuple{8,UInt8}, (utmp0, utmp1))
    mins8 = (utmp1 & kmask1, ((utmp2 >> 4) & kmask2) | (((utmp1 >> 6) & kmask3) << 4))
    utmp1 = (utmp2 & kmask2) | (((utmp0 >> 6) & kmask3) << 4)
    utmp0 &= kmask1

    mins = reinterpret_contiguous(NTuple{8, UInt8}, mins8)
  end
  nothing
end
scales = UInt8.((1,2,3,4, 2,3,4,5, 3,4,5,6)) # NTuple{12, UInt8}
scales = cu([UInt8.((1,2,3,4, 2,3,4,5, 3,4,5,6)) for i in 1:1024]) # NTuple{12, UInt8}
CUDA.@sync @cuda threads=32 blocks=32 reinterpret_kernel_MVP(scales) 
@time CUDA.@sync @cuda threads=32 blocks=32 reinterpret_kernel_MVP(scales)
@display @benchmark CUDA.@sync @cuda threads=32 blocks=32 reinterpret_kernel_MVP($scales) 

CUDA.@sync @cuda threads=32 blocks=32 reinterpret_kernel_MVP_pro(scales)
@time CUDA.@sync @cuda threads=32 blocks=32 reinterpret_kernel_MVP_pro(scales)
@display @benchmark CUDA.@sync  @cuda threads=32 blocks=32 reinterpret_kernel_MVP_pro($scales)

#%%
# A solution where the input is already UInt32 converted.
function reinterpret_kernel_MVP_pro2(scales_vec_uint32)
  idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
  kmask1, kmask2, kmask3 = 0x3f3f3f3f, 0x0f0f0f0f, 0x03030303
  if idx <= length(scales_vec_uint32)
    scales_uint32 = scales_vec_uint32[idx]
    utmp0, utmp1, utmp2 = scales_uint32[1], scales_uint32[2], scales_uint32[3]
    mins8 = (utmp1 & kmask1, ((utmp2 >> 4) & kmask2) | (((utmp1 >> 6) & kmask3) << 4))
    utmp1 = (utmp2 & kmask2) | (((utmp0 >> 6) & kmask3) << 4)
    utmp0 &= kmask1
    mins = reinterpret_contiguous(NTuple{8, UInt8}, mins8)
  end
  nothing
end
scales_uint32 = cu([reinterpret(NTuple{3, UInt32}, UInt8.((1,2,3,4, 2,3,4,5, 3,4,5,6))) for i in 1:1024]) # NTuple{12, UInt8}
CUDA.@sync @cuda threads=32 blocks=32 reinterpret_kernel_MVP_pro2(scales_uint32)
@time CUDA.@sync @cuda threads=32 blocks=32 reinterpret_kernel_MVP_pro2(scales_uint32)
@display @benchmark CUDA.@sync  @cuda threads=32 blocks=32 reinterpret_kernel_MVP_pro2($scales_uint32)
#%% KernelAbstraction kernel for the MVP
@kernel function reinterpret_ka_MVP(scales_vec)
end
#%% KernelAbstraction kernel for the MVP pro
@inline function reinterpret_contiguous(::Type{T}, val::U) where {T,U}
  box = Ref(val)
  ptr = Base.unsafe_convert(Ptr{U}, box)
  return unsafe_load(convert(Ptr{T}, ptr))
end
syncKA = KernelAbstractions.synchronize
@kernel function reinterpret_ka_MVP_pro(scales_vec)
  idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
  kmask1, kmask2, kmask3 = 0x3f3f3f3f, 0x0f0f0f0f, 0x03030303
  if idx <= length(scales_vec)
    asdf = (scales_vec[1, idx], scales_vec[2, idx], scales_vec[3, idx], scales_vec[4, idx], scales_vec[5, idx], scales_vec[6, idx], scales_vec[7, idx], scales_vec[8, idx], )
    # scales_vec_uint32 = reinterpret(NTuple{2,UInt32}, asdf)
    # scales = scales_vec[1, idx]
    # utmp0, utmp1, utmp2 = scales[1, idx], scales[2], scales[3]
    # # scales = reinterpret(NTuple{8,UInt8}, (utmp0, utmp1))
    # mins8 = (utmp1 & kmask1, ((utmp2 >> 4) & kmask2) | (((utmp1 >> 6) & kmask3) << 4))
    # utmp1 = (utmp2 & kmask2) | (((utmp0 >> 6) & kmask3) << 4)
    # utmp0 &= kmask1
    # mins = reinterpret(CuVector{UInt8}, mins8) # NTuple{8, UInt8}
  end
end
scales = cu(hcat([UInt8.([1,2,3,4, 2,3,4,5, 3,4,5,6]) for i in 1:1024]...)) # NTuple{12, UInt8}
kernel = reinterpret_ka_MVP_pro(get_backend(scales), 32)
kernel(scales, ndrange=1024); syncKA(get_backend(scales))
@time kernel(scales, ndrange=1024); syncKA(get_backend(scales))
@display @benchmark kernel(scales, ndrange=1024); syncKA(get_backend(scales))


