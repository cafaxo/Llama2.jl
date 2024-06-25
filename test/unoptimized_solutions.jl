# There are many other parts.
# But to not forget:
# in vecdot_q4 and vecdot_q5 kernels.

using CUDA
using Llama2: extract_bytes

## 1.
# The unoptimized kernel for q4
function vecdot_q4_kernel_V1(scales)
  idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
  
  kmask1, kmask2, kmask3 = 0x3f3f3f3f, 0x0f0f0f0f, 0x03030303
  if idx <= 1

    scales_uint32 = (UInt32(scales[1]) | UInt32(scales[2])<<8 | UInt32(scales[3])<<16 | UInt32(scales[4])<<24,
    UInt32(scales[5]) | UInt32(scales[6])<<8 | UInt32(scales[7])<<16 | UInt32(scales[8])<<24,
    UInt32(scales[9]) | UInt32(scales[10])<<8 | UInt32(scales[11])<<16 | UInt32(scales[12])<<24)
    utmp0, utmp1, utmp2 = scales_uint32[1], scales_uint32[2], scales_uint32[3]
    mins8 = (utmp1 & kmask1, ((utmp2 >> 4) & kmask2) | (((utmp1 >> 6) & kmask3) << 4))
    # ... Some changes modification to the utmp0 and utmp1        

    # this is somewhat optimized.
    scales_new = (extract_bytes(utmp0)..., extract_bytes(utmp1)...)
  end
  nothing
end

scales = UInt8.((1,2,3,4, 2,3,4,5, 3,4,5,6))

@cuda threads=1 blocks=1 vecdot_q4_kernel_V1(scales)

## 2.