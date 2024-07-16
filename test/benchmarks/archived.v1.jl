# This file is for the functions archived at versions.

@inline function extract_bytes(x::UInt32)
  return (x & 0xff, (x >> 8) & 0xff, (x >> 16) & 0xff, (x >> 24) & 0xff)
end