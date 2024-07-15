using KernelAbstractions

@noinline Base.@assume_effects :total function fieldoffset_sym(::Type{T}, s::Symbol) where {T}
    for i in 1:fieldcount(T)
        if fieldname(T, i) == s
            return fieldoffset(T, i)
        end
    end

    return nothing
end

@inline function unsafe_pointer_to_field(y::AbstractVector, index::Int, field::Symbol)
    T = eltype(fieldtype(eltype(y), field))
    return convert(Ptr{T}, pointer(y, index) + fieldoffset_sym(eltype(y), field))
end

# this is a hack that lets us efficiently load and mutate fields of
# an immutable struct that is stored inside a Vector

struct MutableField{T,P<:AbstractVector} <: AbstractVector{T}
    y::P
    index::Int
    fieldoffset::UInt64
    len::Int
end

Base.length(mf::MutableField) = mf.len
Base.eltype(mf::MutableField{T,P}) where {T,P} = T

@inline function MutableField(::Type{T}, y::P, index::Int, field::Symbol) where {T,P<:AbstractVector}
    return MutableField{T,P}(
        y,
        index,
        fieldoffset_sym(eltype(y), field),
        sizeof(fieldtype(eltype(y), field)) ÷ sizeof(T),
    )
end

function Base.getindex(mf::MutableField{T,P}, i::Int) where {T,P}
    y = mf.y

    GC.@preserve y begin
        p = convert(Ptr{T}, pointer(y, mf.index) + mf.fieldoffset) + (i - 1)*sizeof(T)
        unsafe_load(p)
    end
end

function Base.setindex!(mf::MutableField{T,P}, x::T, i::Int) where {T,P}
    y = mf.y

    GC.@preserve y begin
        p = convert(Ptr{T}, pointer(y, mf.index) + mf.fieldoffset) + (i - 1)*sizeof(T)
        unsafe_store!(p, x)
    end
end

# IDK what this was... some utils...
function get_int_from_uint8_rev(x8::Vector{UInt8}, i32::Int)
    # Calculate the starting index in the x8 array based on i32
    start_idx = sizeof(Int) * i32 + 1  # Julia arrays are 1-indexed

    # Ensure the start index is within the bounds of the x8 array
    if start_idx + 3 > length(x8)
        error("Index out of bounds")
    end

    # Extract two UInt16 values from the x8 array and combine them into a single Int32 value
    x32 = UInt32(0)
    x32 |= UInt32(x8[start_idx]) | (UInt32(x8[start_idx + 1]) << 8)
    x32 |= (UInt32(x8[start_idx + 2]) | (UInt32(x8[start_idx + 3]) << 8)) << 16

    return Int32(x32)  # Convert the UInt32 value to Int32 if necessary
end
function get_int_from_int8_aligned_rev(x8::Vector{Int8}, i32::Int)
    # Calculate the starting index in the x8 array based on i32
    start_idx = sizeof(Int) * i32 + 1  # Adjust for Julia's 1-based indexing

    # Ensure the start index is within the bounds of the x8 array
    if start_idx + 3 > length(x8)
        error("Index out of bounds")
    end

    # Reconstruct the 32-bit integer from four consecutive 8-bit integers in the array
    x32 = (Int32(x8[start_idx]) & 0xFF) |
          ((Int32(x8[start_idx + 1]) & 0xFF) << 8) |
          ((Int32(x8[start_idx + 2]) & 0xFF) << 16) |
          ((Int32(x8[start_idx + 3]) & 0xFF) << 24)

    return x32
end
function get_int_from_uint8(x8::Vector{UInt8}, i32::Int)
    # Calculate the byte index in the x8 array based on i32
    byte_idx = sizeof(Int) * i32 + 1  # Adjust for Julia's 1-based indexing

    # Ensure the byte index is within the bounds of the x8 array
    if byte_idx + sizeof(Int) - 1 > length(x8)
        error("Index out of bounds")
    end

    # Use reinterpret to interpret the selected bytes as Int32
    x32 = reinterpret(Int32, x8[byte_idx:byte_idx + sizeof(Int32) - 1])[1]

    return x32
end
function get_int_from_int8_aligned(x8::Vector{Int8}, i32::Int)
    # Calculate the byte index in the x8 array based on i32
    byte_idx = sizeof(Int) * i32 + 1  # Adjust for Julia's 1-based indexing

    # Ensure the byte index is within the bounds of the x8 array
    if byte_idx + sizeof(Int) - 1 > length(x8)
        error("Index out of bounds")
    end

    # Use reinterpret to interpret the selected bytes as Int32
    x32 = reinterpret(Int32, x8[byte_idx:byte_idx + sizeof(Int32) - 1])[1]

    return x32
end

@kernel function sum_blocks_lmem_kernel(
    @Const(x), sums, 
  )
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
@kernel function sum_blocks_kernel(@Const(x), sums, num_blocks, sum_size)
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
function sum_blocks_ka!(sums, x::AbstractVector{Float32}, block_size::Int=32)
    num_blocks = cld(length(x), block_size)
    backend = KernelAbstractions.get_backend(x)
    # kernel! = sum_blocks_kernel(backend, (block_size,))
    # kernel!(x, sums, num_blocks, block_size, ndrange=num_blocks)
    kernel! = sum_blocks_lmem_kernel(backend, (block_size,))
    kernel!(x, sums, ndrange=length(x))
    return sums
end
function sum_blocks_ka(x::AbstractVector{Float32}, block_size::Int=32)
    num_blocks = cld(length(x), block_size)
    sums = KernelAbstractions.zeros(get_backend(x), Float16, num_blocks)
    # sums = similar(x, Float16, num_blocks)
    sum_blocks_ka!(sums, x, block_size)
  
    return sums
end