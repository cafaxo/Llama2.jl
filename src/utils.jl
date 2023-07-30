struct ByteViewIterator
    string::String
    bytes::Vector{UInt8}
end

ByteViewIterator(string::String) = ByteViewIterator(string, Vector{UInt8}(string))

function Base.iterate(iterator::ByteViewIterator, state=1)
    if state > ncodeunits(iterator.string)
        return nothing
    end

    next_state = nextind(iterator.string, state)

    return view(iterator.bytes, state:(next_state-1)), next_state
end

Base.IteratorSize(::Type{ByteViewIterator}) = Base.SizeUnknown()

Base.eltype(::Type{ByteViewIterator}) = SubArray{UInt8,1,Vector{UInt8},Tuple{UnitRange{Int}},true}
