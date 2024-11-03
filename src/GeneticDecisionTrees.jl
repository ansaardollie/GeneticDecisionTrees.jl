module GeneticDecisionTrees

# Write your package code here.
using 
  CategoricalArrays, 
  DataFrames, 
  Random, 
  Distributions,
  Graphs,
  NetworkLayout,
  Makie,
  Base.Threads

import AbstractTrees as AT
import StatisticalMeasuresBase as SMB
import StatisticalMeasures as SM


import LearnAPI
import GraphMakie.graphplot!
import Makie.plot!

canonical_left_index(canonical_index::Int) = 2 * canonical_index
canonical_right_index(canonical_index::Int) = 2 * canonical_index + 1
canonical_depth(canonical_index::Int) = Int(floor(log2(canonical_index)))

left_index(branch_number::Int) = 2 * branch_number
right_index(branch_number::Int) = 2 * branch_number + 1
parent_branch(node_index::Int) = div(node_index, 2)
isleftchild(node_index::Int) = node_index != 1 && iseven(node_index)
isrightchild(node_index::Int) = node_index != 1 && isodd(node_index)



abstract type AbstractTreeNode end


mask(node::AbstractTreeNode) = getproperty(node, :mask)
index(node::AbstractTreeNode) = getproperty(node, :index)
number(node::AbstractTreeNode) = getproperty(node, :number)
canonical_index(node::AbstractTreeNode) = getproperty(node, :canonical_index)
depth(node::AbstractTreeNode) = canonical_depth(canonical_index(node))
isleaf(node::AbstractTreeNode) = node isa AbstractLeafNode
isbranch(node::AbstractTreeNode) = node isa AbstractBranchNode
isroot(node::AbstractTreeNode) = isbranch(node) && index(node) == 1


function info(node::T) where {T<: AbstractTreeNode}
  output = IOBuffer()
  print(output, "[Node ")
  print(output, index(node), " ; ")
  if T <: AbstractBranchNode 
    print(output, "Branch ")
  else
    print(output, "Leaf ")
  end
  print(output, number(node), " ; ")
  print(output, sum(mask(node)),  " Observations]")

  return String(take!(output))
end


abstract type AbstractBranchNode <: AbstractTreeNode end

struct ContinuousBranch <: AbstractBranchNode
  feature::Int
  criterion::Float64
  mask::BitVector
  index::Int
  number::Int
  canonical_index::Int
  criterion_mask::BitVector
end

struct CategoricalBranch <: AbstractBranchNode
  feature::Int
  criterion::UInt32
  mask::BitVector
  index::Int
  number::Int
  canonical_index::Int
  criterion_mask::BitVector
end


feature(bn::AbstractBranchNode) = getproperty(bn, :feature)
criterion(bn::AbstractBranchNode) = getproperty(bn, :criterion)
criterion_mask(bn::AbstractBranchNode) = getproperty(bn, :criterion_mask)

branch(feature::Int, criterion::Float64, mask::BitVector, index::Int, number::Int, canonical_index::Int, criterion_mask::BitVector) = ContinuousBranch(feature, criterion, mask, index, number, canonical_index, criterion_mask)
branch(feature::Int, criterion::UInt32, mask::BitVector, index::Int, number::Int, canonical_index::Int, criterion_mask::BitVector) = CategoricalBranch(feature, criterion, mask, index, number, canonical_index, criterion_mask)

function to_string(bn::ContinuousBranch; feature_name::Union{Nothing, String}=nothing, display_node_info::Bool=false)
  output = IOBuffer()


  if isnothing(feature_name)
    print(output, "Feature ", feature(bn))
  else
    print(output, feature_name)
  end
  print(output, " < ", round(criterion(bn), digits=6))

  
  if display_node_info
    print(output, " ",  info(bn) )
  end
  return String(take!(output))
end

function to_string(bn::CategoricalBranch; feature_name::Union{Nothing, String}=nothing, pool::Union{Nothing, CategoricalPool}=nothing, display_node_info::Bool=false)
  output = IOBuffer()

  if isnothing(feature_name)
    print(output, "Feature ", feature(bn))
  else
    print(output, feature_name)
  end
  
  if isnothing(pool)
    print(output, " = ", criterion(bn))
  else
    print(output, " = ", pool[criterion(bn)])
  end

  
  if display_node_info
    print(output, " ",  info(bn) )
  end

  return String(take!(output))
end

Base.print(io::IO, bn::ContinuousBranch) = print(io, to_string(bn))
Base.print(io::IO, bn::ContinuousBranch, feature_name::String) = print(io, to_string(bn; feature_name = feature_name))
Base.print(bn::ContinuousBranch, args...) = print(stdout, bn, args...)
Base.print(io::IO, bn::CategoricalBranch) = print(io, to_string(bn))
Base.print(io::IO, bn::CategoricalBranch, feature_name::String) = print(io, to_string(bn; feature_name=feature_name))
Base.print(io::IO, bn::CategoricalBranch, pool::CategoricalPool) = print(io, to_string(bn; pool=pool))
Base.print(io::IO, bn::CategoricalBranch, feature_name::String, pool::CategoricalPool) = print(io, to_string(bn; feature_name=feature_name, pool=pool)) 
Base.print(bn::CategoricalBranch, args...) = print(stdout, bn , args...)

Base.display(io::IO, bn::AbstractBranchNode, args...) = print(io, bn, args...)
Base.display(bn::AbstractBranchNode, args...) = print(stdout, bn, args...)
Base.show(io::IO, bn::AbstractBranchNode, args...) = print(io, bn, args...)
Base.show(bn::AbstractBranchNode, args...) = print(stdout, bn, args...)



abstract type AbstractLeafNode <: AbstractTreeNode end


struct ContinousLeaf <: AbstractLeafNode
  prediction::Float64
  mask::BitVector
  index::Int
  number::Int
  canonical_index::Int
  isempty::Bool
end

struct CategoricalLeaf <: AbstractLeafNode
  prediction::UInt32
  mask::BitVector
  index::Int
  number::Int
  canonical_index::Int
  isempty::Bool
end

prediction(ln::AbstractLeafNode) = getproperty(ln, :prediction)
prediction(ln::CategoricalLeaf, pool::CategoricalPool) = pool[prediction(ln)]
leaf(prediction::Float64, mask::BitVector, index::Int, number::Int, canonical_index::Int) = ContinousLeaf(prediction, mask, index, number, canonical_index, sum(mask) == 0)
leaf(prediction::UInt32, mask::BitVector, index::Int, number::Int, canonical_index::Int) = CategoricalLeaf(prediction, mask, index, number, canonical_index, sum(mask) == 0)


function to_string(ln::ContinousLeaf; display_node_info::Bool=false)
  output = IOBuffer()

  print(output, "ŷ = ", round(prediction(ln), digits=6))

  if display_node_info
    print(output, " ", info(ln))
  end

  return String(take!(output))
end

function to_string(ln::CategoricalLeaf; pool::Union{Nothing, CategoricalPool}=nothing, display_node_info::Bool=false)
  output = IOBuffer()

  
  if isnothing(pool)
    print(output, "ŷ = Class ", prediction(ln))
  else
    print(output, "ŷ = ", prediction(ln, pool))
  end

  
  if display_node_info
    print(output, " ", info(ln))
  end

  return String(take!(output))
end

Base.print(io::IO, ln::T) where {T<:AbstractLeafNode} = print(io, to_string(ln))
Base.print(ln::T) where {T<:AbstractLeafNode} = print(stdout, ln)
Base.print(io::IO, ln::CategoricalLeaf, pool::CategoricalPool) = print(io, to_string(ln; pool=pool))
Base.print(ln::AbstractLeafNode, pool::CategoricalPool) = print(stdout, ln, pool)

Base.display(io::IO, ln::AbstractLeafNode, args...) = print(io, ln, args...)
Base.display(ln::AbstractLeafNode, args...) = print(stdout, ln, args...)
Base.show(io::IO, ln::AbstractLeafNode, args...) = print(io, ln, args...)
Base.show(ln::AbstractLeafNode, args...) = print(stdout, ln, args...)


function column_types(X::AbstractDataFrame)
	p = ncol(X)
	ct = Vector{DataType}(undef, p)
	for i in 1:p
		ct[i] = eltype(X[!, i])
	end
	return ct
end

function column_samplers(X::AbstractDataFrame)
	p = ncol(X)
	ct = Vector{Union{Distribution, Vector{UInt32}}}(undef, p)

	for i in 1:p
		col = X[!, i]
		ct[i] = eltype(col) <: CategoricalValue ? collect(values(col.pool.invindex)) : Uniform(minimum(col), maximum(col))
	end

	return ct
end

function smart_column_samplers(X::AbstractDataFrame)
  p = ncol(X)
  ct = Vector{Function}(undef, p)
  for i in 1:p
    col = X[:, i]
    ct[i] = (mask::BitVector) -> begin
    subcol = col[mask]
    if eltype(col) <: CategoricalValue
      length(subcol) == 0 && return collect(values(col.pool.invindex))
      return unique(unique(subcol.refs))
    else
      (length(subcol) == 0 || length(unique(subcol)) < 2) && return Uniform(minimum(col), maximum(col))
      return [0.5 * (subcol[i] + subcol[i+1]) for i in eachindex(subcol[1:end-1]) ] #Uniform(minimum(subcol), maximum(subcol))
    end
    end
  end

  return ct
end

function create_branch_randomiser(X::AbstractDataFrame)
	n = nrow(X)
	p = ncol(X)

	feature_selector() = rand(1:p)

	coltypes = column_types(X)
	colsamplers = column_samplers(X)

	criterion_selector(colindex) = rand(colsamplers[colindex])

	function randomiser(current_mask::BitVector)
		selected_feature = feature_selector()
		selected_criterion = criterion_selector(selected_feature)
		leftchildmask = coltypes[selected_feature] <: CategoricalValue ? 
										X[:, selected_feature].refs .== selected_criterion :
										X[!, selected_feature] .< selected_criterion
		rightchildmask = (!).(leftchildmask)
		# leftchildmask .&= current_mask
		# rightchildmask .&= current_mask
		return (feature = selected_feature, criterion = selected_criterion, leftmask = leftchildmask, rightmask = rightchildmask)
	end

	return randomiser
end



function create_smart_branch_randomiser(X::AbstractDataFrame)
	n = nrow(X)
	p = ncol(X)

	feature_selector() = rand(1:p)

	coltypes = column_types(X)
	colsamplers = smart_column_samplers(X)

	function randomiser(current_mask::BitVector)
		selected_feature = feature_selector()
		selected_criterion = rand(colsamplers[selected_feature](current_mask))
		leftchildmask = coltypes[selected_feature] <: CategoricalValue ? 
										X[:, selected_feature].refs .== selected_criterion :
										X[!, selected_feature] .< selected_criterion
		rightchildmask = (!).(leftchildmask)
		# leftchildmask .&= current_mask
		# rightchildmask .&= current_mask
		return (feature = selected_feature, criterion = selected_criterion, leftmask = leftchildmask, rightmask = rightchildmask)
	end

	return randomiser
end


abstract type PredictionKind end

abstract type RegressionPredictionKind <: PredictionKind end
abstract type ClassificationPredictionKind <: PredictionKind end

struct RandomClass <: ClassificationPredictionKind end
struct ModeClass <: ClassificationPredictionKind end


struct RandomValue <: RegressionPredictionKind end
struct MeanValue <: RegressionPredictionKind end
struct MedianValue <: RegressionPredictionKind end
struct MidpointValue <: RegressionPredictionKind end


function create_leaf_predictor(::Type{T}, y::AbstractVector{<:CategoricalValue}) where {T<:ClassificationPredictionKind}
  pool = typeof(y) <: SubArray ? y.parent.pool : y.pool
  vals = collect(values(pool.invindex))
  n_levels = length(vals)
  if T <: RandomClass
    pf = (mask::BitVector) -> rand(vals)
    return pf
  elseif T <: ModeClass
    pf = (mask::BitVector) -> begin
      sum(mask) == 0 && return rand(vals)
      counts = zeros(Int, n_levels)
      subcol = y[mask].refs
      for i in 1:n_levels
        counts[i] = count(==(vals[i]), subcol)
      end
      idx = argmax(counts)
      return vals[idx]
    end
    return pf
  else
    error("$(T) is not a recognised ClassificationPredictionKind.")
  end
end

function create_leaf_predictor(::Type{T}, y::AbstractVector{<:Real}) where {T<:RegressionPredictionKind}
  overallmean = mean(y)
  if T <: RandomValue
    pf = (mask::BitVector) -> rand(y)
    return pf
  elseif T <: MeanValue
    pf = (mask::BitVector) -> begin
      sum(mask) == 0 && return overallmean
      subcol = y[mask]
      return mean(subcol)
    end
    return pf
  elseif T <: MedianValue
    pf = (mask::BitVector) -> begin
      sum(mask) == 0 && return overallmean
      subcol = y[mask]
      return median(subcol)
    end
    return pf
  elseif T<:MidpointValue
    pf = (mask::BitVector) -> begin
      sum(mask) == 0 && return overallmean
      subcol = y[mask]
      return 0.5 * (maximum(subcol) + minimum(subcol))
    end
    return pf
  else
    error("$(T) is not a recognised RegressionPredictionKind.")
  end
end



const RegressionNode = Union{ContinousLeaf, ContinuousBranch, CategoricalBranch}
const ClassificationNode = Union{CategoricalLeaf, ContinuousBranch, CategoricalBranch}

abstract type ChildDirection end
struct LeftChild <: ChildDirection end
struct RightChild <: ChildDirection end

function random_tree_nodes(::Type{T}, X::AbstractDataFrame, y::AbstractVector; kwargs...) where {T<:Union{RegressionNode, ClassificationNode}}

  max_depth = get(kwargs, :max_depth, 5)
  split_probability = get(kwargs, :split_probability, 0.5)
  smart_branches = get(kwargs, :smart_branches, false)
  prediction_kind = get(kwargs, :prediction_kind, T <: RegressionNode ? MeanValue : RandomClass)
  
  
  branch_randomiser = smart_branches ? create_smart_branch_randomiser(X) :  create_branch_randomiser(X)
  leaf_predictor = create_leaf_predictor(prediction_kind, y)

  nodes = Vector{T}()
  branches_per_level = Dict{Int, Int}(0 => 1)
  leaves_per_level = Dict{Int, Int}(0 => 0)
  branch_map = Dict{Int, Int}(1 => 1)
  leaf_map = Dict{Int, Int}()
  for l in 1:max_depth
    branches_per_level[l] = 0
    leaves_per_level[l] = 0
  end

  nodes_to_process = Vector{Any}()

  root_mask = BitVector(ones(nrow(X)))
  root_info = branch_randomiser(root_mask)

  root = branch(root_info.feature, root_info.criterion, root_mask, 1, 1, 1, root_info.leftmask)
  push!(nodes, root)

  push!(nodes_to_process, (parent_node_index = 1, parent_branch_number = 1, parent_level = 0, child_mask = root_info.leftmask, direction=LeftChild, parent_ci = 1))
  push!(nodes_to_process, (parent_node_index = 1, parent_branch_number = 1, parent_level = 0, child_mask = root_info.rightmask, direction=RightChild, parent_ci = 1))

  get_branch_number(bpl, current_level) = begin
    running_total = 0
    for l in 0:(current_level - 1)
      running_total += bpl[l]
    end

    return running_total + bpl[current_level]
  end

  get_leaf_number(lpl, current_level) = begin
    running_total = 0
    for l in 0:(current_level-1)
      running_total += lpl[l]
    end
    return running_total + lpl[current_level]
  end

  while length(nodes_to_process) > 0
    parent_node_index, parent_branch_number, parent_level, child_mask, direction, parent_ci = popfirst!(nodes_to_process)

    current_level = parent_level + 1
    current_node_index = direction <: LeftChild ? left_index(parent_branch_number) : right_index(parent_branch_number)
    current_node_ci = direction <: LeftChild ? canonical_left_index(parent_ci) : canonical_right_index(parent_ci)

    can_split = current_level < max_depth
    make_branch = can_split ? rand() < split_probability : false

    if make_branch
      branches_per_level[current_level] += 1
      current_branch_number = get_branch_number(branches_per_level,current_level)

      branch_info = branch_randomiser(child_mask)
      branch_node = branch(branch_info.feature, branch_info.criterion, child_mask, current_node_index, current_branch_number, current_node_ci, branch_info.leftmask)
      push!(nodes, branch_node)
      branch_map[current_branch_number] = current_node_index

      push!(nodes_to_process, (
        parent_node_index = current_node_index, 
        parent_branch_number = current_branch_number, 
        parent_level = current_level, 
        mask = branch_info.leftmask .& child_mask, 
        direction=LeftChild,
        parent_ci = current_node_ci
        )
      )

      push!(nodes_to_process, (
        parent_node_index = current_node_index, 
        parent_branch_number = current_branch_number, 
        parent_level = current_level, 
        mask = branch_info.rightmask .& child_mask, 
        direction=RightChild,
        parent_ci = current_node_ci
        )
      )
    else
      leaves_per_level[current_level] += 1
      current_leaf_number = get_leaf_number(leaves_per_level, current_level)
      leaf_prediction = leaf_predictor(child_mask)
      leaf_node = leaf(leaf_prediction, child_mask, current_node_index, current_leaf_number, current_node_ci)
      push!(nodes, leaf_node)
      leaf_map[current_leaf_number] = current_node_index
    end

  end

  return (nodes = nodes, branch_map = branch_map, leaf_map = leaf_map)

end


const RegressionData = @NamedTuple{X::AbstractDataFrame, y::AbstractVector{Float64}}
const ClassificationData = @NamedTuple{X::AbstractDataFrame, y::AbstractVector{<:CategoricalValue}}

abstract type AbstractDecisionTree{T<:AbstractLeafNode} end

struct RegressionTree <: AbstractDecisionTree{ContinousLeaf}
  nodes::Vector{RegressionNode}
  branch_map::Dict{Int, Int}
  leaf_map::Dict{Int, Int}
  data::RegressionData
  max_depth::Int
end

struct ClassificationTree <: AbstractDecisionTree{CategoricalLeaf}
  nodes::Vector{ClassificationNode}
  branch_map::Dict{Int, Int}
  leaf_map::Dict{Int, Int}
  data::ClassificationData
  max_depth::Int
end

treetype(::Type{T}) where {T<:AbstractDecisionTree} = T <: ClassificationTree ? "Classification" : "Regression"
treetype(::T) where {T<:AbstractDecisionTree} = treetype(T)

nodes(tree::T) where {T<: AbstractDecisionTree} = getproperty(tree, :nodes)
branchmap(tree::T) where {T<: AbstractDecisionTree} = getproperty(tree, :branch_map)
leafmap(tree::T) where {T<:AbstractDecisionTree} = getproperty(tree, :leaf_map)
max_depth(tree::T) where {T<:AbstractDecisionTree} = getproperty(tree, :max_depth)
features(tree::T) where {T<:AbstractDecisionTree} = getproperty(tree, :data).X
targets(tree::T) where {T<:AbstractDecisionTree} = getproperty(tree, :data).y

featurenames(tree::T) where {T<:AbstractDecisionTree} = names(features(tree))
outcomespool(tree::ClassificationTree) = typeof(targets(tree)) <: SubArray ? targets(tree).parent.pool : targets(tree).pool

function featurepool(tree::T, feature_index::Int) where {T<:AbstractDecisionTree}
  eltype(features(tree)[!, feature_index]) <: CategoricalValue || error("Selected feature `$(featurenames(tree)[feature_index])` is not categorical.")

  col = features(tree)[!, feature_index]
  if typeof(col) <: SubArray
    return col.parent.pool
  else
    return col.pool
  end
end

Base.getindex(tree::T, index::Int) where {T<:AbstractDecisionTree} = getindex(nodes(tree), index)
Base.setindex!(tree::T, node, index::Int) where {T<:AbstractDecisionTree} = setindex!(nodes(tree), node, index)
Base.lastindex(tree::T) where {T<:AbstractDecisionTree} = lastindex(nodes(tree))
Base.firstindex(tree::T) where {T<:AbstractDecisionTree} = firstindex(nodes(tree))

struct TreeNodeValue
  index::Int
  number::Int
  data::String
  isbranch::Bool
  nobs::Int
end

function Base.display(io::IO, tnv::TreeNodeValue)
  print(io, "#", tnv.index, ") ", tnv.data, " [", tnv.isbranch ? "Branch " : "Leaf ", tnv.number , " | n = ", tnv.nobs,"]")
  # print(io, tnv.data)
end

function Base.display(tnv::TreeNodeValue)
  display(stdout, tnv)
end

Base.show(io::IO, tnv::TreeNodeValue) = display(io, tnv)
Base.show(tnv::TreeNodeValue) = show(stdout, tnv)
Base.print(io::IO, tnv::TreeNodeValue) = display(io, tnv)
Base.print(tnv::TreeNodeValue) = print(stdout, tnv)

function AT.nodevalue(tree::T, node_index) where {T<:AbstractDecisionTree}
  X = features(tree)
  fnames = featurenames(tree)
  node = tree[node_index]

  if node isa ContinuousBranch
    dt=  to_string(node, feature_name = fnames[feature(node)])
  elseif node isa CategoricalBranch
    nodepool = X[:, feature(node)].pool
    dt = to_string(node, pool=nodepool)
  elseif node isa ContinousLeaf
    dt = to_string(node)
  elseif node isa CategoricalLeaf
    dt = to_string(node, pool=outcomespool(tree))
  end

  return TreeNodeValue(
    index(node),
    number(node), 
    dt, 
    node isa AbstractBranchNode,
    sum(mask(node))
  )
end

function nodeplotinfo(tree::T, node_index) where {T<:AbstractDecisionTree}
  X = features(tree)
  fnames = featurenames(tree)
  node = tree[node_index]

  if node isa ContinuousBranch
    dt=  to_string(node, feature_name = fnames[feature(node)])
  elseif node isa CategoricalBranch
    nodepool = X[:, feature(node)].pool
    dt = to_string(node, pool=nodepool, feature_name = fnames[feature(node)])
  elseif node isa ContinousLeaf
    dt = "y = $(round(prediction(node), digits=6))"
  elseif node isa CategoricalLeaf
    dt = "y = $(prediction(node, outcomespool(tree)))"
  end

  return TreeNodeValue(
    index(node),
    number(node), 
    dt, 
    node isa AbstractBranchNode,
    sum(mask(node))
  )
end

function AT.childindices(tree::T, node_index::Int) where {T<:AbstractDecisionTree}
  tree_nodes = nodes(tree)

  node = tree_nodes[node_index]

  if node isa AbstractBranchNode
    branch_number = number(node)
    lc_index = left_index(branch_number)
    rc_index = right_index(branch_number)
    return (lc_index, rc_index)
  else
    return ()
  end
end


function AT.parentindex(tree::T, node_index::Int) where {T<:AbstractDecisionTree}
  if node_index == 1
    return nothing
  else
    parent_branch_number = parent_branch(node_index)
    return branchmap(tree)[parent_branch_number]
  end
end

function AT.nextsiblingindex(tree::T, node_index::Int) where {T<:AbstractDecisionTree}
  parent_branch_number = parent_branch(node_index)
  if isleftchild(node_index)
    return right_index(parent_branch_number)
  else
    return nothing
  end
end

function AT.prevsiblingindex(tree::T, node_index::Int) where {T<:AbstractDecisionTree}
  parent_branch_number = parent_branch(node_index)
  if isrightchild(node_index)
    return right_index(parent_branch_number)
  else
    return nothing
  end
end

function AT.rootindex(tree::T) where {T<:AbstractDecisionTree}
  return 1
end


function Base.display(io::IO, tree::T) where {T<:AbstractDecisionTree} 
  if get(io, :in_vector, false)
    print(io, treetype(tree), " Tree with ", length(keys(branchmap(tree))), " branches and ", length(keys(leafmap(tree))), " nodes")
    return nothing
  elseif get(io, :compact, false)
    print(io, treetype(tree), "Tree(", length(keys(branchmap(tree))), ",", length(keys(leafmap(tree))), ")")
    return nothing
  else
    AT.print_tree(io, AT.IndexNode(tree))
    return nothing
  end

end
Base.display(tree::T)  where {T<:AbstractDecisionTree} = display(stdout, tree)
Base.show(io::IO, tree::T) where {T<:AbstractDecisionTree} = display(io, tree)
Base.show(tree::T) where {T<:AbstractDecisionTree} = show(stdout, tree)
Base.print(io::IO, tree::T) where {T<: AbstractDecisionTree} = display(IOContext(io, :compact => true), tree)

function Base.display(io::IO, trees::AbstractVector{T}) where {T<:AbstractDecisionTree}
  n = length(trees)
  println(io, n, "-element ", treetype(T), " Tree Vector")
  ioctx = IOContext(io, :in_vector => true)
  if n > 10
    for t in trees[1:5]
      print(ioctx, " ")
      println(ioctx, t)
    end
    println(" ⋮")
    for t in trees[end-4:end]
      print(ioctx, " ")
      println(ioctx, t)
    end
  else
    for t in trees[1:end]
      print(ioctx, " ")
      println(ioctx, t)
    end
  end
end
Base.display(trees::AbstractVector{T}) where {T<:AbstractDecisionTree} = display(stdout, trees)
Base.show(io::IO, trees::AbstractVector{T}) where {T<:AbstractDecisionTree} = show(io, trees)
Base.show(trees::AbstractVector{T}) where {T<:AbstractDecisionTree} = show(stdout, trees)
function Base.print(io::IO, trees::AbstractVector{T}) where {T<:AbstractDecisionTree}
  n = length(trees)
  print(io, treetype(T), "Tree[")
  ioctx = IOContext(io, :compact => true)
  if n > 10
    for t in trees[1:5]
      print(ioctx, t)
      print(ioctx, ", ")
    end
    println(" … ")
    for t in trees[end-4:end]
      print(ioctx, ", ")
      print(ioctx, t)
    end
    print(ioctx, "]")
  else
    for t in trees[1:end-1]
      print(ioctx, t)
      print(ioctx, ", ")
    end
    print(ioctx, trees[end])
    print(ioctx, "]")
  end
end


function segment(tree::T, Xnew::AbstractDataFrame) where {T<:AbstractDecisionTree}
  names(Xnew) == names(features(tree)) || errror("New dataset does have the same features as the original data.")

  n = nrow(Xnew)
  ŷ_segments = zeros(Int, n)

  function process_node(current_index::Int, current_mask::BitVector)
    node = tree[current_index]

    if isleaf(node)
      ŷ_segments[current_mask] .= current_index
    else
      branch_number = number(node)
      node_feature = feature(node)
      col = Xnew[!, node_feature]
      if node isa CategoricalBranch
        original_pool = featurepool(tree, node_feature)
        node_criterion = original_pool.levels[criterion(node)]
        lc_mask = col .== node_criterion
        rc_mask = (!).(lc_mask)
      else
        node_criterion = criterion(node)
        lc_mask = col .< node_criterion
        rc_mask = (!).(lc_mask)
      end
      lc_mask .&= current_mask
      rc_mask .&= current_mask
      process_node(left_index(branch_number), lc_mask)
      process_node(right_index(branch_number), rc_mask)
    end
  end

  process_node(1, BitVector(ones(n)))

  return ŷ_segments
end


function predictions(tree::RegressionTree)
  n = nrow(features(tree))

  ŷ = Vector{Float64}(undef, n) 
  for (ln, li) in leafmap(tree)
      node = nodes(tree)[li]
      ŷ[mask(node)] .= prediction(node)
  end

  return ŷ
end

function predictions(tree::RegressionTree, Xnew::AbstractDataFrame)
  n = nrow(Xnew)
  ŷ_segments = segment(tree, Xnew)

  ŷ = Vector{Float64}(undef, n)

  for i in 1:n
    node = tree[ŷ_segments[i]]
    ŷ[i] = prediction(node)
  end
end

function predictions(tree::ClassificationTree)
  n = nrow(features(tree))

  ŷ = CategoricalVector(undef, n; levels=levels(targets(tree)), ordered=outcomespool(tree).ordered)
  for (ln, li) in leafmap(tree)
      node = nodes(tree)[li]
      ŷ[mask(node)] .= prediction(node, outcomespool(tree))
  end 

  return ŷ
end

function predictions(tree::ClassificationTree, Xnew::AbstractDataFrame)
  n = nrow(Xnew)
  ŷ_segments = segment(tree, Xnew)

  tpool = outcomespool(tree)
  ŷ = CategoricalVector(undef, n; levels = tpool.levels, ordered = tpool.ordered)

  for i in 1:n
    node = tree[ŷ_segments[i]]
    ŷ[i] = prediction(node, tpool)
  end

  return ŷ
end






















function Base.convert(::Type{SimpleDiGraph}, tree::T; maxdepth=-1) where {T<:AbstractDecisionTree}
  if maxdepth == -1
    maxdepth = max_depth(tree)
  end

  g = SimpleDiGraph()
  properties = Any[]

  walk_tree!(tree, 1, g, maxdepth, properties)

  return g, properties
end

function walk_tree!(tree::T, node_index::Int, g, depthleft, properties) where {T<:AbstractDecisionTree}

  node = tree[node_index]

  if isleaf(node)
    add_vertex!(g)
    nv::TreeNodeValue = nodeplotinfo(tree, node_index)
    push!(properties, (node=node, label=nv.data, order= "#$(nv.index) [Leaf $(nv.number)]", obs = nv.nobs, isleaf=true))
    return vertices(g)[end]
  else
    add_vertex!(g)

    if depthleft == 0
      push!(properties, (node=nothing, label="...", order=nothing, obs=nothing))
      return vertices(g)[end]
    else
      depthleft -= 1
    end

    current_branch_number = number(node)
    current_vertex = vertices(g)[end]
    nv = nodeplotinfo(tree, node_index)

    push!(properties, (node=node, label=nv.data, order="#$(nv.index) [Branch $(nv.number)]", obs=nv.nobs, isleaf=false))
    child = walk_tree!(tree, left_index(current_branch_number), g, depthleft, properties)

    add_edge!(g, current_vertex, child)

    child = walk_tree!(tree, right_index(current_branch_number), g, depthleft, properties)

    add_edge!(g, current_vertex, child)

    return current_vertex
  end
end

@recipe(PlotDecisionTree) do scene
  Attributes(
    nodecolormap=:rainbow,
    textcolor=RGBf(0,0,0), 
    leafcolor=:darkgreen,
    nodecolor=:white,
    nodesize=150,
    maxdepth=-1
  )
end

function treeplot(tree::T; kwargs...) where {T<:AbstractDecisionTree}
  f,ax,h = plotdecisiontree(tree; kwargs...)
  hidedecorations!(ax)
  hidespines!(ax)
  return f
end

function Makie.plot!(plt::PlotDecisionTree{<:Tuple{<:AbstractDecisionTree}})
  @extract plt leafcolor, textcolor, nodecolormap, nodecolor, nodesize, maxdepth

  tree = plt[1]

  tmpObs = @lift convert(SimpleDiGraph, $tree; maxdepth = $maxdepth)
  graph = @lift $tmpObs[1]
  properties = @lift $tmpObs[2]

  all_labels = @lift [string(p.label) for p in $properties]

  nlabels_color = map(properties, all_labels, leafcolor, textcolor, nodecolormap) do properties, all_labels, leafcolor, textcolor, nodecolormap
    leaf_ix = findall([p.isleaf for p in properties])
    leaf_label_texts = all_labels[leaf_ix]   #[p[1] for p in split.(leaf_labels[leaf_ix], ":")]
    unique_labels = sort(unique(leaf_label_texts))
    inidividual_leaf_colors = resample_cmap(nodecolormap, length(unique_labels))
    nlabels_color = Any[p.isleaf ? leafcolor : textcolor for p in properties]
    for (ix, uLV) = enumerate(unique_labels)
      ixV = leaf_label_texts .== uLV
      nlabels_color[leaf_ix[ixV]] .= inidividual_leaf_colors[ix]
    end
    return nlabels_color
  end


  node_labels = @lift [
    "$(p.order)\n$(p.label)\nn=$(p.obs)"
    for p in $properties
  ]

  graphplot!(
    plt, 
    graph;
    layout = Buchheim(),
    nlabels=node_labels,
    node_size=nodesize,
    node_color=nodecolor,
    nlabels_color=nlabels_color,
    nlabels_align=(:center, :center),
    nlabels_attr=(
      font="Open Sans Bold",
    )
  )

  return plt

end


function random_tree(::Type{T}, X::AbstractDataFrame, y::AbstractVector; kwargs...) where {T<:AbstractDecisionTree}
  max_depth = get(kwargs, :max_depth, 5)

  nodetype = T <: RegressionTree ? RegressionNode : ClassificationNode

  nodes, branch_map, leaf_map = random_tree_nodes(nodetype, X, y; kwargs...)

  if T <: RegressionTree
    return RegressionTree(
      nodes,
      branch_map,
      leaf_map,
      (X=X, y=y),
      max_depth
    )
  elseif T <: ClassificationTree
    return ClassificationTree(
      nodes, 
      branch_map,
      leaf_map,
      (X=X, y=y),
      max_depth
    )
  else
    error("$(T) is not a recognised AbstractDecisionTree.")
  end
end

function random_regression_tree(X::AbstractDataFrame, y::AbstractVector; kwargs...)
  return random_tree(RegressionTree, X, y; kwargs...)
end 

function random_classification_tree(X::AbstractDataFrame, y::AbstractVector; kwargs...)
  return random_tree(ClassificationTree, X, y; kwargs...)
end


function random_trees(::Type{T}, generation_size::Int, X::AbstractDataFrame, y::AbstractVector; kwargs...) where {T<:AbstractDecisionTree}

  tree_tasks = Vector{Task}(undef, generation_size)

  for i in 1:generation_size
    tree_tasks[i] = @spawn random_tree(T, X, y; kwargs...)
  end

  return fetch.(tree_tasks)

end

function random_regression_trees(generation_size::Int, X::AbstractDataFrame, y::AbstractVector; kwargs...)
  return random_trees(RegressionTree,generation_size, X, y; kwargs...)
end 

function random_classification_trees(generation_size::Int, X::AbstractDataFrame, y::AbstractVector; kwargs...)
  return random_trees(ClassificationTree, generation_size,  X, y; kwargs...)
end


dont_export = [
  # :AbstractTreeFitnessKind,
  # :ClassificationTreeFitnessKind,
  # :RegressionTreeFitnessKind,
  # :BinaryF1ScoreFitness,
  # :MultiF1ScoreFitness,
  # :AccuracyFitness,
  # :BalancedAccuracyFitness,
  # :MatthewsCorrelationFitness,
  # :InformednessFitness,
  # :MarkednessFitness
]

abstract type AbstractTreeFitnessKind end
abstract type RegressionTreeFitnessKind <: AbstractTreeFitnessKind end

struct R2Fitness <: RegressionTreeFitnessKind end
struct AdjustedR2Fitness <: RegressionTreeFitnessKind end

abstract type ClassificationTreeFitnessKind <: AbstractTreeFitnessKind end

struct BinaryF1ScoreFitness <: ClassificationTreeFitnessKind end
struct MultiF1ScoreFitness <: ClassificationTreeFitnessKind end
struct AccuracyFitness <: ClassificationTreeFitnessKind end
struct BalancedAccuracyFitness <: ClassificationTreeFitnessKind end
struct MatthewsCorrelationFitness <: ClassificationTreeFitnessKind end
struct InformednessFitness <: ClassificationTreeFitnessKind end
struct MarkednessFitness <: ClassificationTreeFitnessKind end

const truepositiverates = SM.MulticlassTruePositiveRate(; average=SM.NoAvg(), return_type=Vector)
const truenegativerates = SM.MulticlassTrueNegativeRate(; average=SM.NoAvg(), return_type=Vector)

const positivepredictionvalues = SM.MulticlassPositivePredictiveValue(; average=SM.NoAvg(), return_type=Vector)
const negativepredictionvalues = SM.MulticlassNegativePredictiveValue(; average=SM.NoAvg(), return_type=Vector)

function informedness(cm::SM.ConfusionMatrices.ConfusionMatrix; kwargs...)
  n = sum(cm.mat)
  prevalances = (sum(cm.mat, dims=1)./n)[1,:]
  
  tp_rates = truepositiverates(cm)
  tn_rates = truenegativerates(cm)
  informedess_by_class = (tp_rates + tn_rates) .- 1

  return sum(informedess_by_class .* prevalances)
end

informedness(ŷ, y; kwargs...) = informedness(SM.confusion_matrix(ŷ, y); kwargs...)


function markedness(cm::SM.ConfusionMatrices.ConfusionMatrix; kwargs...)
  n = sum(cm)
  biases = (sum(cm.mat, dims=2) ./ n)[:, 1]
  pp_values = positivepredictionvalues(cm)
  np_values = negativepredictionvalues(cm)

  markedness_by_class = (pp_values + np_values) .- 1

  return sum(markedness_by_class .* biases)
end

markedness(ŷ, y; kwargs...) = markedness(SM.confusion_matrix(ŷ, y); kwargs...)

SMB.is_measure(::typeof(informedness)) = true
SMB.is_measure(::typeof(markedness)) = true
SMB.kind_of_proxy(::typeof(informedness)) = LearnAPI.LiteralTarget()
SMB.kind_of_proxy(::typeof(markedness)) = LearnAPI.LiteralTarget()


function create_fitness_function(::Type{T}) where {T<:ClassificationTreeFitnessKind}
  if T == BinaryF1ScoreFitness
    m = SM.FScore()
    return (ŷ, y; kwargs...) -> m(ŷ, y)
  elseif T == MultiF1ScoreFitness
    m = SM.MulticlassFScore()
    return (ŷ, y; kwargs...) -> m(ŷ, y)
  elseif T == AccuracyFitness
    m = SM.Accuracy()
    return (ŷ, y; kwargs...) -> m(ŷ, y)
  elseif T == BalancedAccuracyFitness
    m = SM.BalancedAccuracy()
    return (ŷ,y; kwargs...) -> m(ŷ, y)
  elseif T == MatthewsCorrelationFitness
    m = SM.MatthewsCorrelation()
    return (ŷ,y; kwargs...) -> m(ŷ, y)
  elseif T == InformednessFitness
    return informedness
  elseif T == MarkednessFitness
    return markedness
  else
    error("$(T) is not a recognised ClassificationTreeFitnessKind.")
  end
end


function r_squared(ŷ, y; kwargs...)
  ss_total = sum(( y .- mean(y)) .^ 2)
  ss_residuals = sum((y - ŷ) .^ 2)

  return 1 - (ss_residuals / ss_total)
end

function adjusted_r_squared(ŷ, y; kwargs...)
  ss_total = sum((y .- mean(y)).^2)
  ss_residuals = sum((y - ŷ).^2)

  tree = get(kwargs, :tree)
  n = nrow(features(tree))
  p = length(nodes(tree)) + length(branchmap(tree))

  df_total = n - 1
  df_residuals = n - p - 1

  return 1 - (( ss_residuals / df_residuals ) / ( ss_total / df_total ))
end

function create_fitness_function(::Type{T}) where {T<:RegressionTreeFitnessKind}
  if T == R2Fitness
    return r_squared
  elseif T == AdjustedR2Fitness
    return adjusted_r_squared
  else
    error("$(T) is not a recognised RegressionTreeFitnessKind.")
  end
end

abstract type TreeComplexityPenaltyKind end

struct TreeDepthPenalty <: TreeComplexityPenaltyKind end
struct TreeNodesPenalty <: TreeComplexityPenaltyKind end


function fitness(tree::T; kwargs...) where {T<:AbstractDecisionTree}
  default_kind = T <: RegressionTree ? R2Fitness : InformednessFitness
  fitness_kind = get(kwargs, :fitness_kind, default_kind)
  fitness_function = get(kwargs, :fitness_function, create_fitness_function(fitness_kind))
  maxdepth = get(kwargs, :max_depth, max_depth(tree))
  targetdepth = get(kwargs, :target_depth, Int(floor(sqrt(maxdepth))))
  maxnodes = get(kwargs, :max_nodes, 2^(maxdepth + 1) -1)
  targetnodes = get(kwargs, :target_nodes, 2^(targetdepth + 1) -1)
  complexity_penalty_kind = get(kwargs, :complexity_penalty, TreeDepthPenalty)
  complexity_penalty_weight = get(kwargs, :complexity_penalty_weight, 0.0)
  empty_leaf_penalty_weight = get(kwargs, :empty_leaf_penalty_weight, 0.0)

  y = targets(tree)
  ŷ = predictions(tree)

  core_fitness = fitness_function(ŷ, y; tree=tree)

  tree_depth = depth(nodes(tree)[end])
  if complexity_penalty_weight != 0 
    complexity_penalty = complexity_penalty_kind == TreeDepthPenalty ? 
                       (maximum([0, tree_depth - targetdepth]) / (maxdepth - targetdepth)) :
                       (maximum([0, length(nodes(tree)) - targetnodes]) / (maxnodes - targetnodes))
  else
    complexity_penalty = 0.0
  end
  penalty = complexity_penalty_weight * complexity_penalty

  if empty_leaf_penalty_weight != 0
    n_leaves = length(leafmap(tree))
    n_empty_leaves = 0
    for (ln, li) in leafmap(tree)
      leaf = tree[li]
      if leaf.isempty
        n_empty_leaves += 1
      end
    end
    empty_leaf_penalty = n_empty_leaves / n_leaves
  else
    empty_leaf_penalty = 0.0
  end
  penalty += empty_leaf_penalty_weight * empty_leaf_penalty

  penalised_fitness = core_fitness - penalty

  return (core = core_fitness, raw = penalised_fitness)
end

function fitness(trees::AbstractVector{T}; kwargs...) where {T<:AbstractDecisionTree}
  n = length(trees)
  tasks = Vector{Task}(undef, n)
  fvals = Matrix{Float64}(undef, n, 2)

  for i in 1:n
    tasks[i] = @spawn fitness(trees[i]; kwargs...)
  end

  finished_tasks = fetch.(tasks)

  fvals[:, 1] = getindex.(finished_tasks, :core)
  fvals[:, 2] = getindex.(finished_tasks, :raw)

  return fvals
end

function scaled_fitness(trees::AbstractVector{T}, temp::Float64; kwargs...) where {T<:AbstractDecisionTree}
  n = length(trees)
  fitness_values = Matrix(undef, n, 3)
  
  fitness_values[:, 1:2] .= fitness(trees; kwargs...)
  raw_fitness_values = view(fitness_values, :, 2)

  normaliser_kernel = (f) -> exp(f*temp)
  max_raw_fitness = maximum(raw_fitness_values)
  min_raw_fitness = minimum(raw_fitness_values)
  ub = normaliser_kernel(max_raw_fitness)
  lb = normaliser_kernel(min_raw_fitness)
  normaliser = (f) -> (normaliser_kernel(f) - lb) / (ub - lb)

  fitness_values[:, 3] = normaliser.(fitness_values[:, 2])

  sort_order_tasks = Vector{Task}(undef, 3)
  for i in 1:3
    sort_order_tasks[i] = @spawn sortperm(fitness_values[:, i], rev=true)
  end

  return (fitness_values, fetch.(sort_order_tasks)...)
end



function crossover(mother::T, father::T; ni1::Union{Int, Nothing}=nothing,ni2::Union{Int, Nothing}=nothing, kwargs...) where {T<:AbstractDecisionTree}
  mother_swap_index = something(ni1, rand(2:length(nodes(mother))))
  father_swap_index = something(ni2, rand(1:length(nodes(father))))

  prediction_kind = get(kwargs, :prediction_kind, T <: RegressionTree ? MeanValue : RandomClass)
  maxdepth = get(kwargs, :max_depth, mother.max_depth)

  leaf_predictor = create_leaf_predictor(prediction_kind, targets(mother))

  nodetype = T <: RegressionTree ? RegressionNode : ClassificationNode
  offspring_nodes = Vector{nodetype}()
  branches_per_level = Dict{Int, Int}(0 => 1)
  leaves_per_level = Dict{Int, Int}(0 => 0)
  branch_map = Dict{Int, Int}(1 => 1)
  leaf_map = Dict{Int, Int}()
  for l in 1:maxdepth
    branches_per_level[l] = 0
    leaves_per_level[l] = 0
  end
  
  get_branch_number(bpl, current_level) = begin
    running_total = 0
    for l in 0:(current_level - 1)
      running_total += bpl[l]
    end

    return running_total + bpl[current_level]
  end

  get_leaf_number(lpl, current_level) = begin
    running_total = 0
    for l in 0:(current_level-1)
      running_total += lpl[l]
    end
    return running_total + lpl[current_level]
  end

  nodes_to_process = Vector{Any}()

  mother_root = mother[1]

  n = length(targets(mother))
  root_lc_mask = copy(mother_root.criterion_mask)
  root_rc_mask = (!).(root_lc_mask)
  root = branch(mother_root.feature, mother_root.criterion, BitVector(ones(Int, n)), 1, 1, 1, root_lc_mask)
  push!(offspring_nodes, root)

  mother_lc_index = left_index(1)
  mother_rc_index = right_index(1)
  push!(nodes_to_process, (
    offspring_info = (parent_node_index = 1, parent_branch_number = 1, parent_level = 0, child_mask=root_lc_mask, direction=LeftChild, parent_ci = 1),
    node_to_copy = mother_lc_index == mother_swap_index ? father[father_swap_index] : mother[mother_lc_index],
    from_mother = mother_lc_index != mother_swap_index
  ))

  
  push!(nodes_to_process, (
    offspring_info = (parent_node_index = 1, parent_branch_number = 1, parent_level = 0, child_mask=root_rc_mask, direction=RightChild, parent_ci = 1),
    node_to_copy = mother_rc_index == mother_swap_index ? father[father_swap_index] : mother[mother_rc_index],
    copy_from_mother = mother_rc_index != mother_swap_index
  ))

  while length(nodes_to_process) > 0
    offspring_info, node_to_copy, copy_from_mother = popfirst!(nodes_to_process)

    offspring_current_level = offspring_info.parent_level + 1
    offspring_current_node_index = offspring_info.direction <: LeftChild ? 
                                    left_index(offspring_info.parent_branch_number) : 
                                    right_index(offspring_info.parent_branch_number)
    offspring_current_node_ci = offspring_info.direction <: LeftChild ? 
                                    canonical_left_index(offspring_info.parent_ci) : 
                                    canonical_right_index(offspring_info.parent_ci)
    offspring_mask = offspring_info.child_mask
    
    can_split = offspring_current_level < maxdepth
    node_is_branch = isbranch(node_to_copy)
    
    if can_split && node_is_branch
      branches_per_level[offspring_current_level] += 1
      offspring_current_branch_number = get_branch_number(branches_per_level, offspring_current_level)
      original_branch_number = number(node_to_copy)
      original_branch_feature = feature(node_to_copy)
      original_branch_criterion = criterion(node_to_copy)
      original_branch_lc_mask = copy(criterion_mask(node_to_copy))
      original_branch_rc_mask = (!).(original_branch_lc_mask)

      new_branch = branch(
        original_branch_feature,
        original_branch_criterion,
        offspring_mask,
        offspring_current_node_index,
        offspring_current_branch_number,
        offspring_current_node_ci,
        original_branch_lc_mask
      )

      push!(offspring_nodes, new_branch)
      branch_map[offspring_current_branch_number] = offspring_current_node_index

      lc_index = left_index(original_branch_number)
      rc_index = right_index(original_branch_number)


      push!(nodes_to_process, (
        offspring_info = (
          parent_node_index = offspring_current_node_index, 
          parent_branch_number = offspring_current_branch_number, 
          parent_level = offspring_current_level, 
          child_mask=original_branch_lc_mask .& offspring_mask, 
          direction=LeftChild, 
          parent_ci = offspring_current_node_ci
        ),
        node_to_copy = copy_from_mother ? ( lc_index == mother_swap_index ? father[father_swap_index] : mother[lc_index]  ) : father[lc_index],
        from_mother = copy_from_mother ? (lc_index != mother_swap_index) : false
      ))

      push!(nodes_to_process, (
        offspring_info = (
          parent_node_index = offspring_current_node_index, 
          parent_branch_number = offspring_current_branch_number, 
          parent_level = offspring_current_level, 
          child_mask=original_branch_rc_mask .& offspring_mask, 
          direction=RightChild, 
          parent_ci = offspring_current_node_ci
        ),
        node_to_copy = copy_from_mother ? ( rc_index == mother_swap_index ? father[father_swap_index] : mother[rc_index]  ) : father[rc_index],
        from_mother = copy_from_mother ? (rc_index != mother_swap_index) : false
      ))

    else
      leaves_per_level[offspring_current_level] += 1
      offspring_current_leaf_number = get_leaf_number(leaves_per_level, offspring_current_level)
      if !node_is_branch && (copy_from_mother || prediction_kind == RandomClass)
        leaf_prediction = prediction(node_to_copy)
      else
        leaf_prediction = leaf_predictor(offspring_mask)
      end
      new_leaf = leaf(
        leaf_prediction,
        offspring_mask,
        offspring_current_node_index,
        offspring_current_leaf_number,
        offspring_current_node_ci
      )
      
      push!(offspring_nodes, new_leaf)
      leaf_map[offspring_current_leaf_number] = offspring_current_node_index
    end
  
  
  end

  if T <: RegressionTree
    return RegressionTree(
      offspring_nodes,
      branch_map,
      leaf_map,
      mother.data,
      maxdepth
    )
  else
    return ClassificationTree(
      offspring_nodes,
      branch_map,
      leaf_map,
      mother.data,
      maxdepth
    )
  end

end


function mutate!(tree::T, mutate_index::Int; kwargs...) where {T<:AbstractDecisionTree}
  smart_branches = get(kwargs, :smart_branches, false)
  prediction_kind = get(kwargs, :prediction_kind, T <: RegressionTree ? MeanValue : RandomClass)
  original_node = tree[mutate_index]
  original_mask = mask(original_node)

  
  branch_randomiser = smart_branches ? create_smart_branch_randomiser(features(tree)) :  create_branch_randomiser(features(tree))
  leaf_predictor = create_leaf_predictor(prediction_kind, targets(tree))

  mutate_branch = isbranch(original_node)

  if !mutate_branch
    new_prediction = leaf_predictor(original_mask)
    new_leaf = leaf(new_prediction, original_mask, index(original_node), number(original_node), canonical_index(original_node))
    tree[mutate_index] = new_leaf
  else
    new_branch_info = branch_randomiser(original_mask)
    new_feature = new_branch_info.feature
    new_criterion = new_branch_info.criterion
    new_leftmask = new_branch_info.leftmask
    new_rightmask = new_branch_info.rightmask

    new_branch = branch(new_feature,new_criterion,original_mask, index(original_node), number(original_node), canonical_index(original_node), new_leftmask)
    tree[mutate_index] = new_branch
    child_nodes_to_process = Vector{Any}()

    lci = left_index(number(original_node))
    rci = right_index(number(original_node))

    push!(child_nodes_to_process, (node_index=lci, newmask = original_mask .& new_leftmask))
    push!(child_nodes_to_process, (node_index=rci, newmask = original_mask .& new_rightmask))

    while length(child_nodes_to_process) > 0
      node_index, newmask = popfirst!(child_nodes_to_process)
      current_node = tree[node_index]

      if isleaf(current_node)
        current_node.mask[:] = newmask
        if T <: RegressionTree
          new_prediction = leaf_predictor(newmask)
          new_leaf = leaf(new_prediction, newmask, index(current_node), number(current_node), canonical_index(current_node))
          tree[node_index] = new_leaf
        end
      else
        current_node.mask[:] = newmask

        current_node_leftmask = criterion_mask(current_node)
        current_node_rightmask = (!).(current_node_leftmask)
        current_node_lci = left_index(number(current_node))
        current_node_rci = right_index(number(current_node))

        push!(child_nodes_to_process, (node_index = current_node_lci, newmask = newmask .& current_node_leftmask))
        push!(child_nodes_to_process, (node_index = current_node_rci, newmask = newmask .& current_node_rightmask))
      end
    end

  end

  return tree
end

function mutate!(tree::T; kwargs...) where {T<:AbstractDecisionTree}
  num_mutations = get(kwargs, :num_mutations, 1)


  available_node_indices = T <: RegressionTree ? collect(values(branchmap(tree))) : 1:length(nodes(tree))
  mutated_node_indices = rand(available_node_indices, num_mutations)

  for mni in mutated_node_indices
    mutate!(tree, mni; kwargs...)
  end

  return tree
end

abstract type AbstractSelectionPressure end

struct StochasticUniversalSampling <: AbstractSelectionPressure end
struct FitnessProportionateSampling <: AbstractSelectionPressure end

function evolve(::Type{StochasticUniversalSampling}, trees::AbstractVector{T}, temp::Float64; kwargs...) where {T<:AbstractDecisionTree}
  fv_task = @spawn scaled_fitness(trees, temp; kwargs...)
  elite_proportion = get(kwargs, :elite_proportion, 0.0)
  mutation_probability = get(kwargs, :mutation_probability, 0.5)
  N = length(trees)

  offspring = Vector{T}(undef, N)
  elite_indices = 1:Int(floor(N * elite_proportion))

  fitness_values, core_fitness_sortperm, raw_fitness_sortperm, scaled_fitness_sortperm = fetch(fv_task)
  offspring[elite_indices] .= trees[core_fitness_sortperm[elite_indices]]

  sorted_trees = view(trees, scaled_fitness_sortperm)
  sus_fitness_values = fitness_values[scaled_fitness_sortperm, 3]
  total_sus_fitness = sum(sus_fitness_values)
  num_sus_offspring = Int(ceil(N * (1 - elite_proportion)))

  pointer_distance = total_sus_fitness / num_sus_offspring
  fitness_boundaries = [0, cumsum(sus_fitness_values)...]

  mothers_roulette_start = rand(Uniform(0, pointer_distance))
  mothers_roulette_points = [mothers_roulette_start + (1 - 1) * pointer_distance for i in 1:num_sus_offspring]
  mothers_indices = [findfirst(f -> p < f, fitness_boundaries) - 1 for p in mothers_roulette_points]

  fathers_roulette_start = rand(Uniform(0, pointer_distance))
  fathers_roulette_points = [fathers_roulette_start + (1 - 1) * pointer_distance for i in 1:num_sus_offspring]
  fathers_indices = [findfirst(f -> p < f, fitness_boundaries) - 1 for p in fathers_roulette_points]

  mothers = sorted_trees[mothers_indices]
  fathers = sorted_trees[fathers_indices]

  offspring_tasks = Vector{Task}(undef, num_sus_offspring)
  for i in 1:num_sus_offspring
    offspring_tasks[i] = @spawn ((m, f, mutation_prob; kwargs...) -> begin
      child = crossover(m,f; kwargs...)
      if rand() < mutation_prob
        child = mutate!(child;kwargs...)
      end
      return child
    end)(mothers[i], fathers[i], mutation_probability; kwargs...)
  end

  last_elite_index = length(elite_indices) == 0 ? 0 : elite_indices[end]
  offspring[last_elite_index+1:end] .= fetch.(offspring_tasks)

  return offspring
end



function evolve(::Type{FitnessProportionateSampling}, trees::AbstractVector{T}, temp::Float64; kwargs...) where {T<:AbstractDecisionTree}
  fv_task = @spawn scaled_fitness(trees, temp; kwargs...)
  elite_proportion = get(kwargs, :elite_proportion, 0.0)
  mutation_probability = get(kwargs, :mutation_probability, 0.5)
  N = length(trees)

  offspring = Vector{T}(undef, N)
  elite_indices = 1:Int(floor(N * elite_proportion))

  fitness_values, core_fitness_sortperm, raw_fitness_sortperm, scaled_fitness_sortperm = fetch(fv_task)
  offspring[elite_indices] .= trees[core_fitness_sortperm[elite_indices]]

  sorted_trees = view(trees, scaled_fitness_sortperm)
  fps_fitness_values = fitness_values[scaled_fitness_sortperm, 3]
  total_fps_fitness = sum(fps_fitness_values)
  num_fps_offspring = Int(ceil(N * (1 - elite_proportion)))

  fps_probabilities = fps_fitness_values ./ total_fps_fitness

  mothers = sorted_trees[rand(Distributions.Categorical(fps_probabilities), num_fps_offspring)]
  fathers = sorted_trees[rand(Distributions.Categorical(fps_probabilities), num_fps_offspring)]

  offspring_tasks = Vector{Task}(undef, num_fps_offspring)
  for i in 1:num_fps_offspring
    offspring_tasks[i] = @spawn ((m, f, mutation_prob; kwargs...) -> begin
      child = crossover(m,f; kwargs...)
      if rand() < mutation_prob
        child = mutate!(child;kwargs...)
      end
      return child
    end)(mothers[i], fathers[i], mutation_probability; kwargs...)
  end

  last_elite_index = length(elite_indices) == 0 ? 0 : elite_indices[end]
  offspring[last_elite_index+1:end] .= fetch.(offspring_tasks)

  return offspring
end

















for n in names(@__MODULE__; all = true)
    if Base.isidentifier(n) && n ∉ (Symbol(@__MODULE__), :eval, :include, dont_export...)
        @eval export $n
    end
end

end
