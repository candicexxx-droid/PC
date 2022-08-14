using LogicCircuits
using ProbabilisticCircuits
using ProbabilisticCircuits: CuBitsProbCircuit, loglikelihood, full_batch_em, mini_batch_em
using MLDatasets
using CUDA
using Combinatorics
using SplitApplyCombine
using TikzPictures;
using Dates




function mnist_cpu()
    train_int = transpose(reshape(MNIST.traintensor(UInt8), 28*28, :));
    test_int = transpose(reshape(MNIST.testtensor(UInt8), 28*28, :));

    function bitsfeatures(data_int)
        data_bits = zeros(Bool, size(data_int,1), 28*28*8)
        for ex = 1:size(data_int,1), pix = 1:size(data_int,2)
            x = data_int[ex,pix]
            for b = 0:7
                if (x & (one(UInt8) << b)) != zero(UInt8)
                    data_bits[ex, (pix-1)*8+b+1] = true
                end
            end
        end
        data_bits
    end

    train_cpu = bitsfeatures(train_int);
    test_cpu = bitsfeatures(test_int);

    train_cpu, test_cpu
end


function move_to_gpu(train,test)
    CuArray(train),CuArray(test)
end

function enumerate_partitions(num_child, k)
    #not working
    a=0:k
    group_num=num_child -1
    splits = combinedims(collect(combinations(a, group_num))) #group_num * num_eum

    results = zeros(Int64,num_child, size(splits)[2])


    results[1,:] = splits[1,:]
    results[1+group_num,:] = broadcast(-,k,splits[group_num,:])
    for i in 2:group_num
        results[i,:] = splits[i,:]-splits[i-1,:]
    end

    return results
end

function softmax(x) 
    y= exp.(x) ./    
sum(exp.(x))
end

function compute_scope(root::ProbCircuit)
    #assign scope to each node
    f_i(node) =begin
        node.scope = node.randvars
        node.scope
    end

    f_s(node,ins)= begin
        node.scope = ins[1]
        node.scope
    end

    f_m(node,ins)= begin
        # println("compute scope fm")
        # println(ins)
        # println("inputs node at f_m $(node.inputs)")
        for s in ins
            union!(node.scope, s)
        end
        # println(node.scope)
        node.scope
    end

    foldup_aggregate(root, f_i, f_m, f_s, BitSet)

end

function loglikelihood_k_ones(root::ProbCircuit,n, k,; idx_k=nothing,Float=Float32)

    #input nodes: randvar::BitSet, dist
    f_i(node) = begin
        result = ones(k+1)*(-Inf)
        # println("scope node at f_i $(node.scope)")
        if node.dist.value
            result[1] = log(0.0)
            result[2] = log(1.0)
        else
            result[1] = log(1.0)
            result[2] = log(0.0)
        end

        
        result
    end
    f_s(node, ins) = begin #ins -> a vector of children outpus, each element of vector is of type Array{Union{Float64, Nothing}, 1}
        result = ones(k+1)*(-Inf)
        # println("scope node at f_s $(node.scope)")
        for i in 0:k #mapping: 0~k -> 1~k+1
            child_sum = [child[i+1] for child in ins]
            result[i+1] = reduce(logsumexp, node.params .+ child_sum)              
        end
        #  println("sum node: $(result)")
        result
        # reduce(logsumexp, node.params .+ ins) #sum node     
    end
    
    f_m(node, ins) = begin
        result = ones(k+1)*(-Inf)
        # println("scope node at f_m $(node.scope)")
        # println("inputs node at f_m $(node.inputs)")
        for i in 0:k #mapping: 0~k -> 1~k+1
            child_result_l = ins[1]
            child_result_r = ins[2]
            # println("left child:$(child_result_l)")
            # println("right child: $(child_result_r)")
            temp=[-Inf]
            for j in 0:i
                # if child_result_l[j+1]!=nothing && child_result_r[i-j+1]!=nothing
                child_sum = child_result_l[j+1]+child_result_r[i-j+1]
                # println("childsum $(child_sum)")
                append!(temp,child_sum)
                # end  

            end
            # println("temp $(temp)")
            result[i+1] = logsumexp(temp)
                            
        end
        #  println("prod node: $(result)")
        result
    end
    
    # sum(ins) #product node 
    
    final=foldup_aggregate(root, f_i, f_m, f_s, Vector{Float64})
    println("final array is $(final)")
    marginal=sum(exp.(final))
    
    return final, marginal
    
end

# k = 8
# children = 3

function convert_product_to_binary(root::ProbCircuit)
    total_added=0
    f_i(node) = begin
        0
    end
    f_m(node, ins) = begin
        
        function convert_bin(inputs)    
            if length(inputs) == 2 #base case
                # println("base!")
                bin_mul = multiply(inputs)
            else 
                # println("length of input is $(length(inputs))")
                bin_mul = multiply([inputs[1],convert_bin(inputs[2:end])])
            
            end
            
        end
        # added = length(node.inputs)>2 ? (length(node.inputs)-2) : 0
        # added = sum(ins) + added
        if length(node.inputs)>2
            # println("number of children of this prod node is $(length(node.inputs))")
            # println("should add  $(length(node.inputs)-2) nodes")
            # println(node.inputs)
            total_added+=length(node.inputs)-2
            node.inputs = [node.inputs[1],convert_bin(node.inputs[2:end])]
            # println("after conversion: number of children of this prod node is $(length(node.inputs))")
        end
        
        0
        # println("added $(added)")
        # return added
    end
    f_s(node, ins) = begin
        added = sum(ins)

    end
    foldup_aggregate(root, f_i, f_m, f_s,Int64)
    total_added
end


function compute_k_distribution(train_data)
    #compute the number of instances in training_data for each i=0,..,k (k is the hamming weight of training_data)
    A=sum(train_data,dims=2)
    x = [[i, count(==(i), A)] for i in unique(A)]
    
    x = reduce(hcat,x)#convert x to a matrix
    x = x[:,sortperm(x[1,:])]
    # println("x:")
    # println(x)
    k = maximum(sum(Int.(train_data),dims=2))
    temp = zeros(k+1,1)
    temp[x[1,:].+1] = x[2,:]
    result = Int.(hcat(0:k,temp))'
    # println("result:  $(result)")
    # println("x size $(size(x))")
    # result[:,2][x[1,:]] = x[2,:]
    # println("result $result")
    # return result

    # idx = x[1,:] #collect ks that has more than one instances
    return result
    
end

function compute_k_distribution_wrt_split(train_data,splitted,ks;return_reduced_train_data=false,smooth=0.001)
    result = zeros(ks)
    data_size = size(train_data)[1]
    reduced_train_data = zeros(data_size,length(ks)) #represent each example from binary to k1,k2,...,km

    for (data_ind, group) in enumerate(splitted)
        extract_idx = [i for i in group]
        reduced_train_data[:,data_ind] = sum(train_data[:,extract_idx], dims=2)
    end
    if return_reduced_train_data
        #return_reduced_train_data used as index for result-shaped ll
        reduced_train_data = reduced_train_data .+1
        reduced_train_data = [CartesianIndex(Tuple(Int.(reduced_train_data)[i,:])) for i in 1:data_size]
        # return_reduced_train_data = [typeof(Int.(reduced_train_data)[i,:]) for i in 1:size(reduced_train_data)[1]]
        return reduced_train_data
    end
    uniq = unique(reduced_train_data,dims=1)
    
    x = [[CartesianIndex(Tuple(Int.(uniq[i,:].+1))), sum(sum(reduced_train_data.==(reshape(uniq[i,:], (1,size(uniq[i,:])[1]))),dims=2).==length(ks))] for i in 1:size(uniq)[1]]
    x = reduce(hcat,x)#convert x to a matrix
    # println("size(x)")
    # println(size(x))
    for i in 1:size(x)[2]
        idx,val = x[:,i]
        # println(idx)
        # println(val)
        result[idx] = val
    end
    
    result = result.+smooth
    result = result ./sum(result)
    # /data_size
    # println(result)


end

function generate_training_ID(dataset_id)
    timenow = Dates.now()
    time = Dates.format(timenow, "dd-u-yy-HH-MM-SS")
    training_ID = time * "_" * twenty_dataset_names[dataset_id] * "_$(dataset_id)"
    return training_ID
end


function split_rand_vars(scope::BitSet,group_size::Int)
    #split global scope into group_num group_splitting
    # println(scope)
    l = length(scope)
    scope = deepcopy(scope)
    group_num = div(l,group_size)
    mod = l%group_size
    var_group_map=Dict()
    group_num += (mod>0)
    splited = Vector{BitSet}(undef,group_num)
    for i in 1:group_num
        temp=BitSet()
        if mod>0 && i==group_num
            for j in 1:mod
                var = pop!(scope)
                push!(temp,var)
                var_group_map[var] = i
                
            end
        else
            for j in 1:group_size
                # println("group_size")
                # println(group_size)
                # println("forloop")
                # println(scope)
                var = pop!(scope)
                push!(temp,var)
                var_group_map[var] = i
            end
        end
        
        
        
        splited[i] = temp
    end

    
    # println(group_num)
    


    return splited, var_group_map
    # , group_num
end

function compute_ks(train_cpu,splitted)
    ks = []
    for i in 1:size(splitted)[1]
        group = [j for j in splitted[i]]
        s = minimum(group)
        f = maximum(group)
        temp = maximum(sum(Int.(train_cpu[:,s:f]),dims=2))
        append!(ks, temp)
    end
    ks .+= 1 #including k =0 case #ks is (k1+1,k2+1,...)
    ks = Tuple(k for k in ks)

end


function print_scope(root)
    #for debugging
    f_i(node) = begin
        println("Input Node")
        println(node.scope)
        0
    end
     f_s(node, ins) = begin
         println("Sum Node")
         println(node.scope)
         println(node.params)
         0
    end

    f_m(node, ins) = begin
        println("Prod Node")
        println(node.scope)
        0
    end

    
    foldup_aggregate(root, f_i, f_m, f_s,Int)

end

function log_k_likelihood_wrt_split(root, var_group_map,ks)
    #var_group_map: var-> group index
    #ks: max k wrt each group 
    # result[k1+1, k2+1,...,km+1] -> pr(node, k1,k2,...,km)
    #group num = 2
    f_i(node) = begin
        result = ones(ks)*(-Inf)
        var = [i for i in node.randvars]
        g = var_group_map[var[1]]
        idx_zero =[1 for i in 1:length(ks)] #idx  (k1=1,k2=1,...,km=1)
        idx_one = deepcopy(idx_zero)#idx 
        idx_one[g] = 2 #(k1=1,k2=1,...,g=2,..., km=1)
        idx_zero = CartesianIndex(Tuple(idx_zero))
        idx_one = CartesianIndex(Tuple(idx_one))
        if node.dist.value

            result[idx_zero] = log(0.0)
            result[idx_one] = log(1.0)
        else
            result[idx_zero] = log(1.0)
            result[idx_one] = log(0.0)
        end

        return result


        
    end
    f_s(node, ins) = begin #ins -> a vector of children outpus, each element of vector is of type Array{Union{Float64, Nothing}, 1}
        result = ones(ks)*(-Inf)
        all_idxs = CartesianIndices(ks)
        for i in all_idxs #mapping: 0~k -> 1~k+1
            child_sum = [child[i] for child in ins]
            result[i] = reduce(logsumexp, node.params .+ child_sum)              
        end 
        # println("sum")
        # println(result)
        return result
    end
    
    f_m(node, ins) = begin
        result = ones(ks)*(-Inf)
        
        groups = unique([var_group_map[i] for i in node.scope])
        local_ks = [ i in groups ? i : 1 for i in ks]
        all_idxs = CartesianIndices(Tuple(i for i in local_ks))
        # scope_l = node.inputs[1].scope
        # scope_r = node.inputs[2].scope


        # group_l = unique([var_group_map[i] for i in scope_l])
        # group_r = unique([var_group_map[i] for i in scope_r])
        # println("group l $(group_l)")
        # println("group r $(group_r)")

        # group = unique([var_group_map[i] for i in node.scope])
        # println("group  $(group)")

        child_result_l = ins[1]
        child_result_r = ins[2]
        for i in all_idxs
            temp=[-Inf]
            # println("======curr i========")
            # println(i)
            # println("======curr i========")
            sub_all_idxs = CartesianIndices(i)
            # println("sub_all_idxs=========")
            # println(sub_all_idxs)
            # println("sub_all_idxs=========")
            # println("children result left =========")
            # println(child_result_l)
            # println("children result right =========")
            # println(child_result_r)
            # println("complement idx =========")
            for j in sub_all_idxs
                #compute complement:
                comp_idx = CartesianIndex(Tuple(i).-Tuple(j).+1)
                # print(comp_idx)
                child_sum  = child_result_l[j] + child_result_r[comp_idx]
                append!(temp,child_sum)

            end 
            # println("\ncomplement idx =========")
            # println("prod temp")
            # println(temp)        
            result[i] = logsumexp(temp)
        end
        # println("prod")
        # println(result)
        return result

    end

    result = foldup_aggregate(root, f_i, f_m, f_s, Array{Float64})
    # final_idx = CartesianIndex(ks)
    marginal=logsumexp(result)
    # println(marginal)
    return result,marginal


end


function plot_pc(pc; save_path="plot")
    TikzPictures.standaloneWorkaround(true)  # workaround
    z = plot(pc);
    save(PDF(save_path), z);

end



#####testing#####
if abspath(PROGRAM_FILE) == @__FILE__
    X1=InputNode(1, Literal(true))
    X2=InputNode(2, Literal(true))
    X3=InputNode(3, Literal(true))
    X4=InputNode(4, Literal(true))

    # pc=summate([multiply([X1,X2,X3,X4]),multiply([X1,X2,X3,X4])])
    pc=summate([multiply([X1,X2,X3]),multiply([X1,X2,X3])])
    # pc = multiply([X1,X2,X3,X4])

    init_parameters(pc);

    println("Number of free parameters: $(num_parameters(pc))")
    println("Number of nodes: $(num_nodes(pc)) before conversion ")
    # TikzPictures.standaloneWorkaround(true)  # workaround
    # z=plot(pc);
    # save(PDF("plot_before"), z);
    added = convert_product_to_binary(pc)

    println("total Number of added nodes: $(added) after conversion ")

    println("Number of nodes: $(num_nodes(pc)) after conversion ")

    # TikzPictures.standaloneWorkaround(true)  # workaround
    # z=plot(pc);
    # save(PDF("plot_after"), z);

    v=loglikelihood_k_ones(pc,3,3)
    println("loglikelihood of <= k is $(v)")
end


# X1, X2, X3 = literals(ProbCircuit, 3)

