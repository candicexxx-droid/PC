using LogicCircuits
using ProbabilisticCircuits
using ProbabilisticCircuits: CuBitsProbCircuit, loglikelihood, full_batch_em, mini_batch_em
using MLDatasets
using CUDA
using Combinatorics
using SplitApplyCombine
using TikzPictures;




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


function mnist_gpu()
    cu.(mnist_cpu())
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



function loglikelihood_k_ones(root::ProbCircuit,n, k; Float=Float32)
    println("Using modified loglikelihood")
    
    # cache = Dict()
    f_i(node) = begin
        result = Array{Union{Float64, Nothing}, 1}(nothing, (k+1))
        result[1] = 0
        result[2] = 1
        result
        # val = data[example_id, first(randvars(node))
        # ismissing(val) ? Float(0.0) : loglikelihood(dist(node), val)
    end
    f_s(node, ins) = begin #ins -> a vector of children outpus, each element of vector is of type Array{Union{Float64, Nothing}, 1}
        result = Array{Union{Float64, Nothing}, 1}(nothing, (k+1))
        flag=true
        for i in 0:k #mapping: 0~k -> 1~k+1
            # child_result_l = ins[1]
            # child_result_r = ins[2]
            for child_result in ins
                if(child_result[i+1]==nothing)
                    flag=false
                end
            end
                 
            if flag
                child_sum = [child_result[i+1] for child_result in ins]
                result[i+1] = reduce(logsumexp, node.params .+ child_sum) 
            end                  
        end
        result
        # reduce(logsumexp, node.params .+ ins) #sum node     
    end
    
    f_m(node, ins) = begin
        result = Array{Union{Float64, Nothing}, 1}(nothing, (k+1))
        for i in 0:k #mapping: 0~k -> 1~k+1
            child_result_l = ins[1]
            child_result_r = ins[2]
            if child_result_l[i+1]!=nothing && child_result_r[k-i+1]!=nothing
                child_sum = (child_result_l[i+1],child_result_r[k-i+1])
                result[i+1] = sum(child_sum)
            end                  
        end
        result
    end
    
    # sum(ins) #product node 
    
    foldup_aggregate(root, f_i, f_m, f_s, Array{Union{Float64, Nothing}, 1})
end

# k = 8
# children = 3

function convert_product_to_binary(root::ProbCircuit)

    f_i(node) = begin
        1
    end
    f_m(node, ins) = begin
        
        function convert_bin(inputs)    
            if length(inputs) == 2 #base case
                println("base!")
                bin_mul = multiply(inputs)
            else 
                # println("length of input is $(length(inputs))")
                bin_mul = multiply([inputs[1],convert_bin(inputs[2:end])])
            
            end
            
        end
        if length(node.inputs)>2
            println("number of children of this prod node is $(length(node.inputs))")
            println("should add  $(length(node.inputs)-2) nodes")
            # println(node.inputs)
            node.inputs = [node.inputs[1],convert_bin(node.inputs[2:end])]
            println("after conversion: number of children of this prod node is $(length(node.inputs))")
        end
        1
    end
    f_s(node, ins) = begin
        1
    end
    foldup_aggregate(root, f_i, f_m, f_s,Int64)
end


#####testing#####
if abspath(PROGRAM_FILE) == @__FILE__
    X1=InputNode(1, Literal(true))
    X2=InputNode(2, Literal(true))
    X3=InputNode(3, Literal(true))
    X4=InputNode(4, Literal(true))
    pc=multiply([X1,X2,X3,X4])
    pc.inputs
    println("Number of free parameters: $(num_parameters(pc))")
    println("Number of nodes: $(num_nodes(pc)) before conversion ")
    TikzPictures.standaloneWorkaround(true)  # workaround
    z=plot(pc);
    save(PDF("plot_before"), z);
    convert_product_to_binary(pc)

    println("Number of nodes: $(num_nodes(pc)) after conversion ")

    TikzPictures.standaloneWorkaround(true)  # workaround
    z=plot(pc);
    save(PDF("plot_after"), z);
end
# X1, X2, X3 = literals(ProbCircuit, 3)
