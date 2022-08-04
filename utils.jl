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

function loglikelihood_k_ones(root::ProbCircuit,n, k; Float=Float32)

    f_i(node) = begin
        result = Array{Union{Float64, Nothing}, 1}(nothing, (k+1))
        if node.dist.value
            result[1] = log(0)
            result[2] = log(1)
        else
            result[1] = log(1)
            result[2] = log(0)
        end

        
        result
    end
    f_s(node, ins) = begin #ins -> a vector of children outpus, each element of vector is of type Array{Union{Float64, Nothing}, 1}
        result = Array{Union{Float64, Nothing}, 1}(nothing, (k+1))
        
        for i in 0:k #mapping: 0~k -> 1~k+1

            flag=true
            for child_result in ins
                # println("child_result[$(i+1)]: $(child_result[i+1])")
                if(child_result[i+1]==nothing)
                    flag=false
                end
            end
                 
            if flag
                child_sum = [child_result[i+1] for child_result in ins]
                result[i+1] = reduce(logsumexp, node.params .+ child_sum) 
            end                  
        end
        #  println("sum node: $(result)")
        result
        # reduce(logsumexp, node.params .+ ins) #sum node     
    end
    
    f_m(node, ins) = begin
        result = Array{Union{Float64, Nothing}, 1}(nothing, (k+1))
        for i in 0:k #mapping: 0~k -> 1~k+1
            child_result_l = ins[1]
            child_result_r = ins[2]
            # println("left child:$(child_result_l)")
            # println("right child: $(child_result_r)")
            temp=[-Inf]
            for j in 0:i
                if child_result_l[j+1]!=nothing && child_result_r[i-j+1]!=nothing
                    child_sum = child_result_l[j+1]+child_result_r[i-j+1]
                    # println("childsum $(child_sum)")
                    append!(temp,child_sum)
                end  

            end
            # println("temp $(temp)")
            result[i+1] = logsumexp(temp)
                            
        end
        #  println("prod node: $(result)")
        result
    end
    
    # sum(ins) #product node 
    
    final=foldup_aggregate(root, f_i, f_m, f_s, Array{Union{Float64, Nothing}, 1})
    println("final array is $(final)")
    final=logsumexp(final)
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
