using Pkg; Pkg.activate(@__DIR__)
using LogicCircuits
using ProbabilisticCircuits
using ProbabilisticCircuits: CuBitsProbCircuit, loglikelihood, full_batch_em, mini_batch_em
using MLDatasets
using CUDA
using Combinatorics
using SplitApplyCombine
using TikzPictures;
import Plots
include("utils.jl")
using Dates
using NPZ

using Debugger
dataset_id=1

train_cpu_t, valid_data, test_cpu_t = twenty_datasets(twenty_dataset_names[dataset_id])   

function main()
    
    train_cpu=Matrix(train_cpu_t)
    train_cpu = train_cpu[1:100,1:7]
    n = size(train_cpu)[2]
    k = maximum(sum(Int.(train_cpu),dims=2))



    println("size of train cpu")
    println(size(train_cpu))
    # 
    # # 
    test_cpu = Matrix(test_cpu_t)

    latents = 2
    pseudocount = 0.01

    @time pc = hclt(train_cpu, latents; pseudocount, input_type = Literal);
    convert_product_to_binary(pc)
    compute_scope(pc)

    # print_scope(pc)

    #group_splitting
    group_size = 5 #group_num = ceil(num_vars/group_size)
    group_num = Int(ceil(n/group_size))

    @assert group_num ==2 #check if group num is desired, if not adjust group_size!
    global_scope = pc.scope

    splitted, var_group_map=split_rand_vars(global_scope,group_size)

    # println(var_group_map)
    # ll=loglikelihood_k_ones(pc,n,k)


    #train prior wrt to splited group

    ks = compute_ks(train_cpu, splitted)


    log_k_likelihood_wrt_split(pc, var_group_map,ks)




#hclt prior wrt splited group

# plot_pc(pc)

end

main()



