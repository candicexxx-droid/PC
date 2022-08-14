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
using Statistics
using Debugger
dataset_id=1

train_cpu_t, valid_data, test_cpu_t = twenty_datasets(twenty_dataset_names[dataset_id])   

function main()
    
    train_cpu=Matrix(train_cpu_t)
    # train_cpu = train_cpu[1:100,1:2]
    n = size(train_cpu)[2]
    k = maximum(sum(Int.(train_cpu),dims=2))
    println("k is $(k)")
    test_cpu = Matrix(test_cpu_t)
    train_gpu,test_gpu = move_to_gpu(train_cpu,test_cpu)

    training_ID = generate_training_ID(dataset_id)
    log_path = "log/$(training_ID)_splited.txt"
    println("size of train cpu")
    println(size(train_cpu))
    # 
    # # 
    test_cpu = Matrix(test_cpu_t)

    latents = 64
    pseudocount = 0.01

    @time pc = hclt(train_cpu, latents; pseudocount, input_type = Literal);
    convert_product_to_binary(pc)
    compute_scope(pc)



    #group_splitting
    group_size = 60 #group_num = ceil(num_vars/group_size)
    group_num = Int(ceil(n/group_size))

    @assert group_num ==2 #check if group num is desired, if not adjust group_size!
    global_scope = BitSet(1:n)
    # pc.scope

    splitted, var_group_map=split_rand_vars(global_scope,group_size)

    # println(var_group_map)
    # ll=loglikelihood_k_ones(pc,n,k)
    open(log_path, "a+") do io
        write(io, "group size: $group_size; group_num: $group_num\n")
    end;

    #train prior wrt to splited group
    batch_size  = 32
    pseudocount = .005
    softness    = 0


    ks = compute_ks(train_cpu, splitted)
    open(log_path, "a+") do io
        write(io, "computing ks distribution...\n")
    end;
    println("computing ks distribution...")
    ks_train_dist=compute_k_distribution_wrt_split(train_cpu,splitted,ks)
    open(log_path, "a+") do io
        write(io, "ks: $ks\n")
    end;
    print("Moving circuit to GPU... ")
    CUDA.@time bpc = CuBitsProbCircuit(pc)
     #move circuit to gpu
    function training()
        

        
        

        print("First round of minibatch EM... ")
        CUDA.@time mini_batch_em(bpc, train_gpu, 100; batch_size, pseudocount, 
                    softness, param_inertia = 0.01, param_inertia_end = 0.95)
                    
        CUDA.@time mini_batch_em(bpc, train_gpu, 100; batch_size, pseudocount, 
                    softness, param_inertia = 0.95, param_inertia_end = 0.999)
        
        CUDA.@time full_batch_em(bpc, train_gpu, 10; batch_size, pseudocount, softness)

        print("Update parameters... ")
        @time ProbabilisticCircuits.update_parameters(bpc)
        return pc
    end
    # ll, marginal = loglikelihood_k_ones(pc,n,k)
    # ll, marginal = log_k_likelihood_wrt_split(pc, var_group_map,ks)
    # println("before training marginal wrt to all ks")
    # println(marginal)
    # open(log_path, "a+") do io
    #     write(io, "before training marginal wrt to all ks: $marginal\n")
    # end;
    pc= training()

    write("log/$(training_ID)_model_final.jpc",pc)
    # ll, marginal = loglikelihood_k_ones(pc,n,k)
    ll, marginal = log_k_likelihood_wrt_split(pc, var_group_map,ks)
    npzwrite("log/$(training_ID)_model_k_splitted_loglikelihood.npz",ll)
    println("after training wrt to all ks")
    println(marginal)
    open(log_path, "a+") do io
        write(io, "after training marginal wrt to all ks: $marginal\n")
    end;
    # println("typeof(test_gpu): $(typeof(test_gpu))")

    test_ll = loglikelihoods(bpc,test_gpu;batch_size=batch_size)
    open(log_path, "a+") do io
        write(io, "after training avg test likelihoood without recalibration: $(mean(test_ll))\n")
    end;
    #recalibration:
    reduced_train_data = compute_k_distribution_wrt_split(train_cpu,splitted,ks;return_reduced_train_data=true)
    # println(size(reduced_train_data))
    ks_test_train_dist = ks_train_dist[reduced_train_data]
    ks_test_modeled_dist = ll[reduced_train_data]
    af_bf_diff =  mean(ks_test_train_dist.-ks_test_modeled_dist) #should be less than 0
    open(log_path, "a+") do io
        write(io, "recalibration improvement (<0 then performance improves): $af_bf_diff\n")
    end;


end


main()


