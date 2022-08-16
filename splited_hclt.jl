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

dataset_id=21
train_cpu_t, valid_data, test_cpu_t = twenty_datasets(twenty_dataset_names[dataset_id])   

function main()

    not_debugging = check_debugging()

    #parameters
    param_dict=Dict()
    param_dict["latents"] = 64
    param_dict["pseudocount"] = 0.005
    param_dict["batch_size"] = 1024 #use 1024 for actual training
    param_dict["softness"] = 0
    param_dict["group_size"] = 392
    param_dict["group_num"] = 2
    param_dict["run_single_dim"] = false
    param_dict["train"] = true
    param_dict["chpt_id"] = "15-Aug-22-18-46-01_binarized_mnist_21"
    latents = param_dict["latents"] 
    pseudocount = param_dict["pseudocount"]
    batch_size  = param_dict["batch_size"]
    softness    = param_dict["softness"]
    group_size = param_dict["group_size"] #group_num = ceil(num_vars/group_size)
    expected_group_num = param_dict["group_num"]
    
    train_cpu=Matrix(train_cpu_t)
    # [1:100,1:2]
    train_cpu = train_cpu
    # [1:100,1:15]
    # 
    
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
    println("pseudocount")
    println(pseudocount)
    @time pc = hclt(train_cpu, latents; pseudocount, input_type = Literal);

    

    
    convert_product_to_binary(pc)
    init_parameters(pc; perturbation = 0.4);
    compute_scope(pc)
    group_num = Int(ceil(n/group_size))

    @assert group_num ==expected_group_num #check if group num is desired, if not adjust group_size!
    global_scope = BitSet(1:n)
    # pc.scope
    
    splitted, var_group_map=split_rand_vars(global_scope,group_size)
    ks = compute_ks(train_cpu, splitted)
    println("computing ks distribution...")
    ks_train_dist=compute_k_distribution_wrt_split(train_cpu,splitted,ks)
    


    if length(param_dict["chpt_id"])>0

        param_dict["train"] = false
        println(param_dict["chpt_id"])
        chpt="log/" * param_dict["chpt_id"]*"_model_final.jpc"
        pc = read(chpt, ProbCircuit)
        training_ID = param_dict["chpt_id"]
        compute_scope(pc)
        println("loaded!")
    end
    println("number of multiplication nodes: $(length(mulnodes(pc)))")
    println("number of sum nodes: $(length(sumnodes(pc)))")
    print("Moving circuit to GPU... ")
    CUDA.@time bpc = CuBitsProbCircuit(pc)


    #group_splitting
    
    
    if not_debugging && param_dict["train"]
        open(log_path, "a+") do io
            write(io, "params: $param_dict\n")
            write(io, "group size: $group_size; group_num: $group_num\n")
            write(io, "computing ks distribution...\n")
            write(io, "n: $(n); k: $(k)\n")
            write(io, "ks: $ks\n")
        end;

    end

    
    
     #move circuit to gpu
    function training()
        print("First round of minibatch EM... ")
        CUDA.@time mini_batch_em(bpc, train_gpu, 100; batch_size, pseudocount, 
                    softness, param_inertia = 0.01, param_inertia_end = 0.95)
                    
        CUDA.@time mini_batch_em(bpc, train_gpu, 100; batch_size, pseudocount, 
                    softness, param_inertia = 0.95, param_inertia_end = 0.999)
        
        CUDA.@time full_batch_em(bpc, train_gpu, 100; batch_size, pseudocount, softness)

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

    #Training
    if param_dict["train"]
        pc= training()
    end
    test_ll = loglikelihoods(bpc,test_gpu;batch_size=batch_size)
    if not_debugging && param_dict["train"]
        open(log_path, "a+") do io
            write(io, "after training avg test likelihoood without recalibration: $(mean(test_ll))\n")
        end;
    end

    

    write("log/$(training_ID)_model_final.jpc",pc)
    if param_dict["run_single_dim"]
        ll, marginal = loglikelihood_k_ones(pc,n,k)
    else
        println("hihi")
        ll, marginal = log_k_likelihood_wrt_split(pc, var_group_map,ks,group_num)
    end
    # ll, marginal = loglikelihood_k_ones(pc,n,k)
    
    
    println("after training wrt to all ks")
    println(marginal)
    if not_debugging
        npzwrite("log/$(training_ID)_model_k_splitted_loglikelihood.npz",ll)
        open(log_path, "a+") do io
            write(io, "after training marginal wrt to all ks in log space: $marginal\n")
        end;

    end

    
    # println("typeof(test_gpu): $(typeof(test_gpu))")

    
    #recalibration:
    reduced_train_data = compute_k_distribution_wrt_split(train_cpu,splitted,ks;return_reduced_train_data=true)
    # println(sum(train_cpu,dims=2))
    # println(reduced_train_data)
    ks_test_train_dist = ks_train_dist[reduced_train_data]
    ks_test_modeled_dist = ll[reduced_train_data]
    af_bf_diff =  mean(ks_test_train_dist.-ks_test_modeled_dist) #should be less than 0
    if not_debugging
        open(log_path, "a+") do io
            write(io, "recalibration improvement (>0 then performance improves): $af_bf_diff\n")
        end;
    end
    
    println("recalibration improvement (>0 then performance improves): $af_bf_diff")

    # plot_pc(pc)


end


main()


