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

# using Base:run



function run()
    # train_cpu, test_cpu = mnist_cpu()
    # train_gpu, test_gpu = move_to_gpu(train_cpu,test_cpu)

    # twenty_dataset_names
    dataset_id=21

    train_cpu_t, valid_data, test_cpu_t = twenty_datasets(twenty_dataset_names[dataset_id])   
    train_cpu=Matrix(train_cpu_t)
    # # [1:20,1:5]
    test_cpu = Matrix(test_cpu_t)
    [1:20,1:5]

    train_gpu,test_gpu = move_to_gpu(train_cpu,test_cpu)


    latents = 64
    pseudocount = 0.01

    timenow = Dates.now()
    time = Dates.format(timenow, "dd-u-yy-HH-MM-SS")
    training_ID = time * "_" * twenty_dataset_names[dataset_id] * "_$(dataset_id)"
    log_path = "log/$(training_ID)_hclt.txt"

    open(log_path, "a+") do io
        write(io, "Generating HCLT structure with $latents latents...\n ")
    end;
    println("Generating HCLT structure with $latents latents... ");
    @time pc = hclt(train_cpu, latents; pseudocount, input_type = Literal);
    init_parameters(pc; perturbation = 0.4);
    open(log_path, "a+") do io
        write(io, "Number of nodes: $(num_nodes(pc)) before conversion\n")
    end;
    println("Number of nodes: $(num_nodes(pc)) before conversion ")
    
    added=convert_product_to_binary(pc)
    println("total Number of added nodes: $(added) after conversion ")
    println("Number of nodes: $(num_nodes(pc)) after conversion ")
    idx_ks = compute_k_distribution(train_cpu)
    idx_ks_store = twenty_dataset_names[dataset_id] * "_$(dataset_id)"
    npzwrite("log/$(idx_ks_store)_k_distribution.npz",idx_ks)


    # train_data_dist = 
    # Plots.plot(idx_ks[1,:], idx_ks[2,:]./size(train_cpu)[1],label = "train_data_distribution")
    w=900
    h=500
    Plots.plot(idx_ks[1,:], idx_ks[2,:]./size(train_cpu)[1],title="$(twenty_dataset_names[dataset_id]) Distribution: Training Data v.s. Modeled Likelihood", label = "train_data_distribution",size = (w, h),left_margin = 3Plots.cm,top_margin =1.5Plots.cm,right_margin=3Plots.cm, bottom_margin = 1.5Plots.cm)
    # savefig(train_data_dist,"train_data_dist_$(dataset_id)_$(twenty_dataset_names[dataset_id]).png")

    open(log_path, "a+") do io
        write(io, "total Number of added nodes: $(added) after conversion\n")
        write(io, "Number of nodes: $(num_nodes(pc)) before conversion\n")
        write(io, "ks that have more than one instances $(idx_ks)\n")
    end;
    


    write("log/$(training_ID)_model.jpc",pc)


    
    println("Finish saving weights")

    n = size(train_gpu)[2]
    k = maximum(sum(Int.(train_cpu),dims=2))
    println("n: $(n); k: $(k)\n n/k: $(n/k)")

    open(log_path, "a+") do io
        write(io, "n: $(n); k: $(k)\n")
        # write(io, "initial loglikelihood is $(ll)\n")
        
    end;


    #training
    function training()
        print("Moving circuit to GPU... ")
        CUDA.@time bpc = CuBitsProbCircuit(pc) #move circuit to gpu

        batch_size  = 1024
        pseudocount = .005
        softness    = 0
        

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


    pc = training()

    #Results Saving
    ll=loglikelihood_k_ones(pc,n,k,idx_ks)
    open(log_path, "a+") do io
        write(io, "final loglikelihood is $(ll)\n")   
    end;
    write("log/$(training_ID)_model_test.jpc",pc)
    println("final loglikelihood is $(ll)")
    npzwrite("log/$(training_ID)_model_k_loglikelihood.npz",ll)
    Plots.plot!(0:k,exp.(ll),label = "modeled_distribution",xlabel="k")

    
    Plots.plot!(0:k,exp.(ll),xlabel = "number of ones in a sample",label = "HCLT modeled_distribution",size = (w, h))
    Plots.savefig("log/model_dist_v.s._train_$(training_ID)_test.png")





    


end


run()
