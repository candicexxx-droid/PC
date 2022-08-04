using Pkg; Pkg.activate(@__DIR__)
using LogicCircuits
using ProbabilisticCircuits
using ProbabilisticCircuits: CuBitsProbCircuit, loglikelihood, full_batch_em, mini_batch_em
using MLDatasets
using CUDA
using Combinatorics
using SplitApplyCombine
using TikzPictures;
include("utils.jl")
# using Base:run



# function run()
    train_cpu, test_cpu = mnist_cpu()
    # train_gpu, test_gpu = mnist_gpu()
    println("hi")
    # twenty_dataset_names
    dataset_id=1

    train_cpu_t, valid_data, test_cpu_t = twenty_datasets(twenty_dataset_names[dataset_id])
    
    train_cpu=Matrix(train_cpu_t)
    test_cpu = Matrix(test_cpu_t)
    train_cpu = train_cpu[1:20, 1:20]
    latents = 5
    pseudocount = 0.01
    
    println("Generating HCLT structure with $latents latents... ");
    @time pc = hclt(train_cpu, latents; pseudocount, input_type = Literal);
    init_parameters(pc; perturbation = 0.4);

    
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


    # print("Moving circuit to GPU... ")
    # CUDA.@time bpc = CuBitsProbCircuit(pc) #move circuit to gpu

    batch_size  = 512
    pseudocount = .005
    softness    = 0
    n = num_nodes(pc)
    k = maximum(sum(Int.(train_cpu),dims=2))

    ll=loglikelihood_k_ones(pc,n,k)

    
    # print("First round of minibatch EM... ")
    # CUDA.@time mini_batch_em(bpc, train_gpu, 100; batch_size, pseudocount, 
    # 			 softness, param_inertia = 0.01, param_inertia_end = 0.95)
    			 
    # CUDA.@time mini_batch_em(bpc, train_gpu, 100; batch_size, pseudocount, 
    # 			 softness, param_inertia = 0.95, param_inertia_end = 0.999)
    
    # CUDA.@time full_batch_em(bpc, train_gpu, 10; batch_size, pseudocount, softness)

    # print("Update parameters... ")
    # @time ProbabilisticCircuits.update_parameters(bpc)
    pc


# end


# run()
