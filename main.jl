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



function run()
    train_cpu, test_cpu = mnist_cpu()
    train_gpu, test_gpu = move_to_gpu(train_cpu,test_cpu)

    # twenty_dataset_names
    dataset_id=1

    # train_cpu_t, valid_data, test_cpu_t = twenty_datasets(twenty_dataset_names[dataset_id])
    
    # train_cpu=Matrix(train_cpu_t)
    # # [1:20,1:5]
    # test_cpu = Matrix(test_cpu_t)
    # [1:20,1:5]

    train_gpu,test_gpu = move_to_gpu(train_cpu,test_cpu)


    latents = 64
    pseudocount = 0.01


    
    println("Generating HCLT structure with $latents latents... ");
    @time pc = hclt(train_cpu, latents; pseudocount, input_type = Literal);
    init_parameters(pc; perturbation = 0.4);

    println("Number of nodes: $(num_nodes(pc)) before conversion ")
    
    added=convert_product_to_binary(pc)
    println("total Number of added nodes: $(added) after conversion ")
    println("Number of nodes: $(num_nodes(pc)) after conversion ")

    



    print("Moving circuit to GPU... ")
    CUDA.@time bpc = CuBitsProbCircuit(pc) #move circuit to gpu

    batch_size  = 1024
    pseudocount = .005
    softness    = 0
    n = num_nodes(pc)
    k = maximum(sum(Int.(train_cpu),dims=2))
    println("n: $(n); k: $(k)")
    ll=loglikelihood_k_ones(pc,n,k)

    println("initial loglikelihood is $(ll)")
    print("First round of minibatch EM... ")
    CUDA.@time mini_batch_em(bpc, train_gpu, 100; batch_size, pseudocount, 
    			 softness, param_inertia = 0.01, param_inertia_end = 0.95)
    			 
    CUDA.@time mini_batch_em(bpc, train_gpu, 100; batch_size, pseudocount, 
    			 softness, param_inertia = 0.95, param_inertia_end = 0.999)
    
    CUDA.@time full_batch_em(bpc, train_gpu, 10; batch_size, pseudocount, softness)

    print("Update parameters... ")
    @time ProbabilisticCircuits.update_parameters(bpc)
    
    ll=loglikelihood_k_ones(pc,n,k)

    println("final loglikelihood is $(ll)")
    pc


end


run()
