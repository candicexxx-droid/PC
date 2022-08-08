using Pkg; Pkg.activate(@__DIR__)
using ProbabilisticCircuits
using CUDA
include("utils.jl")
using Statistics
import Dates
using NPZ

function test(model_path::AbstractString,train_data_k_dist_path::AbstractString, modeled_k_dist_path::AbstractString,dataset_id::Int, use_train=false)
    pc = read(model_path, ProbCircuit)
    println("loaded!")
    # CUDA.@time bpc = CuBitsProbCircuit(pc) 
    
    # idx_ks[2,:] = float(idx_ks[2,:])

    ll = npzread(modeled_k_dist_path)
    println("size ll $(size(ll))") #size
    
    train_cpu_t, valid_data, test_cpu_t = twenty_datasets(twenty_dataset_names[dataset_id])   
    train_cpu=Matrix(train_cpu_t)

    idx_ks = float(compute_k_distribution(train_cpu)) #k+1, 2

    test_cpu = Matrix(test_cpu_t)
    if use_train
        test_cpu=train_cpu
    end
    # idx_ks =float(npzread(train_data_k_dist_path))
    # idx_ks = float(compute_k_distribution(train_cpu))

    train_gpu,test_gpu = move_to_gpu(train_cpu,test_cpu)
    idx_ks[:,2]=log.((idx_ks[:,2]./size(train_cpu)[1]))
    test_cpu_t_k = sum(test_cpu,dims=2)
    
    println("size idx_ks $(size(idx_ks))")
    k_test_train_dist = reshape(idx_ks[:,2],(size(idx_ks[:,2])[1],1))[test_cpu_t_k] #size: num_data,1
    k_test_modeled_dist = reshape(ll,(size(ll)[1],1))[test_cpu_t_k]
    
    # println(" idx_k_test_train_dist $((idx_k_test_train_dist))") 
    batch_s = 1024
    println("compute log likelihoood")
    test_ll = loglikelihoods(pc,test_cpu;batch_size=batch_s)
    println("size test_ll $(size(test_ll))") #size
    modified_test_ll = test_ll .+ k_test_train_dist .-k_test_modeled_dist




    return test_ll, modified_test_ll




end
train_data_k_dist_path = "/space/candicecai/PC_julia/PC/log/08-Aug-22-12-19-30_binarized_mnist_21_model_k_loglikelihood.npz"
modeled_k_dist_path = "/space/candicecai/PC_julia/PC/log/08-Aug-22-12-19-30_binarized_mnist_21_model_k_loglikelihood.npz"
path="/space/candicecai/PC_julia/PC/log/08-Aug-22-15-37-40_cr52_8_model_final.jpc"
data_ID=21
test_ll, modified_test_ll = test(path, train_data_k_dist_path,modeled_k_dist_path,data_ID,true)

timenow = Dates.now()
time = Dates.format(timenow, "dd-u-yy-HH-MM-SS")
test_log_ID = "test_$(twenty_dataset_names[data_ID])-$(time)"

npzwrite("log/$(test_log_ID)_test_ll_result.npz",test_ll)
println("test_ll $(mean(test_ll))")
println("modified_test_ll $(mean(modified_test_ll))")
