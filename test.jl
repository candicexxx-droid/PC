using Pkg; Pkg.activate(@__DIR__)
using ProbabilisticCircuits
using CUDA
include("utils.jl")
using Statistics
import Dates
using NPZ

function test(model_path::AbstractString,train_data_k_dist_path::AbstractString, modeled_k_dist_path::AbstractString,dataset_id::Int)
    pc = read(model_path, ProbCircuit)
    println("loaded!")
    CUDA.@time bpc = CuBitsProbCircuit(pc) 
    idx_ks = float(npzread(train_data_k_dist_path))
    ll = npzread(modeled_k_dist_path)
    dataset_id = 21
    train_cpu_t, valid_data, test_cpu_t = twenty_datasets(twenty_dataset_names[dataset_id])   
    train_cpu=Matrix(train_cpu_t)
    test_cpu = Matrix(test_cpu_t)
    train_gpu,test_gpu = move_to_gpu(train_cpu,test_cpu)
    idx_ks[2,:]=log.((idx_ks[2,:]./size(train_cpu)[1]))


    test_ll = loglikelihoods(bpc,test_gpu)
    return test_ll




end
train_data_k_dist_path = "/space/candicecai/PC_julia/PC/log/binarized_mnist_21_k_distribution.npz"
modeled_k_dist_path = "/space/candicecai/PC_julia/PC/log/04-Aug-22-23-35-30_binarized_mnist_21_model_k_loglikelihood.npz"
path="/space/candicecai/PC_julia/PC/log/08-Aug-22-12-19-30_binarized_mnist_21_model.jpc"
data_ID=21
test_ll = test(path, train_data_k_dist_path,modeled_k_dist_path,data_ID)

timenow = Dates.now()
time = Dates.format(timenow, "dd-u-yy-HH-MM-SS")
test_log_ID = "test_$(twenty_dataset_names[data_ID])-$(time)"

npzwrite("log/$(test_log_ID)_test_ll_result.npz",test_ll)
println(mean(test_ll))
