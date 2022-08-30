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


dataset_id=1

train_cpu_t, valid_data, test_cpu_t = twenty_datasets(twenty_dataset_names[dataset_id])   
train_cpu=Matrix(train_cpu_t)
train_cpu = train_cpu[1:100,1:2]
println(size(train_cpu))
# 
# # 
test_cpu = Matrix(test_cpu_t)

latents = 2
pseudocount = 0.01
batch_s = 1

@time pc = hclt(train_cpu, latents; pseudocount, input_type = Literal);
model_path = "/space/candicecai/PC/log/15-Aug-22-14-39-16_accidents_1_model_final_2vars_sanitycheck.jpc"#
pc = read(model_path, ProbCircuit)
println("loaded!")
CUDA.@time bpc = CuBitsProbCircuit(pc) 


sanity_check_data = [0 0; 1 0; 0 1;1 1]
sanity_check_data,train_gpu = move_to_gpu(sanity_check_data,train_cpu)

ll = loglikelihoods(bpc,sanity_check_data;batch_size=batch_s)
t=time()
ll_k,_ = loglikelihood_k_ones(pc,2,2)
el = time()-t
println("t is $el")
TikzPictures.standaloneWorkaround(true)  # workaround
z = plot(pc);
save(PDF("plot"), z);


marginal = logsumexp(ll)
marginal_wrt_k = logsumexp(ll_k)
println("brute force marginal is $marginal")
println("marginal_wrt_k is $marginal_wrt_k")
#should be
#brute force marginal is 0.0
# marginal_wrt_k is 5.9604645e-8

