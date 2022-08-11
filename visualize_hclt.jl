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
train_cpu = train_cpu[1:100,1:3]
println(size(train_cpu))
# 
# # 
test_cpu = Matrix(test_cpu_t)

latents = 2
pseudocount = 0.01

@time pc = hclt(train_cpu, latents; pseudocount, input_type = Literal);

TikzPictures.standaloneWorkaround(true)  # workaround
z = plot(pc);
save(PDF("plot"), z);


