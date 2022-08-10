using Plots
using NPZ
using LogicCircuits
using ProbabilisticCircuits
using Pkg; Pkg.activate(@__DIR__)
include("utils.jl")


dataset_id = 21
idx_ks_store = twenty_dataset_names[dataset_id] * "_$(dataset_id)"
training_ID = "04-Aug-22-23-35-30_binarized_mnist_21"
train_cpu_t=twenty_datasets(twenty_dataset_names[dataset_id])[1]
train_cpu=Matrix(train_cpu_t)

idx_ks = compute_k_distribution(train_cpu)

# idx_ks = npzread("log/"*idx_ks_store*"_k_distribution.npz")

ll = npzread("log/"*training_ID*"_model_k_loglikelihood.npz")
pth="/space/candicecai/PC/log/test_binarized_mnist-08-Aug-22-21-42-43_modified_test_ll_result_worked.npz"
modified_test_ll = npzread(pth)
w=900
h=500

k = maximum(sum(Int.(train_cpu),dims=2))
Plots.plot(idx_ks[1,:], idx_ks[2,:]./size(train_cpu)[1],title="$(twenty_dataset_names[dataset_id]) Distribution: Training Data v.s. Modeled Likelihood", label = "train_data_distribution",size = (w, h),left_margin = 3Plots.cm,top_margin =1.5Plots.cm,right_margin=3Plots.cm, bottom_margin = 1.5Plots.cm)
Plots.plot!(modified_test_ll[:,1],modified_test_ll[:,2]./sum(modified_test_ll[:,2]),xlabel = "number of ones in a sample",label = "Modified HCLT modeled_distribution",size = (w, h))
println("modified_test_ll_normalized")
println(maximum(modified_test_ll[:,2]./sum(modified_test_ll[:,2])))
Plots.plot!(0:k,exp.(ll),xlabel = "number of ones in a sample",label = "HCLT modeled_distribution",size = (w, h))
Plots.savefig("log/model_dist_v.s._train_$(training_ID)_v.s.modified_distribution_normalized.png")
