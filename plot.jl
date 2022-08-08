using Plots
using NPZ
using LogicCircuits
using ProbabilisticCircuits
using Pkg; Pkg.activate(@__DIR__)


dataset_id = 21
idx_ks_store = twenty_dataset_names[dataset_id] * "_$(dataset_id)"
training_ID = "04-Aug-22-23-35-30_binarized_mnist_21"
train_cpu_t=twenty_datasets(twenty_dataset_names[dataset_id])[1]
train_cpu=Matrix(train_cpu_t)
idx_ks = npzread("log/"*idx_ks_store*"_k_distribution.npz")

ll = npzread("log/"*training_ID*"_model_k_loglikelihood.npz")

w=900
h=500

k = maximum(sum(Int.(train_cpu),dims=2))
Plots.plot(idx_ks[1,:], idx_ks[2,:]./size(train_cpu)[1],title="$(twenty_dataset_names[dataset_id]) Distribution: Training Data v.s. Modeled Likelihood", label = "train_data_distribution",size = (w, h),left_margin = 3Plots.cm,top_margin =1.5Plots.cm,right_margin=3Plots.cm, bottom_margin = 1.5Plots.cm)
Plots.plot!(0:k,exp.(ll),xlabel = "number of ones in a sample",label = "HCLT modeled_distribution",size = (w, h))
Plots.savefig("log/model_dist_v.s._train_$(training_ID)_test.png")
