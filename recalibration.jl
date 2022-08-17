using Pkg; Pkg.activate(@__DIR__)
using ProbabilisticCircuits
using CUDA
include("utils.jl")
using Statistics
import Dates
using NPZ

function test(model_path::AbstractString, modeled_k_dist_path::AbstractString,dataset_id::Int, use_train=true, original_ll=false, return_test_cpu_t_k=true,KL_verified=false)
    pc = read(model_path, ProbCircuit)
    println("loaded!")
    CUDA.@time bpc = CuBitsProbCircuit(pc) 
    
    # idx_ks[2,:] = float(idx_ks[2,:])

    ll = npzread(modeled_k_dist_path)
    println("size ll $(size(ll))") #size
    
    train_cpu_t, valid_data, test_cpu_t = twenty_datasets(twenty_dataset_names[dataset_id])   
    train_cpu=Matrix(train_cpu_t)
    
    
    # train_cpu=unique(train_cpu;dims=1)
    # println("size train_cpu $(size(train_cpu))") #size

    
    test_cpu = Matrix(test_cpu_t)
    
    # idx_ks =float(npzread(train_data_k_dist_path))
    # idx_ks = float(compute_k_distribution(train_cpu))
    batch_s = 1024
    train_gpu,test_gpu = move_to_gpu(train_cpu,test_cpu)
    if use_train
        println("using training data")
        test_cpu=train_cpu
        test_gpu=train_gpu
    end
    println("compute log likelihoood")
    # test_ll = loglikelihoods(pc,test_cpu;batch_size=batch_s)
    test_ll = loglikelihoods(bpc,test_gpu;batch_size=batch_s)
    if original_ll
        println("only return oirginal ll")
        println(test_ll[1:100])
        return test_ll, [0,0]
    end

    idx_ks = float(compute_k_distribution(train_cpu)) #size = k+1, 2, k distribution from training data 
    smooth=1
    train_size = sum(idx_ks[2,:])
    # idx_ks[2,:]=log.(((idx_ks[2,:].+smooth)./(size(train_cpu)[1]+smooth))) #with smoothing
    # println(idx_ks[2,:][:,idx_ks[2,:]==0])
  

    q_k = reshape(idx_ks[2,:],(size(idx_ks[2,:])[1],))
    # println(q_k.==0)
    q_k[q_k.==0] .= smooth
    added_total = sum(q_k)-train_size
    # print(q_k.>=0)
    # for i in 1:added_total
    #     # println(typeof(i))
    #     q_k[q_k.>0][Int(i)] = q_k[q_k.>0][Int(i)]-1
    # end
    smoothed_train_size = sum(q_k)
    println("$(added_total) train size after smooth = $(train_size)")

    # smoothed_train_size = sum((idx_ks[2,:]))
    log_q_k=log.(((q_k)./smoothed_train_size)) #q(k)
    test_cpu_t_k = sum(test_cpu,dims=2)
    println("test_cpu_t_k[1:10]")
    println(test_cpu_t_k[1:10])
    println("log_q_k[1:10]")
    println(log_q_k[1:10])
    # println("size idx_ks $(size(idx_ks))")
    k_test_train_dist = log_q_k[test_cpu_t_k.+1]#q(k(x))size: num_data,1,CuArray{Float32, 1, CUDA.Mem.DeviceBuffer}
    
    k_test_modeled_dist = reshape(ll,(size(ll)[1],1))[test_cpu_t_k.+1] #p(k(x)) 
    println("k_test_train_dist size ")
    println(size(k_test_train_dist))
    # println(" idx_k_test_train_dist $((idx_k_test_train_dist))") 
    
    

    test_ll = reshape(test_ll, (size(test_ll)[1],1))
    println("size test_ll $(size(test_ll))") #size
    k_test_train_dist,k_test_modeled_dist=move_to_gpu(k_test_train_dist,k_test_modeled_dist)

    # test_ll, k_test_train_dist, k_test_modeled_dist=Float64(test_ll),Float64(k_test_train_dist), Float64(k_test_modeled_dist)
    modified_test_ll = test_ll.-k_test_modeled_dist .+ k_test_train_dist
    # 
    if KL_verified
        println("computing KL divergence!")
        println(size(log_q_k))
        KL_d = sum(exp.(log_q_k).*(log_q_k.-ll))
        println(size(KL_d))
        println("size check::::")
        println(size(k_test_train_dist))
        println(size(k_test_modeled_dist))
        af_bf_diff =  mean(k_test_train_dist.-k_test_modeled_dist)

        # return k_test_modeled_dist, k_test_train_dist, idx_ks[2,:],ll #p(k(x)), q(k(x)), q(k), p(k)
        return KL_d,af_bf_diff
    end

    
    
    if return_test_cpu_t_k
        return test_ll, modified_test_ll,test_cpu_t_k
    end

    

    return test_ll, modified_test_ll




end
# train_data_k_dist_path = "/space/candicecai/PC/log/08-Aug-22-12-19-30_binarized_mnist_21_model_k_loglikelihood.npz"
modeled_k_dist_path = "/space/candicecai/PC/log/08-Aug-22-18-27-04_binarized_mnist_21_model_k_loglikelihood.npz"
# path="/space/candicecai/PC/log/08-Aug-22-15-37-40_cr52_8_model_final.jpc"
# path="/space/candicecai/PC/log/08-Aug-22-18-27-04_binarized_mnist_21_model_final.jpc"
path = "/space/candicecai/PC/log/08-Aug-22-18-27-04_binarized_mnist_21_model_final.jpc"
data_ID=21


timenow = Dates.now()
time = Dates.format(timenow, "dd-u-yy-HH-MM-SS")
test_log_ID = "test_$(twenty_dataset_names[data_ID])-$(time)"
log_path = "log/"*(test_log_ID)*"_test_log.txt"
compute_calibrated_ll=false
if compute_calibrated_ll
    test_ll, modified_test_ll,test_cpu_t_k = test(path, modeled_k_dist_path,data_ID,true,false)
    temp=hcat(test_cpu_t_k,modified_test_ll)

    marginal_modified_test_ll=Dict()
    for i in test_cpu_t_k
        if !(i in keys(marginal_modified_test_ll))
            marginal_modified_test_ll[i] = exp(temp[i,2])
        else

            marginal_modified_test_ll[i] +=exp(temp[i,2])
        end

    end
    # println("sum prob of modified_test_ll")
    # println(logsumexp(test_ll))

    marginal_modified_test_ll=hcat([convert(Float64, i) for i in keys(marginal_modified_test_ll)],[i for i in values(marginal_modified_test_ll)])
    # println(marginal_modified_test_ll)
    npzwrite("log/$(test_log_ID)_marginal_modified_test_ll_result.npz",marginal_modified_test_ll)
    println("test_ll $(mean(test_ll))")
    println("modified_test_ll $(mean(modified_test_ll))")
    
    open(log_path, "a+") do io
        write(io, "($test_log_ID)\n")   
        write(io,"test_ll $(mean(test_ll))\n")
        write(io,"modified_test_ll $(mean(modified_test_ll))\n")
    end;
else
    KL,af_bf_diff = test(path, modeled_k_dist_path,data_ID,true,false,true,true) #all in log space 
    println("KL divergence is  $(KL)")
    println("After before diff is $(af_bf_diff)")
    open(log_path, "a+") do io
        write(io, "$test_log_ID\n")   
        write(io,"KL divergence is  $(KL)\n")
        write(io,"After before diff is $(af_bf_diff)\n")
    end;


end




