hf_model_path="/data_fast/jiaruiyu/workstation/user_data_analysis/LLM_post_training/output/250530_250630_completion_v1_sp_training_0731_deduplication_0.75_5_50k"

declare -a configs=(
    # "/data_large/liangxiaoyun/model_output/grpo-v8.6.19_250811/global_step_70/actor|grpo-v8.6.19_250811_e5|5|v1"
    # "/data_large/liangxiaoyun/model_output/grpo-v8.6.19_250811/global_step_56/actor|grpo-v8.6.19_250811_e4|4|v1"
    # "/data_large/liangxiaoyun/model_output/grpo-v8.6.19_250811/global_step_84/actor|grpo-v8.6.19_250811_e6|6|v1"
    # "/data_large/liangxiaoyun/model_output/grpo-v8.6.20_250811/global_step_56/actor|grpo-v8.6.20_250811_e2|2|v1"
    # "/data_large/liangxiaoyun/model_output/grpo-v8.6.20_250811/global_step_84/actor|grpo-v8.6.20_250811_e3|3|v1"
    # "/data_large/liangxiaoyun/model_output/grpo-v8.6.20_250811/global_step_112/actor|grpo-v8.6.20_250811_e4|4|v1"
    # "/data_large/liangxiaoyun/model_output/grpo-v8.7.1_250910/global_step_77/actor|grpo-v8.7.1_250910_e1|1|v1"
    # "/data_large/liangxiaoyun/model_output/grpo-v8.7.1_250910/global_step_154/actor|grpo-v8.7.1_250910_e2|2|v1"
    # "/data_large/liangxiaoyun/model_output/grpo-v8.7.1_250910/global_step_231/actor|grpo-v8.7.1_250910_e3|3|v1"
    "/data_large/liangxiaoyun/model_output/grpo-v8.6.19_250811_multi_node/global_step_70/actor|grpo-v8.6.19_250811_multi_e5|5|v1"
)

# Python路径
PYTHON_PATH="/data_train/liangxiaoyun/miniconda3/envs/verl/bin/python3"
TEST_PYTHON_PATH="/data_train/liangxiaoyun/miniconda3/envs/main/bin/python3"

# 遍历配置并执行
for config in "${configs[@]}"; do
    IFS='|' read -r model_path model_name epoch fim_format <<< "$config"
    save_model_path=$model_path"_merge"

    echo "Processing: $model_name"
    echo "Model path: $model_path"
    echo "save_model_path: $save_model_path"
    echo "Epoch: $epoch"
    echo "Fim format: $fim_format"
    echo "----------------------------------------"
    
    cd "/data_train/liangxiaoyun/projects/verl/scripts/"
    # merge
    $PYTHON_PATH model_merger.py merge --backend "fsdp" --local_dir "$model_path" --hf_model_path "$hf_model_path" --target_dir "$save_model_path"
    echo "finished model merge\n"
    
    cd "/data_train/liangxiaoyun/projects/safety/"
    # 执行三个评估脚本
    $TEST_PYTHON_PATH evaluate.py --model_path "$save_model_path" --model_name "$model_name" --fim_format "$fim_format"
    $TEST_PYTHON_PATH online_data_test.py --model_path "$save_model_path" --epoch "$epoch" --fim_format "$fim_format"
    $TEST_PYTHON_PATH badcase_test_set.py --model_path "$save_model_path" --epoch "$epoch" --fim_format "$fim_format"
    echo "========================================\n"
done


# model_path="/data_large/liangxiaoyun/model_output/grpo-v8.6.16_250811/global_step_78/actor"
# /data_train/liangxiaoyun/miniconda3/envs/verl/bin/python3 model_merger.py merge --backend "fsdp" --local_dir $model_path --hf_model_path $hf_model_path --target_dir $model_path"_merge"
# model_path="/data_large/liangxiaoyun/model_output/grpo-v8.6.16_250811/global_step_65/actor"
# /data_train/liangxiaoyun/miniconda3/envs/verl/bin/python3 model_merger.py merge --backend "fsdp" --local_dir $model_path --hf_model_path $hf_model_path --target_dir $model_path"_merge"
# model_path="/data_large/liangxiaoyun/model_output/grpo-v8.6.18_2_250811/global_step_70/actor"
# /data_train/liangxiaoyun/miniconda3/envs/verl/bin/python3 model_merger.py merge --backend "fsdp" --local_dir $model_path --hf_model_path $hf_model_path --target_dir $model_path"_merge"
# model_path="/data_large/liangxiaoyun/model_output/grpo-v8.6.18_2_250811/global_step_84/actor"
# /data_train/liangxiaoyun/miniconda3/envs/verl/bin/python3 model_merger.py merge --backend "fsdp" --local_dir $model_path --hf_model_path $hf_model_path --target_dir $model_path"_merge"

# model_path="/data_large/liangxiaoyun/model_output/grpo-v8.6.16_250811/global_step_91/actor"
# /data_train/liangxiaoyun/miniconda3/envs/verl/bin/python3 model_merger.py merge --backend "fsdp" --local_dir $model_path --hf_model_path $hf_model_path --target_dir $model_path"_merge"
# model_path="/data_large/liangxiaoyun/model_output/grpo-v8.6.17_250811/global_step_24/actor"
# /data_train/liangxiaoyun/miniconda3/envs/verl/bin/python3 model_merger.py merge --backend "fsdp" --local_dir $model_path --hf_model_path $hf_model_path --target_dir $model_path"_merge"
# model_path="/data_large/liangxiaoyun/model_output/grpo-v8.6.17_250811/global_step_48/actor"
# /data_train/liangxiaoyun/miniconda3/envs/verl/bin/python3 model_merger.py merge --backend "fsdp" --local_dir $model_path --hf_model_path $hf_model_path --target_dir $model_path"_merge"
# model_path="/data_large/liangxiaoyun/model_output/grpo-v8.6.17_250811/global_step_72/actor"
# /data_train/liangxiaoyun/miniconda3/envs/verl/bin/python3 model_merger.py merge --backend "fsdp" --local_dir $model_path --hf_model_path $hf_model_path --target_dir $model_path"_merge"

# model_path="/data_large/liangxiaoyun/model_output/grpo-v8.6.17_250811/global_step_96/actor"
# /data_train/liangxiaoyun/miniconda3/envs/verl/bin/python3 model_merger.py merge --backend "fsdp" --local_dir $model_path --hf_model_path $hf_model_path --target_dir $model_path"_merge"
# model_path="/data_large/liangxiaoyun/model_output/grpo-v8.6.17_250811/global_step_168/actor"
# /data_train/liangxiaoyun/miniconda3/envs/verl/bin/python3 model_merger.py merge --backend "fsdp" --local_dir $model_path --hf_model_path $hf_model_path --target_dir $model_path"_merge"
# model_path="/data_large/liangxiaoyun/model_output/grpo-v8.6.18_2_250811/global_step_98/actor"
# /data_train/liangxiaoyun/miniconda3/envs/verl/bin/python3 model_merger.py merge --backend "fsdp" --local_dir $model_path --hf_model_path $hf_model_path --target_dir $model_path"_merge"

# model_path="/data_large/liangxiaoyun/model_output/grpo-v8.6.14_250811/global_step_44/actor"
# /data_train/liangxiaoyun/miniconda3/envs/verl/bin/python3 model_merger.py merge --backend "fsdp" --local_dir $model_path --hf_model_path $hf_model_path --target_dir $model_path"_merge"
# model_path="/data_large/liangxiaoyun/model_output/grpo-v8.6.14_250811/global_step_13/actor"
# /data_train/liangxiaoyun/miniconda3/envs/verl/bin/python3 model_merger.py merge --backend "fsdp" --local_dir $model_path --hf_model_path $hf_model_path --target_dir $model_path"_merge"
# model_path="/data_large/liangxiaoyun/model_output/grpo-v8.6.14_250811/global_step_26/actor"
# /data_train/liangxiaoyun/miniconda3/envs/verl/bin/python3 model_merger.py merge --backend "fsdp" --local_dir $model_path --hf_model_path $hf_model_path --target_dir $model_path"_merge"
# model_path="/data_large/liangxiaoyun/model_output/grpo-v8.6.14_250811/global_step_39/actor"
# /data_train/liangxiaoyun/miniconda3/envs/verl/bin/python3 model_merger.py merge --backend "fsdp" --local_dir $model_path --hf_model_path $hf_model_path --target_dir $model_path"_merge"
# model_path="/data_large/liangxiaoyun/model_output/grpo-v8.6.14_250811/global_step_52/actor"
# /data_train/liangxiaoyun/miniconda3/envs/verl/bin/python3 model_merger.py merge --backend "fsdp" --local_dir $model_path --hf_model_path $hf_model_path --target_dir $model_path"_merge"
# model_path="/data_large/liangxiaoyun/model_output/grpo-v8.6.14_250811/global_step_65/actor"
# /data_train/liangxiaoyun/miniconda3/envs/verl/bin/python3 model_merger.py merge --backend "fsdp" --local_dir $model_path --hf_model_path $hf_model_path --target_dir $model_path"_merge"
# model_path="/data_large/liangxiaoyun/model_output/grpo-v8.6.14_250811/global_step_78/actor"
# /data_train/liangxiaoyun/miniconda3/envs/verl/bin/python3 model_merger.py merge --backend "fsdp" --local_dir $model_path --hf_model_path $hf_model_path --target_dir $model_path"_merge"
# model_path="/data_large/liangxiaoyun/model_output/grpo-v8.6.14_250811/global_step_91/actor"
# /data_train/liangxiaoyun/miniconda3/envs/verl/bin/python3 model_merger.py merge --backend "fsdp" --local_dir $model_path --hf_model_path $hf_model_path --target_dir $model_path"_merge"
# model_path="/data_large/liangxiaoyun/model_output/grpo-v8.6.14_250811/global_step_104/actor"
# /data_train/liangxiaoyun/miniconda3/envs/verl/bin/python3 model_merger.py merge --backend "fsdp" --local_dir $model_path --hf_model_path $hf_model_path --target_dir $model_path"_merge"
# model_path="/data_large/liangxiaoyun/model_output/grpo-v8.6.14_250811/global_step_117/actor"
# /data_train/liangxiaoyun/miniconda3/envs/verl/bin/python3 model_merger.py merge --backend "fsdp" --local_dir $model_path --hf_model_path $hf_model_path --target_dir $model_path"_merge"
# model_path="/data_large/liangxiaoyun/model_output/grpo-v8.6.14_250811/global_step_130/actor"
# /data_train/liangxiaoyun/miniconda3/envs/verl/bin/python3 model_merger.py merge --backend "fsdp" --local_dir $model_path --hf_model_path $hf_model_path --target_dir $model_path"_merge"
# model_path="/data_large/liangxiaoyun/model_output/grpo-v8.6.14_250811/global_step_143/actor"
# /data_train/liangxiaoyun/miniconda3/envs/verl/bin/python3 model_merger.py merge --backend "fsdp" --local_dir $model_path --hf_model_path $hf_model_path --target_dir $model_path"_merge"
# model_path="/data_large/liangxiaoyun/model_output/grpo-v8.6.14_250811/global_step_156/actor"
# /data_train/liangxiaoyun/miniconda3/envs/verl/bin/python3 model_merger.py merge --backend "fsdp" --local_dir $model_path --hf_model_path $hf_model_path --target_dir $model_path"_merge"
# model_path="/data_large/liangxiaoyun/model_output/grpo-v8.6.14_250811/global_step_169/actor"
# /data_train/liangxiaoyun/miniconda3/envs/verl/bin/python3 model_merger.py merge --backend "fsdp" --local_dir $model_path --hf_model_path $hf_model_path --target_dir $model_path"_merge"

# model_path="/data_large/liangxiaoyun/model_output/grpo-v8.6.10_250811/global_step_90/actor"
# /data_train/liangxiaoyun/miniconda3/envs/verl/bin/python3 model_merger.py merge --backend "fsdp" --local_dir $model_path --hf_model_path $hf_model_path --target_dir $model_path"_merge"
# model_path="/data_large/liangxiaoyun/model_output/grpo-v8.6.10_250811/global_step_100/actor"
# /data_train/liangxiaoyun/miniconda3/envs/verl/bin/python3 model_merger.py merge --backend "fsdp" --local_dir $model_path --hf_model_path $hf_model_path --target_dir $model_path"_merge"
# model_path="/data_large/liangxiaoyun/model_output/grpo-v8.6.10_250811/global_step_110/actor"
# /data_train/liangxiaoyun/miniconda3/envs/verl/bin/python3 model_merger.py merge --backend "fsdp" --local_dir $model_path --hf_model_path $hf_model_path --target_dir $model_path"_merge"

