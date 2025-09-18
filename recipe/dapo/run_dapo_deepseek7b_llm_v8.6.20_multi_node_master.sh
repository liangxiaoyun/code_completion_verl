#ray 启动！！！！！！！！！！！！！！！
#master:10.0.8.6
# ray start --head --dashboard-host=0.0.0.0

# #worker
# ray start --address='10.0.8.6:6379'
# ray status #可以看到目前有几个节点，重点修改以下运行配置：trainer.nnodes=2
#！！！！！！！！！！！！！！！！！！！！！！！！！！！

#在master启动以下命令
#nohup bash run_deepseek7b_llm_v8.6.19_multi_node_master.sh >> logs/multi_node_7b_19_grpo.log 2>&1 &

#启动以下命令后可流式查看 log,raysuubmit 从log中可以找到
#ray job logs raysubmit_LL41NYX684HyiEPT -f 

#master 任务停止ray: ray stop
set -x
# 临时清除当前会话的代理环境变量
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset ftp_proxy
unset FTP_PROXY
unset no_proxy
unset NO_PROXY

source /data_train/liangxiaoyun/miniconda3/etc/profile.d/conda.sh
export CUDA_HOME=/data_train/liangxiaoyun/miniconda3
export PATH=$CUDA_HOME/bin:$PATH
export WANDB_API_KEY="2205fe10e62e95b3f624aa1aaa18e4accd300d9e"

project_name='code-completion-grpo-project'
exp_name='dapo-v8.6.20_250918'

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=8000
max_response_length=$((256 * 3))
enable_overlong_buffer=True
overlong_buffer_len=256
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=True
filter_groups_metric=score
max_num_gen_batches=10
train_prompt_bsz=512
gen_prompt_bsz=$((train_prompt_bsz * 3))
n_resp_per_prompt=16
train_prompt_mini_bsz=32


NNODES=2
MODEL_PATH="/data_fast/jiaruiyu/workstation/user_data_analysis/LLM_post_training/output/250530_250630_completion_v1_sp_training_0731_deduplication_0.75_5_50k_user_50k_github_from_v4.0"
CKPTS_DIR="/data_large_v2/liangxiaoyun/model_output/${exp_name}"
TRAIN_FILE="/data_train/liangxiaoyun/datas/completion_sft_datas/train_data_merge_user_distillation/250530_250630_completion_v1_sp_training_0731_deduplication_0.75_5_deduplicated_001_less_8000_filter0819_cursor_type_output_length_language_balance_sample_10000_none_10000_serious_badcase_2434-dul4.parquet"
TEST_FILE="/data_train/liangxiaoyun/datas/completion_sft_datas/train_data_merge_user_distillation/test_python_output.parquet"

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# Performance Related Parameter
sp_size=8
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
offload=True
gen_tp=4
ray job submit --address="http://10.0.8.6:8265" \
    --runtime-env=/data_train/liangxiaoyun/projects/verl/recipe/dapo/runtime_env.yaml \
    --no-wait \
    -- \
/data_train/liangxiaoyun/miniconda3/envs/verl/bin/python3 -m recipe.dapo.main_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=True \
    trainer.test_freq=28 \
    trainer.save_freq=28 \
    trainer.total_epochs=5 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    custom_reward_function.path=/data_train/liangxiaoyun/projects/verl/verl/utils/reward_score/code_completion.py \
    trainer.resume_mode=auto $@
