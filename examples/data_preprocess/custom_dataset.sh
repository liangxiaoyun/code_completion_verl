# data_path=/data_train/liangxiaoyun/datas/completion_sft_datas/train_data_merge_user_distillation/250530_250630_completion_v1_sp_training_0731_deduplication_0.75_5_deduplicated_001_less_8000.jsonl
# /data_train/liangxiaoyun/miniconda3/envs/verl/bin/python3 custom_dataset.py --data_path $data_path --split train --fim_format v1 --sample_num 10000

# data_path=/data_large/liangxiaoyun/data/online_complete_data/250701-250730_badcase/250701-250730_badcase_distillation_7b_train_data.jsonl
# /data_train/liangxiaoyun/miniconda3/envs/verl/bin/python3 custom_dataset.py --data_path $data_path --split train --fim_format v1 --balance_sample_key data_source --sample_num 2000

data_path=/data_large/liangxiaoyun/data/online_complete_data/250701-250730_badcase/250701-250730_badcase_serious_train_data.jsonl
/data_train/liangxiaoyun/miniconda3/envs/verl/bin/python3 custom_dataset.py --data_path $data_path --split train --fim_format v1 --balance_sample_key data_source --each_sample_key_num 766