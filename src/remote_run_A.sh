#!/usr/bin/env bash
#exp_name
prefix=EXP_A
gpu=0

## Dimension
relation_dim=64
node_dim=64
message_dim=64
memory_dim=64
time_dim=64
## Structure of Attention
n_layer=1
n_head=2
n_degree=40
## Initial weight
mem_weight=0.1
memEmb_weight=0.9
## Description of dataset
data="A"
min_period=1
max_period=1300
split_time=0
n_relation=249 # num of relation, equals max_id+1
## Batch size = bs_predict+bs_neg_from_random+bs_neg_from_positive*3
bs=1024
bs_predict=1024
bs_neg_from_random=1024
bs_neg_from_positive=1024
n_epoch=10
n_runs=1
## LR
lr=0.001
drop_out=0.1
## Balance strategy of loss weight
pos_weight=1.0
bl_relation="none"
## Sampling strategy
random_method_random="local" #random,global,local
random_method_positive="local" #random,global,local
# period of TIME POSITIVE
shift_times_n=0
shift_base_n=36
shift_span_n=12
# period of TIME NEGATIVE
shift_times_p=0
shift_base_p=0
shift_span_p=12

use_random_relation="False"
mem_as_predict="False"
reply_times=1


## Module
attn_mode="w_loop" #n_loop,w_loop
embedding_module="graph_attention"
classifier_fun="X_MMLP" #multi_mlp
method_init_memory="zero"
jump_method="jump"
share_embedding="False"
use_edge_feat_as_emb="False"
use_node_feat="False"
use_edge_feat="False"
normal_time_diff="True"
init_emb_method="just_mem"
message_function="mlp"
memory_updater="gru_v2"
time_diff_method="from_memory"
uniform="False"
use_edge_type_feats="True"

# garbage
patience=100
backprop_every=1
use_memory="True"
aggregator="last"
memory_update_at_end="False"
different_new_nodes="False"
randomize_features="False"
use_destination_embedding_in_message="False"
use_source_embedding_in_message="False"
update_memory_rate_in_val=0
update_memory_rate_in_test=0
ngh_finder_in_val="train"
ngh_finder_in_test="train"

#cp run_A.sh ./shell_scripts/run_A_${prefix}.sh

python train_self_supervised_v2.py  \
--data ${data} \
--share_embedding ${share_embedding} \
--use_node_feat ${use_node_feat} \
--use_edge_feat ${use_edge_feat} \
--use_edge_feat_as_emb ${use_edge_feat_as_emb} \
--relation_dim ${relation_dim} \
--normal_time_diff ${normal_time_diff} \
--bs ${bs} \
--bs_predict ${bs_predict} \
--bs_neg_from_random ${bs_neg_from_random} \
--bs_neg_from_positive ${bs_neg_from_positive} \
--min_period ${min_period} \
--max_period ${max_period} \
--jump_method ${jump_method} \
--prefix ${prefix} \
--n_degree ${n_degree} \
--n_head ${n_head} \
--n_epoch ${n_epoch} \
--n_layer ${n_layer} \
--lr ${lr} \
--patience ${patience} \
--n_runs ${n_runs} \
--drop_out ${drop_out} \
--gpu ${gpu} \
--backprop_every ${backprop_every} \
--use_memory ${use_memory} \
--embedding_module ${embedding_module} \
--message_function ${message_function} \
--memory_updater ${memory_updater} \
--aggregator ${aggregator} \
--memory_update_at_end ${memory_update_at_end} \
--different_new_nodes ${different_new_nodes} \
--uniform ${uniform} \
--randomize_features ${randomize_features} \
--use_destination_embedding_in_message ${use_destination_embedding_in_message} \
--use_source_embedding_in_message ${use_source_embedding_in_message} \
--node_dim ${node_dim} \
--time_dim ${time_dim} \
--message_dim ${message_dim} \
--memory_dim ${memory_dim} \
--init_emb_method ${init_emb_method} \
--update_memory_rate_in_val ${update_memory_rate_in_val} \
--update_memory_rate_in_test ${update_memory_rate_in_test} \
--ngh_finder_in_val ${ngh_finder_in_val} \
--ngh_finder_in_test ${ngh_finder_in_test} \
--n_relation ${n_relation} \
--classifier_fun ${classifier_fun} \
--time_diff_method ${time_diff_method} \
--split_time ${split_time} \
--method_init_memory ${method_init_memory} \
--random_method_random ${random_method_random} \
--random_method_positive ${random_method_positive} \
--reply_times ${reply_times} \
--use_random_relation ${use_random_relation} \
--mem_as_predict ${mem_as_predict} \
--attn_mode ${attn_mode} \
--bl_relation ${bl_relation} \
--shift_times_n ${shift_times_n} \
--shift_base_n ${shift_base_n} \
--shift_span_n ${shift_span_n} \
--shift_times_p ${shift_times_p} \
--shift_base_p ${shift_base_p} \
--shift_span_p ${shift_span_p} \
--use_edge_type_feats ${use_edge_type_feats} \
--pos_weight ${pos_weight} \
--mem_weight ${mem_weight} \
--memEmb_weight ${memEmb_weight}