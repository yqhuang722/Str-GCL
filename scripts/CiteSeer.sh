dataset="CiteSeer"
path="../datasets/"
model_type="MLPGCN"
log_dir="../log/CiteSeer.txt"
learning_rate=(0.01)
mlp_hidden_dim=(512)
n_components=(64)
num_hidden=(512)
num_proj_hidden=(256)
num_epochs=(1000)
weight_decay=(0.00001)
drop_edge_rate_1=(0.3)
drop_edge_rate_2=(0.2)
drop_feature_rate_1=(0.1)
drop_feature_rate_2=(0.1)
tau=(0.5)
tau_rule=(0.7)
loss_base=(1)
loss_rule=(1)
loss_cross=(1)
num_layers=(1)

python ../train.py --dataset $dataset  --log_dir $log_dir --path $path --model_type $model_type --learning_rate $learning_rate  --mlp_hidden_dim $mlp_hidden_dim   --num_hidden $num_hidden   --num_proj_hidden $num_proj_hidden --num_epochs $num_epochs --weight_decay $weight_decay --drop_edge_rate_1 $drop_edge_rate_1 --drop_edge_rate_2 $drop_edge_rate_2 --drop_feature_rate_1 $drop_feature_rate_1 --drop_feature_rate_2 $drop_feature_rate_2 --tau $tau --tau_rule $tau_rule --loss_base $loss_base --loss_rule $loss_rule --loss_cross $loss_cross --num_layers $num_layers --n_components $n_components