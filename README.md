# vaso_rl 

Code for: 
"Realistic CDSS Drug Dosing with End-to-end Recurrent Q-learning for Dual Vasopressor Control"

Neurips 2025 TS4H workshop 

move data "sample_data_oviss.csv" to vaso_rl folder 

### training binary Q-learning model with CQL penalty (specify alpha): 
python3 run_binary_cql_alpha00.py 

### training Dual Mixed Q-learning model with CQL penalty: 
python3 run_dualmixed_cql_allalphas.py 

### training Block Discrete (BD) Q-learning model: 
python3 run_block_discrete_cql_allalphas.py 

### training Directional Stepwise Q-learning model: 
python3 run_unified_stepwise_cql_allalphas.py --alpha 0.0 --max_step 0.2 

### training LSTM Block Discrete Q-learning model: 
python3 run_lstm_block_discrete_cql_with_logging.py 

### FQE off-policy evaluation: 
read and use ope_exp.py and fqe_gaussian_analysis.py, the pkl files could be saved during training 

### WIS importance sampling evaluation: 
read and use is_block_discrete.py, for example of evaluating the block discrete model. The same evaluation can be implemented for other models by changing the model definition and loading paths 

### reward functions: 
neurips reward definition is given in integrated_data_pipeline_v2_simple_reward.py, and OVISS reward definition is given in integrated_data_pipeline_v2.py 

### hyper-parameters 

# neurips simple reward 
Learning Rate (lr) 10−3 Adam optimizer learning rate
Batch Size 128 Training batch size
Gamma (γ) 0.95 Discount factor for future rewards
Tau (τ ) 0.8 Soft target network update rate
Gradient Clipping 1.0 Maximum gradient norm
Epochs 100 Training epochs
Validation Batches 10 Batches for validation evaluation
Random Seed 42 For reproducibility 

# oviss reward 
Learning Rate (lr) 10−3 Adam optimizer learning rate
Batch Size 128 Training batch size
Gamma (γ) 0.99 Discount factor for future rewards [aligned with oviss]
Tau (τ ) 0.95 Soft target network update rate [more aligned with oviss using FQI]
Gradient Clipping 1.0 Maximum gradient norm
Epochs 100 Training epochs
Validation Batches 10 Batches for validation evaluation
Random Seed 42 For reproducibility