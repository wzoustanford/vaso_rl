# vaso_rl 

Code for: 
"Realistic CDSS Drug Dosing with End-to-end Recurrent Q-learning for Dual Vasopressor Control"

Neurips 2025 TS4H workshop 

# run training for binary Q-learning model with CQL penalty (specify alpha): 
python3 run_binary_cql_alpha00.py 

# run training for Dual Mixed Q-learning model with CQL penalty: 
python3 run_dualmixed_cql_allalphas.py 

# run training for Block Discrete (BD) Q-learning model: 
python3 run_block_discrete_cql_allalphas.py 

# run training for Directional Stepwise Q-learning model: 
python3 run_unified_stepwise_cql_allalphas.py --alpha 0.0 --max_step 0.2 

# run training for LSTM Block Discrete Q-learning model: 
python3 run_lstm_block_discrete_cql_with_logging.py 

# run the FQE off-policy evaluation: 
read and use ope_exp.py and fqe_gaussian_analysis.py, the pkl files could be saved during training 
