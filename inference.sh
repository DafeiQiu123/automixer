python models/predict_validation.py \
  --a_path /users/rye13/automixer/data/inference/000002_2.wav \
  --b_path /users/rye13/automixer/data/inference/000005_1.wav \
  --ckpt_dir exp/checkpoints_d128_L1_pe_naive \
  --output_dir  /users/rye13/automixer/data/inference/ \
  --output_json_path /users/rye13/automixer/data/inference/vanilla.json \
  --save_time_series \
  --pe_type naive --d_model_override 128 --num_layers_override 1
