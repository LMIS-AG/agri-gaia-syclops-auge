import subprocess

cmd = ['accelerate launch train_dreambooth.py',
        '--pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"',
        '--instance_data_dir=$INSTANCE_DIR',
        '--output_dir=$OUTPUT_DIR',
        '--instance_prompt="a closeup photo of a leaf with rust disease"',
        '--resolution=512',
        '--train_batch_size=1 ',
        '--gradient_accumulation_steps=1',
        '--learning_rate=5e-6 ',
        '--lr_scheduler="constant"',
        '--lr_warmup_steps=0',
        '--max_train_steps=400']

subprocess.run(cmd)

