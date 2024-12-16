
setup:
	pip install -r requirements.txt

check:
	ruff format
	ruff check --fix

train-bald-converter:
	accelerate launch train_controlnet_sdxl.py  \
		--pretrained_model_name_or_path=$(MODEL_DIR) \
		--output_dir=$(OUTPUT_DIR)\
		--dataset_name=/app/dataset/images/non_hair_ffhq\
		--mixed_precision="fp16" \
		--resolution=1024 \
		--learning_rate=5e-5 \
		--max_train_steps=10000 \
		--train_batch_size=4 \
		--gradient_accumulation_steps=4