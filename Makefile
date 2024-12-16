
setup:
	pip install -r requirements.txt

check:
	ruff format
	ruff check --fix

train-bald-converter:
	accelerate launch train_controlnet_sdxl.py  \
		--pretrained_model_name_or_path=$(MODEL_DIR) \
		--output_dir=$(OUTPUT_DIR) \
		--dataset_name=$(DATASET_PATH) \
		--mixed_precision="no" \
		--resolution=1024 \
		--learning_rate=5e-5 \
		--validation_image "$(DATASET_PATH)/hair/00003.png" "$(DATASET_PATH)/hair/00083.png" \
		--validation_prompt "" "" \
		--max_train_steps=10000 \
		--train_batch_size=4 \
		--gradient_accumulation_steps=4