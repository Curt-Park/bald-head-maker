
setup:
	pip install -r requirements.txt

check:
	ruff format
	ruff check --fix

train-bald-converter:
	accelerate launch train_controlnet_sdxl.py  \
		--pretrained_model_name_or_path=$(MODEL_DIR) \
		--pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix \
		--output_dir=$(OUTPUT_DIR) \
		--dataset_name=$(DATASET_PATH) \
		--mixed_precision="fp16" \
		--resolution=1024 \
		--learning_rate=5e-5 \
		--validation_image="$(DATASET_PATH)/hair/00003.png" \
		--validation_prompt="" \
		--max_train_steps=15000 \
		--train_batch_size=4 \
		--gradient_accumulation_steps=4 \
		--gradient_checkpointing \
        --enable_xformers_memory_efficient_attention