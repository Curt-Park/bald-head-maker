
## Training
Download [non-hair-FFHQ](https://github.com/oneThousand1000/non-hair-FFHQ),
and store `hair` and `non-hair` images in `dataset`.

Example:
```bash
# Training
export MODEL_DIR="stabilityai/stable-diffusion-xl-base-1.0" 
export OUTPUT_DIR="/home/jovyan/train/no-hair-sdxl"
export DATASET_PATH="/app/dataset/images/non_hair_ffhq"
make train-bald-converter

# Test
python test_controlnet_sdxl.py \
    --controlnet-model-path $OUTPUT_DIR/checkpoint-1000/controlnet/diffusion_pytorch_model.safetensors \
    --input-image imgs/00003.png
```

## Powered By
```
@misc{zhang2024stablehairrealworldhairtransfer,
      title={Stable-Hair: Real-World Hair Transfer via Diffusion Model}, 
      author={Yuxuan Zhang and Qing Zhang and Yiren Song and Jiaming Liu},
      year={2024},
      eprint={2407.14078},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.14078}, 
}

@InProceedings{Wu_2022_CVPR,
    author    = {Wu, Yiqian and Yang, Yong-Liang and Jin, Xiaogang},
    title     = {HairMapper: Removing Hair From Portraits Using GANs},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {4227-4236}
}
```
`+` [mav-rik/facerestore_cf](https://github.com/mav-rik/facerestore_cf)