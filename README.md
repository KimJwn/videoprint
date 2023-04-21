# videoprint

# exam_print.py : 
using single gpu, toyset, normalized transform

# torchrun_print.py : 
using multi gpu, toyset

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=gpu torchrun_templete.py 10 1


# TSNE_video.ipynb.py : 
using pretrained model check the tsne or noiseprint of images/videos


