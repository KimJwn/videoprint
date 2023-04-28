# videoprint

# exam_print.py : 
using single gpu, toyset, normalized transform

# torchrun_print.py : 
using multi gpu, toyset

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=gpu torchrun_templete.py 10 1 pj 50
- 10: epoch
- 1 : saving points
- pj: Folder name for your project
- 50: Batch_size
---
- Set the `CUDA_VISIBLE_DEVICES` environment variable 
- Call the `torchrun` function with the `--standalone` and `--nproc_per_node=gpu` options and pass the `.py` file and two arguments 'epoch', 'saving points'
- Exchange the `MD_PTH` variable from `'/media/data2/jiwon/'` to your root path.
- Exchange the `path` variable from `'/media/data2/jiwon/VISION/dataset/base_files/'` to your media path.
---
- the `GroupDataset` class that inherits from the `Dataset` class: Initialize the `file_list` and `transform` attributes in the constructor.
- `__getitem__` method that returns a concatenated tensor of transformed patches extracted from the four positive images.
- `ImageTransform` class that applies a data transformation: composed of only `ToTensor` transformation on the input image. (not to use normalization)
- `make_dir_dict_list` function that returns a list of dictionaries, where each dictionary contains four image dictionaries with the same camera and frame index.
- `create_united_4_homo_dataset` function that creates a dictionary of four lists of patches, where each list corresponds to a 4 image type extracted from the [Vision].
---
- `DnCNN` class : Implement the forward method to apply the convolutional layers on the 3 channel input tensor and return the 1 channel output, and 3 channel of input - output, output - input.

- DnCNN is a deep convolutional neural network designed for image denoising. The model takes an input image with noise and outputs a denoised image. The model consists of a series of convolutional layers with batch normalization and ReLU activation functions. The number of layers and filters can be customized.



# TSNE_video.ipynb.py : 
using pretrained model check the tsne or noiseprint of images/videos

