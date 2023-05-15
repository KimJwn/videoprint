# noiseprint/Inference/Inference.py

import os, sys, random, warnings, argparse, cv2, json
sys.path.append('../')
sys.path.append('/home/jiwon/anaconda3/envs/intern2/lib/python3.8/site-packages/')
sys.path.append('/home/jiwon/csiro/noiseprint/')

import dlib
face_detector = dlib.get_frontal_face_detector()

import numpy as np
import pandas as pd
import gc
import multiprocessing as mp
import seaborn as sns
warnings.filterwarnings('ignore')

from sys import argv
from time import time
from PIL import Image
from PIL.JpegImagePlugin import convert_dict_qtables
from os.path import join
from os import cpu_count
from multiprocessing.pool import Pool
from collections import OrderedDict
from glob import glob
from tqdm import tqdm 
from functools import partial 

# for interactive plot
# If you use this option, plot will appear at first-drawn position
import matplotlib #%matplotlib notebook
import matplotlib.pyplot as plt
plt.style.use('ggplot') # You can also use different style

from sklearn.decomposition import NMF # Use this for training Non-negative Matrix Factorization
from sklearn.utils.extmath import randomized_svd # Use this for training Singular Value Dec
from sklearn.manifold import TSNE # Use this for training t-sne manifolding
# import mplcursors # Use this is for creating a cursor-interactive plot with "%matplotlib notebook"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split  
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

import torchvision
import torchvision.models as models
from torchvision import transforms as T

import utils as ut
tf = T.ToTensor()

from scipy.stats import chi2
from sklearn.mixture import GaussianMixture

def splicebuster(image, mask_path=None, block_size=16, n_components=2, threshold=15, verbose=False):
    if mask_path:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 128).astype(np.uint8)
        image = cv2.bitwise_and(image, mask)
    # Compute feature vectors for each block
    feature_vectors = []
    for i in range(0, image.shape[0], block_size):
        for j in range(0, image.shape[1], block_size):
            block = image[i:i+block_size, j:j+block_size]
            cooc_mat = cv2.calcHist([block], [0], None, [256], [0, 256])
            cooc_mat /= cooc_mat.sum()
            feature_vectors.append(cooc_mat.flatten())
    feature_vectors = np.array(feature_vectors)
    # GMM clustering
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(feature_vectors)
    predicted_labels = gmm.predict(feature_vectors)
    # Compute Mahalanobis distance for each block
    mahalanobis_distances = []
    for i, feature_vector in enumerate(feature_vectors):
        mean = gmm.means_[predicted_labels[i]]
        covariance = gmm.covariances_[predicted_labels[i]]
        mahalanobis_distance = np.sqrt(np.dot(np.dot((feature_vector-mean).T, np.linalg.inv(covariance)), feature_vector-mean))
        mahalanobis_distances.append(mahalanobis_distance)
    mahalanobis_distances = np.array(mahalanobis_distances)
    # Compute chi-square distribution threshold
    chi_square_threshold = chi2.ppf(q=1-threshold/100, df=feature_vectors.shape[1])
    n_blocks_h = (image.shape[0] - 1) // block_size + 1
    n_blocks_w = (image.shape[1] - 1) // block_size + 1
    # Reshape predicted_labels and mahalanobis_distances back into the shape of the image
    predicted_labels = predicted_labels.reshape(n_blocks_h, n_blocks_w)
    mahalanobis_distances = mahalanobis_distances.reshape(n_blocks_h, n_blocks_w)
    # Apply threshold and generate output map
    output_map = np.zeros_like(predicted_labels, dtype=np.uint8)
    output_map[mahalanobis_distances > chi_square_threshold] = 255
    # Display output map if verbose=True
    if verbose:
        cv2.imshow("Output Map", output_map)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return output_map, mahalanobis_distances, predicted_labels, chi_square_threshold

class FrameDataset(Dataset):
    def __init__(self, frame_dir, transform=tf) -> None:
        super().__init__()
        self.frame_dir_list = os.listdir(frame_dir)
        self.frame_dir= frame_dir
        self.transform = transform
        self.sample_img = Image.open(frame_dir+'/0000.png')
        self.w, self.h = self.sample_img.size
        self.sample_img = self.transform(self.sample_img)
        
    def __len__(self):
        return len(self.frame_dir_list)
    
    def __getitem__(self, index):
        x_transformed = self.sample_img
        frame_idx =  "{:04d}.png".format(index)
        frame_name = join(self.frame_dir, frame_idx)
        with Image.open(frame_name) as x:
            x_transformed = self.transform(x)#.unsqueeze(0)
        return x_transformed
    
def transe_np(tensor_img):
    return np.transpose(tensor_img.squeeze(1).detach().cpu().numpy(), (1,2,0))

def averaging_stacked_noises(output_concate):
    if len(output_concate) != 0:
        # first = output_concate[0].squeeze(1).detach().numpy()
        first = transe_np(output_concate[0])
        height, width, batch_sz = first.shape
        print(height, width)
        #sum_noise = np.zeros_like(first)
        sum_noise = np.sum(first, axis=2) 
        for i in range(1, len(output_concate)):
            n_noise = transe_np(output_concate[i])
            n_noise = np.sum(n_noise, axis=2)
            sum_noise += n_noise
        mean_noise = sum_noise/len(output_concate)/batch_sz
    return sum_noise, mean_noise

def save_tensor_img(tensor_img, dir, frame_num):
    np_img = np.transpose(tensor_img.squeeze(0).detach().numpy(), (1,2,0))
    tnp_save = join(dir, '{:04d}.png'.format(frame_num))
    cv2.imwrite(tnp_save, np_img)

def biggest_face_idx(faces):
    max_area = 0
    max_idx = 0
    for idx, face in enumerate(faces):
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        area_idx = (x2 - x1) * (y2 - y1)
        if area_idx > max_area:
            max_area = area_idx
            max_idx = idx
    return max_idx

def get_boundingbox(result_dir, face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb

def read_video(result_dir, input_root, video_name, file_name, folder_save, margin_scale=1.3):
    filename = join(input_root, file_name)
    # if 'raw' in filename:
    face_locate = OrderedDict() 
    
    # 원본 프레임 이미지들을 저장할 폴더를 생성합시다.
    frame_dir = join(folder_save, 'frame')
    os.makedirs(frame_dir, exist_ok=True)
    face_dir = join(folder_save, 'face')
    os.makedirs(face_dir, exist_ok=True)
    noiseprint_dir = join(folder_save, 'noiseprint')
    os.makedirs(noiseprint_dir, exist_ok=True)
    
    # 영상에서 프레임들을 추출합시다.
    reader = cv2.VideoCapture(filename)
    # 프레임 수 구하기
    frame_total = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    print("프레임 수: ", frame_total)

    frame_num = 0
    
    tq = tqdm(total=frame_total*(frame_num+1))
    tq.set_description(f"[{frame_total}] | frame_num: {frame_num}")
    
    while reader.isOpened():
        success, image = reader.read()
        if not success: break
        if image is None:  continue
        
        # 우선 원본 프레임 이미지를 저장해봅시다. 
        file_save = join(frame_dir, '{:04d}.png'.format(frame_num))
        cv2.imwrite(file_save, image)
        
        #원본 프레임을 모델에 적용해봅시다. 텐서로 바꿔서 모델에 적용해보아요.
        # tensor_img = tf(image).unsqueeze(0)
        # out, o_i, i_o = model(tensor_img)
        # save_tensor_img(out, noiseprint_dir, frame_num)
        
        # 이미지에 얼굴 영역을 추출해봅시다.
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1) # face_detector = dlib.get_frontal_face_detector()
        
        if len(faces):
            face_idx = 0 if len(faces)==1 else biggest_face_idx(faces)
            
            # For now only take biggest face
            face = faces[face_idx]
            x, y, size = get_boundingbox(result_dir, face, width, height, scale=1.3, minsize=None)
            cropped_face = image[y:y+size, x:x+size]
            
            # 얼굴만 남기고 나머지 배경을 제거한 이미지를 생성해요.
            image[:y, :] = 0
            image[y+size:, :] = 0
            image[:, :x] = 0
            image[:, x+size:] = 0
            
            file_save = os.path.join(face_dir, '{:04d}.png'.format(frame_num))
            # if os.path.exists(file_save):
            #         continue
            success_ = False
            try: 
                cv2.imwrite(file_save, image)
                success_ = True
            except: pass
            if success_: face_locate[frame_num] = [x, y, size]
        frame_num += 1 
        tq.set_postfix(frame='{:.2f}'.format(frame_num))
        tq.update(frame_num)        
        #
    tq.close()
    print('[Done]  Number of frame counted: ', frame_num)           
    reader.release()       
    with open(os.path.join(folder_save, 'face_locate.json'), 'w') as outfile:
            json.dump(face_locate, outfile)
    return 1

def read_frames_save_faces(result_dir, video_dir_list, input_root, margin_scale=1.3):
    
    keyNames = video_dir_list
    print(input_root)
    try:
        for keyname in keyNames:
            
            video_name = keyname.split('/')[-1][:-4] # Not include '.mp4'
            # if video_name == '1921__outside_talking_pan_laughing': continue
            # if video_name == '0308__podium_speech_happy': continue
            # if video_name == '1405__hugging_happy': continue
            # if video_name == '1512__exit_phone_room': continue
            # if video_name == '1727__walking_and_outside_surprised': continue
            # if video_name == '2111__outside_talking_pan_laughing': continue
            # if video_name == '2610__kitchen_still': continue
            # if video_name == '2701__meeting_serious': continue
        
            folder_save = join(result_dir, video_name)
            print(f'\nVideo [{video_name}] would be saved in [{folder_save}]')
            os.makedirs(folder_save, exist_ok=True)
            
            read_video(result_dir, input_root, video_name, keyname, folder_save, margin_scale) 

        return "Reading Done!"
    except Exception as e:
        print(e)
        return "Reading Error!"

def save_tensor_img(tensor_img, dir, frame_num):
    np_img = np.transpose(tensor_img.squeeze(0).detach().numpy(), (1,2,0))
    tnp_save = join(dir, '{:04d}.png'.format(frame_num))
    cv2.imwrite(tnp_save, np_img)

def ddp_setup():
    init_process_group(backend="nccl")

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

def model_setup(model_dir, gpu_id):
    model_ = torch.load(model_dir)#'/media/data2/jiwon/noiseprint/save/e200s10/snapshot.pt'
    model = ut.DnCNN()
    model.load_state_dict(model_['MODEL_STATE'])
    epochs_run = model_["EPOCHS_RUN"]
    model.eval() # (dropout) and (batch normalization layers) change mode to (evaluation)
    model = model.to(gpu_id)
    model = DDP(model, device_ids=[gpu_id])
    print(f"Load model trained at Epoch {epochs_run} from {model_dir}")
    return model

def run(dataloader, gpu_id, model):    
    gc.collect()
    torch.cuda.empty_cache()
    output_concate = list()
    b_sz = len(next(iter(dataloader))[0])
    tq = tqdm(total=len(dataloader)*b_sz)
    tq.set_description(f"[GPU {gpu_id}] | Batchsize: {b_sz} | Steps: {len(dataloader)}")
    with torch.no_grad():
        for b_idx, (source) in enumerate(dataloader):
            mini_b_sz = source.shape[0]
            source = source.to(gpu_id)
            output, in_out, out_in = model(source)
            output_concate.append(output)
            tq.set_postfix(IDX='{:.2f}'.format(b_idx))
            tq.update(b_sz)
    tq.close()
    return output_concate
    
def main(dataset: Dataset, batch_size: int, model_dir: str, pj_name:str, input_root: str, output_root:str):

    gpu_id = int(os.environ["LOCAL_RANK"])
    
    # 1. bring model
    model = model_setup(model_dir, gpu_id)
    # 1-2. initialize optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4) #torch.optim.SGD(model.parameters(), lr=1e-3)
        
    # 4. initialize dataset
    loader = prepare_dataloader(dataset, args.batch_size)

    output_concate = run(loader, gpu_id, model)
        
    return output_concate
    

#
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch_Size')
    parser.add_argument('--save_frames', '-f', default=False, type=bool, help='Save frame image before inference')
    parser.add_argument('--pj_name', '-n',  type=str, default='Dfaker', help='What is your project name? It also could be your result folder')
    parser.add_argument('--model_dir', '-m', default='/media/data2/jiwon/CSIRO/nopiseprint_model_save/e200s10/snapshot.pt', type=str, help='Exact directory of the pre-trained model')
    parser.add_argument('--input_root', '-i', default='/media/data2/jiwon/CSIRO/generated_video/', type=str, help='Exact directory of the pre-trained model')
    parser.add_argument('--output_root', '-o', default='/media/data2/jiwon/CSIRO/Inference/', type=str, help='Exact directory of the pre-trained model')
    args = parser.parse_args()    
    
    ddp_setup()
    
    # 2. get the all of videos in the input_root
    video_dir_list = os.listdir(join(args.input_root, args.pj_name))
    print(f'[{join(args.input_root, args.pj_name)}] There are total {len( video_dir_list)} videos!')
    # 2-2. make sure that the result folder exist actually
    os.makedirs(args.output_root, exist_ok=True)
    result_dir = join(args.output_root, f'{args.pj_name}/')
    os.makedirs(result_dir, exist_ok=True)

    # 3. read video and transform every frame
    print(f'[{args.save_frames}] - frames are already extracted.')
    if args.save_frames:
        read_frames_save_faces(result_dir, video_dir_list, join(args.input_root, args.pj_name), margin_scale=1.3)
        
    # 4. initialize dataset
    dataloader_list = list()
    output_concate = None
    cnt = 1
    for keyname in video_dir_list:
        video_name = keyname.split('/')[-1][:-4] # Not include '.mp4'
        print(f'[{video_name}] Count: {cnt}')
        
        result_dir = join(args.output_root, args.pj_name)
        video_result_dir = join(result_dir, video_name)
        frame_dir = join(video_result_dir, 'frame')
        
        dataset = FrameDataset(frame_dir)
    
        output_concate = main(dataset, args.batch_size, args.model_dir, args.pj_name, args.input_root, args.output_root)
        sum_noise, mean_noise = averaging_stacked_noises(output_concate)
    
        img_color = cv2.cvtColor(mean_noise, cv2.COLOR_GRAY2BGR)
        gray_img = np.mean(img_color* 255.0, axis=2).astype(np.uint8)
        
        a = splicebuster(gray_img, mask_path=None, block_size=16, n_components=2, threshold=15, verbose=False)
        temp_map = a[1].astype(np.uint8)
        # heatmap = cv2.applyColorMap(temp_map, cv2.COLORMAP_JET)
        sns_plot = sns.heatmap(temp_map, cmap='coolwarm', alpha=0.5, cbar=False, linewidths=0)
        sns_plot.axis('off')
        sns_plot.figure.savefig(join(video_result_dir, 'noise.png'), bbox_inches='tight')
        
    print("\n[DONE]")
    destroy_process_group()