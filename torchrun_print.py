import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split    
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torchvision import transforms as T
import os
import tqdm
import logging
import numpy as np
import gc
from PIL import Image
import torch.fft

''' CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=gpu .py 4 2 '''

MD_PTH = '/media/data2/jiwon/'
path = '/media/data2/jiwon/VISION/dataset/base_files/'

class GroupDataset(Dataset):
    def __init__(self, homo_4_patches, transform, split='train') -> None:
        super().__init__()
        self.file_list = homo_4_patches
        self.transform = transform
        
    def __len__(self):
        return len(self.file_list['0'])
    
    def __getitem__(self, index):
        x = self.file_list[str(0)][index].get('patch')
        x_transformed = self.transform(x)
        
        y = self.file_list[str(1)][index].get('patch')
        y_transformed = self.transform(y)
        xy01 = torch.cat([x_transformed,y_transformed], dim=1)
        
        x = self.file_list[str(2)][index].get('patch')
        x_transformed = self.transform(x)
        
        y = self.file_list[str(3)][index].get('patch')
        y_transformed = self.transform(y)
        xy23 = torch.cat([x_transformed,y_transformed], dim=1)
        
        xy0123 = torch.cat([xy01,xy23], dim=1)
        return xy0123
    
'''이미지를 transform해서 전처리1 하자'''
class ImageTransform():

    def __init__(self, mean=([0.4741, 0.4799, 0.4672]), std=[0.2029, 0.2042, 0.2034]):
        self.data_transform = T.Compose([
            T.ToTensor(),
            # T.Normalize(mean, std)
        ])

    def __call__(self, img):
        return self.data_transform(img)

def make_dir_dict_list(frame_index): # return img_dict_list, len(img_dict_list)
    img_dict_list = list()
    patern_class_list = ["flat", "natFBH", "nat", "natWA"]
    str_frame_index  = "{:04d}".format(frame_index)
    
    for camera in range(1,36):
        str_index_camera = str(camera)
        if camera < 10: str_index_camera = "0"+str_index_camera
        img_4_per_camera = list()
        
        for x in range(0,4):
            img_name = "D"+str_index_camera+"_I_"+ patern_class_list[x] +"_" + str_frame_index+".jpg"
            img_4_per_camera.append({"img_path": path+img_name, 'type': x})

        img_dict_list.append({'camera': camera, "img_4_per_camera": img_4_per_camera, 'frame': frame_index})
        
    return img_dict_list, len(img_dict_list)

def create_united_4_homo_dataset(frame_index=60, patch_size=64, hom_patches=4):
    
    # 일단 이미지가 4개 필요함
    img_dict_list, num_images = make_dir_dict_list(frame_index)
    
    homo_4_patches = {'0':list(), '1': list(), '2':list(), '3': list()}
    for dict in img_dict_list:
        # print(dict)
        camera = dict.get('camera', None)
        img_4_per_camera = dict.get('img_4_per_camera', None) # img_4_per_camera에서 4개의 같은 카메라 출신 이미지 4개가 담겨있음
        frame = dict.get('frame', None)
        
        for idx in range(0,hom_patches):
            img_type_dict = img_4_per_camera[idx]
            type = img_type_dict.get('type', None)
            if idx != type: print(idx, type)
            
            with Image.open(img_type_dict.get('img_path', None)) as img:
                img = img.resize((patch_size*20, patch_size*20)) # 64*20 = 1280*1280
                # 패치 자르기
                for y in range(0, img.height, patch_size):
                    for x in range(0, img.width, patch_size):
                        # dict 으로 저장 
                        patch = {'patch': img.crop((x, y, x+patch_size, y+patch_size)), 'camera':camera, 'pos':(x,y)}
                        # 리스트에 dict 저장 
                        homo_4_patches[str(type)].append(patch)
                        

    return homo_4_patches    

def Accuracy(output, targets):
    preds = output.argmax(dim=1)    # 가장 높은 값을 가진 인덱스를 출력한다. 
    return int( preds.eq(targets).sum() )

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calculate_norm(dataset):
    # dataset의 axis=1, 2에 대한 평균 산출
    mean_ = np.array([np.mean(x.numpy(), axis=(1, 2)) for x, _ in dataset])
    # r, g, b 채널에 대한 각각의 평균 산출
    mean_r = mean_[:, 0].mean()
    mean_g = mean_[:, 1].mean()
    mean_b = mean_[:, 2].mean()

    # dataset의 axis=1, 2에 대한 표준편차 산출
    std_ = np.array([np.std(x.numpy(), axis=(1, 2)) for x, _ in dataset])
    # r, g, b 채널에 대한 각각의 표준편차 산출
    std_r = std_[:, 0].mean()
    std_g = std_[:, 1].mean()
    std_b = std_[:, 2].mean()
    
    return (mean_r, mean_g, mean_b), (std_r, std_g, std_b)

def create_save_dir(name, default_dir = 'result'):
    try:
        check_mkdir(default_dir)
        check_mkdir(default_dir+'/'+name)
    except:
        print("Error : Failed to create the directory")
    return default_dir +'/'+name+'/'
#
def check_mkdir(dir_name):
    if not os.path.exists(dir_name): os.mkdir(dir_name)
    return dir_name
#
def get_logger(path, mode='train'):
    logger = logging.getLogger()
    if len(logger.handlers) > 0 : return logger
    
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    file_handler = logging.FileHandler(os.path.join(path, mode+'.log' )) # 'train.log'
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        # Adding first Conv+ReLU layer
        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))

        # Adding Conv+BN+ReLU layers
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels))
            layers.append(nn.ReLU(inplace=True))

        # Adding last Conv layer
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        noise = self.dncnn(x)
        return noise, x-noise, noise-x
    
class DistanceBasedLogitLoss(nn.Module):
    def __init__(self, len = 200, margin=1.0, lambda_ = 0.1):
        super(DistanceBasedLogitLoss, self).__init__()
        self.margin = margin
        self.lambda_ = lambda_
        self.len = len
        # self.upper_matrix = torch.triu(torch.ones((self.len,self.len)), diagonal=1)

    def forward(self, r_matrix): # [200,64,64]
        self.len = r_matrix.shape[0]
        self.positive_matrix = torch.zeros((self.len,self.len))
        self.neg_matrix = torch.zeros((self.len,self.len))
        for i in range(self.len):
            i_ = int(i/4)*4
            for j in range(i,self.len):
                if i == j : continue
                j_ = int(j/4)*4
                euclidean_distance_ij = F.pairwise_distance(r_matrix[i].reshape(-1), r_matrix[j].reshape(-1)) 
                #((r_matrix[i] - r_matrix[j])**2).sum()**.5
                # print(euclidean_distance_ij, euclidean_distance_ij.shape)
                if i_!= j_ : self.neg_matrix[i,j] = -euclidean_distance_ij
                else : self.positive_matrix[i,j] = -euclidean_distance_ij

        S_matrix = torch.sum(self.neg_matrix) + torch.sum(self.positive_matrix)
        Pos_prob = self.positive_matrix*(1/S_matrix)
        Pos_prob = Pos_prob + Pos_prob.T
        loss_i = -torch.log(torch.sum(Pos_prob, axis=0))
        loss_all = torch.sum(loss_i)

        ff_matrix = torch.fft.fftn(r_matrix, dim=[1, 2]) # 200 x 64 x 64
        psd = torch.mean(torch.abs(ff_matrix) ** 2, dim=0) # 64 x 64
        # print(psd.shape) # torch.Size([64, 64])
        left = torch.mean(torch.log(psd))
        right = torch.log(torch.mean(psd))
        reg_term = left - right

        return loss_all - self.lambda_*reg_term	



'''*Newrly Updated For Torchrun Multi GPU DDP*'''
def ddp_setup():
    init_process_group(backend="nccl")

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn,
        save_every: int,
        snapshot_path: str, #'''*Newrly Updated For Torchrun Multi GPU DDP*'''
        pj_name: str,
        valid_data = False
    ) -> None:
        '''*Newrly Updated For Torchrun Multi GPU DDP*'''
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.valid_data = valid_data
        self.dataloader = {'train': train_data}
        if valid_data != False :
            self.dataloader['valid'] = valid_data
        self.optimizer = optimizer
        self.criterion = loss_fn
        self.save_every = save_every
        
        '''*Newrly Updated For Torchrun Multi GPU DDP*'''
        self.epochs_run = 0
        PJ_PTH = check_mkdir(os.path.join(MD_PTH, pj_name))
        SAVE_PTH = check_mkdir( os.path.join(PJ_PTH, 'save'))
        if self.gpu_id == 0:    
            self.logger = get_logger(SAVE_PTH)
                
        
        self.snapshot_path = os.path.join(SAVE_PTH, snapshot_path)
        if os.path.exists(self.snapshot_path):  self._load_snapshot()

        self.model = DDP(self.model, device_ids=[self.gpu_id])
        self.mode = 'train'
        

    '''*Newrly Updated For Torchrun Multi GPU DDP*'''
    def _load_snapshot(self):
        self.printonce("Loading snapshot")
        loc = f"cuda:{self.gpu_id}"
        
        snapshot = torch.load(self.snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        
        self.printonce(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, bt_sz):
        pt_sz = 64
        self.optimizer.zero_grad()
        source = source.reshape([bt_sz*4, 3, pt_sz, pt_sz])
        output, in_out, out_in = self.model(source)
        output = output.reshape([bt_sz* 4, pt_sz, pt_sz])
        loss = self.criterion(output)
        loss.backward(retain_graph=True)
        self.optimizer.step()
        return loss.item()

    def _run_epoch(self, epoch, log):
        b_sz = len(next(iter(self.dataloader.get(self.mode)))[0])
        losses = AverageMeter()
        
        self.dataloader.get(self.mode).sampler.set_epoch(epoch)
        
        tq = tqdm.tqdm(total=len(self.dataloader.get(self.mode))*b_sz)
        tq.set_description(f"[GPU {self.gpu_id} - {self.mode}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.dataloader.get(self.mode))}")
        for b_idx, (source) in enumerate(self.dataloader.get(self.mode)):
            mini_b_sz = source.shape[0]
            source = source.to(self.gpu_id)
            loss = self._run_batch(source, mini_b_sz)
            losses.update(loss, mini_b_sz)
            tq.set_postfix(loss='{:.5f}'.format(losses.avg))
            tq.update(b_sz)
        tq.close()
        log[f'{self.mode} loss'] = losses.avg
        return log

    '''*Newrly Updated For Torchrun Multi GPU DDP*'''
    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, os.path.join(self.snapshot_path))
        # self.printonce(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    '''*Newrly Updated For Torchrun Multi GPU DDP*'''
    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            log = {'epoch': epoch}
            self.printonce('\n')
            #train
            self.mode = 'train'
            self.model.train()
            log = self._run_epoch(epoch, log)
            
            if self.valid_data != False:
                #valid
                self.mode = 'valid'
                self.model.eval()
                log = self._run_epoch(epoch, log)
                
            if self.gpu_id == 0 :
                self.logger.info(log)
                if epoch % self.save_every == 0:    self._save_snapshot(epoch)
            
        gc.collect()
        torch.cuda.empty_cache()

    def printonce(self,message:str):
        if self.gpu_id == 0: print(message)
            
def load_train_objs():
    homo_4_patches = create_united_4_homo_dataset(frame_index=60, patch_size=64, hom_patches=4)
    train_set = GroupDataset(homo_4_patches, transform=ImageTransform()) # load your dataset
    model =  DnCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4) #torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer

def split_set(dataset: Dataset, scale=1, prop=.8):
    origin_sz = int(len(dataset))
    use_sz = int(origin_sz* scale)
    if scale < 1 : dataset, _ = random_split(dataset, [use_sz, origin_sz-use_sz])
    print(int(use_sz*prop), use_sz-int(use_sz*prop))
    train, test = random_split(dataset, [int(use_sz*prop), use_sz-int(use_sz*prop)])
    return train, test

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

'''*Newrly Updated For Torchrun Multi GPU DDP*'''
def main(save_every: int, total_epochs: int, batch_size: int, pj_name:str, snapshot_path: str = "snapshot.pt"):
    gc.collect()
    torch.cuda.empty_cache()
    
    ddp_setup()
    dataset, model, optimizer = load_train_objs()
    train_set, valid_set =  split_set(dataset, scale=1, prop=.8)
    train_loader = prepare_dataloader(train_set, batch_size)
    valid_loader = prepare_dataloader(valid_set, batch_size)
    
    loss_fn = DistanceBasedLogitLoss(len = batch_size*4)
    
    trainer = Trainer(model, train_loader, optimizer, loss_fn, save_every, snapshot_path, pj_name, valid_loader)
    trainer.train(total_epochs)
    destroy_process_group()


'''*Newrly Updated For Torchrun Multi GPU DDP*'''
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--pj_name', '-n',  type=str, default='noiseprint', help='What is your project name')
    parser.add_argument('--batch_size', '-b', default=50, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    main(args.save_every, args.total_epochs, args.batch_size, args.pj_name)