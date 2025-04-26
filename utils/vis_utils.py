import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image

def vis_mat(mat, img=None, h=None, w=None, pos=None):
    plt.rcParams['figure.figsize'] = [44.8, 44.8]#[22.4, 22.4]
    if h is not None:
        plt.rcParams['figure.figsize'] = [w, h]
        plt.subplot(h, w, pos)
        ######
    if img is not None:
        # img = img.resize((224, 224))
        mat = mat.reshape(14, 14)
        assert(mat.shape[0] == 14)
        mat = mat.float()
        mat = nn.functional.interpolate(mat.unsqueeze(0).unsqueeze(0), size=(28, 28), mode="bilinear",
                                        align_corners=True)
        mat = nn.functional.interpolate(mat, size=(224, 224), mode="bilinear", align_corners=True) #
        mat = mat.squeeze(dim=0).squeeze(dim=0)
        # print(mat.shape)
        # print(f'max {mat.max()} min {mat.min()}')
        
        cm = mat/mat.max()
        cm = mat# /mat.max()
        plt.imshow(img)
        # else:
        #     plt.imshow(img, alpha=cm)
        plt.axis('off')
        if torch.isnan(cm.max()):
            pass
        else:
            plt.imshow(cm, alpha=0.6, cmap='hot')
            plt.axis('off')
    else:
        print("high-lighting")
        cm = mat
        w = cm.shape[0]
        max_ele = cm.max()/5
        for i in range(w):
            for j in range(w):
                cm[i, j] = min(max_ele, cm[i, j])
        plt.imshow(cm, cmap='viridis')
        plt.colorbar()
    # plt.title("Attention Map Visualization")
    # plt.xlabel("X-axis")
    # plt.ylabel("Y-axis")
    # plt.show()

if __name__ == '__main__':
    path = '/data/zhonghanwen_2022/23wb/SSF/ssf_note/SSF/output/confusion_matirx/Epoch50/dogs/True DINOV2 linear_probe adamw 0.000125 img_size-224 using_aug-0 tune_cls-0 cplx_head-0 gp-token seb-114514 -20230530-162205/confusion_res.pth'
    img_path = '/data/datasets/FGVC/stanford_cars/cars_train/05237.jpg'
    cm = torch.load(path)
    img = Image.open(img_path)
    vis_mat(cm)
    
    
    
    
    
    
    
    
# nan
"""

def vis_mat(mat, img=None, h=None, w=None, pos=None):
    plt.rcParams['figure.figsize'] = [44.8, 44.8]#[22.4, 22.4]
    if h is not None:
        plt.rcParams['figure.figsize'] = [w*3, h*3]
        plt.subplot(h, w, pos)
        ######
    if img is not None:
        # img = img.resize((224, 224))
        mat = mat.reshape(14, 14)
        assert(mat.shape[0] == 14)
        mat = mat.float()
        mat = nn.functional.interpolate(mat.unsqueeze(0).unsqueeze(0), size=(28, 28), mode="bilinear",
                                        align_corners=True)
        mat = nn.functional.interpolate(mat, size=(224, 224), mode="bilinear", align_corners=True) #
        mat = mat.squeeze(dim=0).squeeze(dim=0)
        # print(mat.shape)
        # print(f'max {mat.max()} min {mat.min()}')
        cm = mat/mat.max()
        # img = img * cm.expand(3, 224, 224).permute(1, 2, 0)
        # print('i done it')
        if torch.isnan(cm.max()) :
            plt.imshow(img, extent=[0, 224, 0, 224])
        else:
            # print('*'*20)
            # print('alpha')
            # print(cm.shape)
            plt.imshow(img, alpha=cm)
        plt.axis('off')
        plt.imshow(cm, alpha=0.5, cmap='hot')
        plt.axis('off')
    else:
        print("high-lighting")
        cm = mat
        w = cm.shape[0]
        max_ele = cm.max()/5
        for i in range(w):
            for j in range(w):
                cm[i, j] = min(max_ele, cm[i, j])
        plt.imshow(cm, cmap='viridis')
        plt.colorbar()
    # plt.title("Attention Map Visualization")
    # plt.xlabel("X-axis")
    # plt.ylabel("Y-axis")
    # plt.show()
"""
   
   
   
#none 
"""
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image

def vis_mat(mat=None, img=None, h=None, w=None, pos=None):
    # plt.rcParams['figure.figsize'] = [44.8, 44.8]#[22.4, 22.4]
    print('into')
    if h is not None:
        plt.rcParams['figure.figsize'] = [w, h]
        plt.subplot(h, w, pos)
        ######
    print('into')
    if img is not None:
        print('into')
        print("pic")
        # print(mat)
        if mat is not None:
            mat = mat.reshape(14, 14)
            assert(mat.shape[0] == 14)
            mat = mat.float()
            mat = nn.functional.interpolate(mat.unsqueeze(0).unsqueeze(0), size=(28, 28), mode="bilinear",
                                            align_corners=True)
            mat = nn.functional.interpolate(mat, size=(224, 224), mode="bilinear", align_corners=True) #
            mat = mat.squeeze(dim=0).squeeze(dim=0)
            cm = mat/(mat.max()+0.00000001)
        plt.imshow(img, extent=[0, 224, 0, 224])
        plt.axis('off')
        if mat is not None:
            plt.imshow(cm, alpha=0.5, cmap='hot')
            plt.axis('off')
    else:
        print("high-lighting")
        cm = mat
        w = cm.shape[0]
        max_ele = cm.max()/5
        for i in range(w):
            for j in range(w):
                cm[i, j] = min(max_ele, cm[i, j])
        plt.imshow(cm, cmap='viridis')
        plt.colorbar()
    # plt.title("Attention Map Visualization")
    # plt.xlabel("X-axis")
    # plt.ylabel("Y-axis")
    # plt.show()

if __name__ == '__main__':
    path = '/data/zhonghanwen_2022/23wb/SSF/ssf_note/SSF/output/confusion_matirx/Epoch50/dogs/True DINOV2 linear_probe adamw 0.000125 img_size-224 using_aug-0 tune_cls-0 cplx_head-0 gp-token seb-114514 -20230530-162205/confusion_res.pth'
    img_path = '/data/datasets/FGVC/stanford_cars/cars_train/05237.jpg'
    cm = torch.load(path)
    img = Image.open(img_path)
    vis_mat(cm)
"""