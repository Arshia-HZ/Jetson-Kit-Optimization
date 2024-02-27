from __future__ import print_function, division
import argparse
import numpy as np
import cv2
import os
# from utils import all_sample_iou, plot_success_curve
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import copy
import matplotlib
import matplotlib.pyplot as plt
# from DIM import *
from collections import OrderedDict
import numpy as np
import imutils
import cv2
import time
from operator import itemgetter
from baseFewShotMatcher import BaseFewShotMatcher
import math

# DIM .................................................................
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
def odd(f):
    return int(np.ceil(f)) // 2 * 2 + 1
 
def imcrop_odd(I,box,indxtrans=False):
    b,c,w,h=I.shape
    if indxtrans==True:
        box=(round(box[0])-1,round(box[1])-1,odd(box[2]),odd(box[3]))
    else:
        box=(round(box[0]),round(box[1]),odd(box[2]),odd(box[3]))
    boxInbounds=(min(h-odd(box[2]),max(0,round(box[0]))),min(w-odd(box[3]),max(0,round(box[1]))),
                 odd(box[2]),odd(box[3]))
    box=boxInbounds
    Ipatch=I[:,:,box[1]:box[1]+box[3],box[0]:box[0]+box[2]]
    return Ipatch,box



def ellipse(alen,blen):
    x,y=np.mgrid[-alen:alen+1,-blen:blen+1]
    s=np.zeros_like(x)
    tem=(x**2/alen**2)+(y**2/blen**2)
    s[tem<=1]=1
    return s
    
def extract_additionaltemplates(image,template,numadditionaltemplates,keypoints):
    
    template=torch.flip(template,[2, 3])
    similarity=conv2_same(image,template).squeeze().cpu().numpy()
    
    n,c,x,y=image.shape
    n,c,h,w=template.shape
    
    ind=np.argsort(similarity,axis=None)[::-1]
    keypointsCandidates_row= np.array((np.unravel_index(ind, similarity.shape))).transpose(1,0)
    df=pd.DataFrame(keypointsCandidates_row,columns=list('AB'))
    keypointsCandidates=df[(df['A']+1>np.ceil(h/2)) & (df['A']<x-np.ceil(h/2)) & 
                           (df['B']+1>np.ceil(w/2)) & (df['B']<y-np.ceil(w/2))].values.astype('float32')
    keypointCandidatesAccepted=[]
    addtemplates=[]
    numAccepted=0
    for i in range(len(keypointsCandidates)):
        if numadditionaltemplates==numAccepted:
            break
        if any(np.logical_and((abs(keypointsCandidates[i][0]-keypoints[:,0])<h),(abs(keypointsCandidates[i][1]-keypoints[:,1])<w))):
            skip=1
        else:
            keypointCandidatesAccepted.append(keypointsCandidates[i,:])
            keypoints = np.vstack((keypoints,keypointsCandidates[i,:]))
            numAccepted+=1
            addtemplate,box=imcrop_odd(image,(int(keypointsCandidates[i,1]-(w-1)/2),
                              int(keypointsCandidates[i,0]-(h-1)/2),w,h))
            addtemplates.append(addtemplate)
    if len(addtemplates):
        addtemplates=torch.cat(addtemplates,0)
    return addtemplates

def preprocess(image):
    '''The preprocess described in original DIM paper'''
    cuda=torch.cuda.current_device()
    n, c, h, w = image.size()
    X=torch.zeros(n,2*c,h,w,device=cuda)
    for i in range(len(image)):
        for j in range(len(image[i])):
            X[i][2*(j-1)+2]=torch.clamp(image[i][j],min=0)
            X[i][2*(j-1)+3]=torch.clamp(image[i][j],max=0).abs()    
    '''    
    Alternatively, you can use the fellowing code which run faster  
    imageon=torch.clamp(image,min=0)
    imageoff=torch.clamp(image,max=0).abs()  
    out=torch.cat((imageon,imageoff),1)
    return out
    '''
    return X

def conv2_same(Input, weight,num=1):
    padding_rows = weight.size(2)-1
    padding_cols = weight.size(3)-1  
    rows_odd = (padding_rows % 2 != 0)
    cols_odd = (padding_cols % 2 != 0)
    if rows_odd or cols_odd:
        Input = F.pad(Input, [0, int(cols_odd), 0, int(rows_odd)])
    weight=torch.flip(weight,[2, 3])
    return F.conv2d(Input, weight, padding=(padding_rows // 2, padding_cols // 2), groups=num)


def DIM_matching(X,w,iterations):
    cuda=torch.cuda.current_device()
    v=torch.zeros_like(w)
    Y=torch.zeros(X.shape[0],len(w),X.shape[2],X.shape[3],device=cuda)
    tem1=w.clone()
    
    for i in range(len(w)):
        v[i]=torch.max(torch.tensor(0, dtype=torch.float32,device=cuda),
        w[i]/torch.max(torch.tensor(1e-6, dtype=torch.float32,device=cuda),torch.max(w[i])))
        tem1[i]=w[i]/torch.max(torch.tensor(1e-6,dtype=torch.float32,device=cuda),torch.sum(w[i]))
    w=torch.flip(tem1,[2, 3])
    sumV=torch.sum(torch.sum(torch.sum(v,0),1),1)
    epsilon2=1e-2
    epsilon1=torch.tensor(epsilon2,dtype=torch.float32,device=cuda)/torch.max(sumV)
    for count in range(iterations):
        R=torch.zeros_like(X)
        if not torch.sum(Y)==0:
            R=conv2_same(Y,v.permute(1,0,2,3))
            R=torch.clamp(R, min=0)
        E=X/torch.max(torch.tensor(epsilon2,dtype=torch.float32,device=cuda),R)
        Input=torch.zeros_like(E)
        Input=conv2_same(E,w)
        tem2=Y.clone()
        for i in range(len(Input)):
            for j in range(len(Input[i])):
                tem2[i][j]=Input[i][j]*torch.max(epsilon1,Y[i][j])
        Y=torch.clamp(tem2, min=0)
    Y=Y[:,0,:,:].squeeze(0).cpu().numpy()
    return Y

# Utils ..................................................
# from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
def IoU( r1, r2 ):
    x11, y11, w1, h1 = r1
    x21, y21, w2, h2 = r2
    x12 = x11 + w1-1; y12 = y11 + h1-1
    x22 = x21 + w2-1; y22 = y21 + h2-1
    x_overlap = max(0, min(x12,x22) - max(x11,x21) )
    y_overlap = max(0, min(y12,y22) - max(y11,y21) )
    I = 1. * x_overlap * y_overlap
    U = (y12-y11)*(x12-x11) + (y22-y21)*(x22-x21) - I
    J = I/U
    return J


def score2curve( score, thres_delta = 0.01 ):
    thres = np.linspace( 0, 1, int(1./thres_delta)+1 )
    success_num = []
    for th in thres:
        success_num.append( np.sum(score >= (th+1e-6)) )
    success_rate = np.array(success_num) / len(score)
    return thres, success_rate

def all_sample_iou( gt_list,pd_list):
    num_samples = len(gt_list)
    iou_list = []
    for idx in range(num_samples):
        image_gt,image_pd = gt_list[idx],pd_list[idx]
        iou = IoU( image_gt, image_pd )
        iou_list.append( iou )
    return iou_list


def plot_success_curve( iou_score,method,title='' ):
    imageroot='results'+'/{m}/{n}.png'
    thres, success_rate = score2curve( iou_score, thres_delta = 0.05 )
    auc_ = np.mean( success_rate[:-1] ) # this is same auc protocol as used in previous template matching papers #auc_ = auc( thres, success_rate ) # this is the actual auc
    plt.figure()
    plt.grid(True)
    plt.xticks(np.linspace(0,1,11))
    plt.yticks(np.linspace(0,1,11))
    plt.ylim(0, 1)
    plt.title(title + 'auc={}'.format(auc_))
    plt.plot( thres, success_rate )
    plt.savefig(imageroot.format(n='AUC',m=method))
    plt.show()

# Deep-DIM .................................................................
matplotlib.use('Agg')
class Featex():
    def __init__(self, model, use_cuda,layer1,layer2,layer3):
        self.use_cuda = use_cuda
        self.feature1 = None
        self.feature2 = None
        self.feature3 = None
        self.U1=None
        self.U2=None
        self.U3=None
        self.model= copy.deepcopy(model.eval())
        self.model = self.model[:36]
        for param in self.model.parameters():
            param.requires_grad = False
        if self.use_cuda:
            self.model = self.model.cuda()
        self.model[layer1].register_forward_hook(self.save_feature1)
        self.model[layer1+1]=torch.nn.ReLU(inplace=False)
        self.model[layer2].register_forward_hook(self.save_feature2)
        self.model[layer2+1]=torch.nn.ReLU(inplace=False)
        self.model[layer3].register_forward_hook(self.save_feature3)
        self.model[layer3+1]=torch.nn.ReLU(inplace=False)
        
    def save_feature1(self, module, input, output):
        self.feature1 = output.detach()

    def save_feature2(self, module, input, output):
        self.feature2 = output.detach()
    
    def save_feature3(self, module, input, output):
        self.feature3 = output.detach()
    def __call__(self, input, mode='normal'):
        channel=64
        if self.use_cuda:
            input = input.cuda()
        _ = self.model(input)
        
        if channel<self.feature1.shape[1]:
            reducefeature1,self.U1=runpca(self.feature1,channel,self.U1)
        else:
            reducefeature1=self.feature1
        if channel<self.feature2.shape[1]:
            reducefeature2,self.U2=runpca(self.feature2,channel,self.U2)
        else:
            reducefeature2=self.feature2
        if channel<self.feature3.shape[1]:
            reducefeature3,self.U3=runpca(self.feature3,channel,self.U3)
        else:
            reducefeature3=self.feature3 
            
        if mode=='big':
            # resize feature1 to the same size of feature2 
            reducefeature1 = F.interpolate(reducefeature1, size=(self.feature3.size()[2], self.feature3.size()[3]), mode='bilinear', align_corners=True)
            reducefeature2 = F.interpolate(reducefeature2, size=(self.feature3.size()[2], self.feature3.size()[3]), mode='bilinear', align_corners=True)
        else:        
            reducefeature2 = F.interpolate(reducefeature2, size=(self.feature1.size()[2], self.feature1.size()[3]), mode='bilinear', align_corners=True)
            reducefeature3 = F.interpolate(reducefeature3, size=(self.feature1.size()[2], self.feature1.size()[3]), mode='bilinear', align_corners=True)
        return torch.cat((reducefeature1, reducefeature2,reducefeature3), dim=1)

def runpca(x,components,U):
    whb=x.squeeze(0).permute(1,2,0).cpu().numpy()
    shape=whb.shape
    raw=whb.reshape((shape[0] * shape[1],shape[2]))
    X_norm,mu,sigma = featureNormalize(raw)
    if U is None:
        Sigma = np.dot(np.transpose(X_norm),X_norm)/raw.shape[0]
        U,S,V = np.linalg.svd(Sigma)
    Z = projectData(X_norm,U,components)
    return torch.tensor(Z.reshape((shape[0], shape[1],components))).permute(2,0,1).unsqueeze(0).cuda(),U

def featureNormalize(X):
    n = X.shape[1]
    
    sigma = np.zeros((1,n))
    mu = np.zeros((1,n))
    mu = np.mean(X,axis=0)  
    sigma = np.std(X,axis=0)
    for i in range(n):
        X[:,i] = (X[:,i]-mu[i])/sigma[i]
    return X,mu,sigma

def projectData(X_norm,U,K):
    Z = np.zeros((X_norm.shape[0],K))
    
    U_reduce = U[:,0:K]          
    Z = np.dot(X_norm,U_reduce) 
    return Z

def read_gt( file_path ):
    with open( file_path ) as IN:
        x, y, w, h = [ eval(i) for i in IN.readline().strip().split(',')]
    return x, y, w, h

def apply_DIM(I_row,SI_row,template_bbox,pad,pad1,image,numaddtemplates):
    I=preprocess(I_row)
    SI=preprocess(SI_row)
    template = I
    template,oddTbox=imcrop_odd(I,template_bbox,True)
    targetKeypoints=[oddTbox[1]+(oddTbox[3]-1)/2,oddTbox[0]+(oddTbox[2]-1)/2]
    addtemplates=extract_additionaltemplates(I,template,numaddtemplates,np.array([targetKeypoints]))
    if len(addtemplates):
        if template.shape[-1] == 1 and template.shape[-2] == 1:  # ---------- sara added 
            templates=torch.cat((template,addtemplates),0)
        else:
            templates=template    
    else:
        templates=template
#     templates=template
#     similarity=DIM_matching(SI,templates,10)[pad[0]:pad[0]+I.shape[2],pad[1]:pad[1]+I.shape[3]]
    similarity=DIM_matching(SI,templates,10)[pad[0]:-pad[0],pad[1]:-pad[1]]

    #post processing
    similarity = cv2.resize( similarity, (image.shape[1], image.shape[0]) )
    scale=0.025 
    region=torch.from_numpy(ellipse(round(max(1,scale*pad1[1]))
    ,round(max(1,scale*pad1[0])))).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
    similarity=conv2_same(torch.from_numpy(similarity).unsqueeze(0).unsqueeze(0)
    ,region).squeeze().numpy()
    return similarity
from PIL import Image 
import PIL

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor.cpu(), dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def model_eval(featex, temp, tar, image_transform, a, b,  x1, y1, x2, y2):
        # load image and ground truth
        template_raw = temp
        h = temp.shape[0]
        w = temp.shape[1]
        template_bbox = [x1, y1, x2-x1, y2-y1] #[0, 0, int(w)-1, int(h)-1]

        image = tar
       
        T_image=image_transform(template_raw).unsqueeze(0)

        T_search_image=image_transform(image).unsqueeze(0)
       
        if w*h <= 4000:
            I_feat=featex(T_image)
            SI_feat=featex(T_search_image)
            resize_bbox=[i/a for i in template_bbox]
        else:
            I_feat=featex(T_image,'big')
            SI_feat=featex(T_search_image,'big')
            resize_bbox=[i/b for i in template_bbox]
        pad1=[int(round(t)) for t in (template_bbox[3],template_bbox[2])]
        pad2=[int(round(t)) for t in (resize_bbox[3],resize_bbox[2])]

        SI_pad=torch.from_numpy(np.pad(SI_feat.cpu().numpy(),((0,0),(0,0),
                              (pad2[0],pad2[0]),(pad2[1],pad2[1])),'symmetric'))
        similarity=apply_DIM(I_feat,SI_pad,resize_bbox,pad2,pad1,image,0) #/.............. 
        ptx,pty=np.where(similarity == np.amax(similarity))
#         print("ptx, pty = ", ptx, pty, np.amax(similarity))
        image_pd=tuple([pty[0]+1-(odd(template_bbox[2])-1)/2,ptx[0]+1-(odd(template_bbox[3])-1)/2,
              template_bbox[2],template_bbox[3]])
        for t in range(len(ptx)):
            img_d = cv2.circle(tar, (pty[t],ptx[t]), 5, (255,0,0), 2)

#         cv2.imwrite('tar_detect.png', img_d)
#         cv2.imwrite('feature.png', similarity*100)
        return (np.amax(similarity), image_pd)
            
class DeepDIM(BaseFewShotMatcher):
    def __init__(self):
        super().__init__()
        self.image_transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225],
              )
           ])
        
        checkpoint=torch.load('trainModel/pretrain_models/model_D.pth.tar', map_location=lambda storage, loc: storage)
        state_dict =checkpoint['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove 'module.' of dataparallel
            new_state_dict[name]=v
        
#         layers=(0,2,5,7,10,12,14,16,19,21,23,25,28,30,32,34)
        layers=(2,19,25)
#         layers=(0,16,21)

        layer1=layers[0]
        layer2=layers[1]
        layer3=layers[2]
        
        if 0<=layer1<=4:
            a=1
        if 4<layer1<=9:
            a=2
        if 9<layer1<=18:
            a=4
        if 18<layer1<=27:
            a=8
        if 27<layer1<=36:
            a=16
        if 0<=layer3<=4:
            b=1
        if 4<layer3<=9:
            b=2
        if 9<layer3<=18:
            b=4
        if 18<layer3<=27:
            b=8
        if 27<layer3<=36:
            b=16
        self.a = a
        self.b = b
        
        model=models.vgg19(pretrained=False)
        model.load_state_dict(new_state_dict)
        model=model.features
        
        self.featex=Featex(model, True, layer1, layer2, layer3)    
        
            
    def predict(self, target, templates, temps_box):
        predictions = []
#         scales = np.linspace(0.8, 1.2, 9)
#         scales = [0.3, 0.45, 0.6, 0.75, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
        scales = [1.0]
#         cv2.imwrite('target.png', target)
        for i, tem in enumerate(zip(templates, temps_box)): 
            temp, temp_box = tem
#             image_plot = cv2.rectangle( temp, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2 )
#             cv2.imwrite('temp.png', temp)
            
            for scale in scales: 
                x1,y1,x2,y2 = temp_box
                start = time.time()
                if scale == 1:
                    resized = target
                    resized_t = temp
                elif scale < 1:
                    resized = imutils.resize(target, width=int(target.shape[1]*scale), height=int(target.shape[0]*scale))
                    resized_t = temp
                else:
                    resized_t = imutils.resize(temp, width=int(temp.shape[1]/scale), height=int(temp.shape[0]/scale))
                    x1,y1,x2,y2 = int(x1/scale),int(y1/scale),int(x2/scale),int(y2/scale)
                    resized = target

                r = target.shape[1] / float(resized.shape[1])
                
#                 h = resized_t.shape[0]
#                 w = resized_t.shape[1] 
                
                h = int(y2-y1)
                w = int(x2-x1)
                
                if resized.shape[0] < h or resized.shape[1] < w:
                    end = time.time()
                    timePred = float("{:.3f}".format(end - start))
                    x1, y1, x2, y2 = -1, -1, -1, -1
                    print('resized.shape == ', resized.shape, h, w)
                    continue
                
                if w*h <300:
                    end = time.time()
                    timePred = float("{:.3f}".format(end - start))
                    x1, y1, x2, y2 = -1, -1, -1, -1
                    print('w*h == ', w*h, h, w)
                    continue    
                similarity, pd = model_eval(self.featex, resized_t, resized, self.image_transform,  self.a , self.b , x1, y1, x2, y2)
                if scale > 1:
                    similarity /= (scale ** 0.5)
                x1, y1, w, h = pd
                x2, y2 = x1 + w, y1 + h
#                 print('bef r = ', r, x1, y1,(x1+w), (y1+h))
                x1 = int(x1 * r)
                y1 = int(y1 * r)
                x2 = int((x1+w) * r)
                y2 = int((y1+h) * r)
#                 print('aff r = ', r, x1, y1, x2, y2)
                end = time.time()
                timePred = float("{:.3f}".format(end - start))
                predictions.append([x1, y1, x2, y2,
                    float("{:.3f}".format(similarity)), timePred])
            
        if len(predictions) == 0:
            predictions.append([x1, y1, x2, y2,
                0, timePred])
        
        predictions.sort( reverse = True, key = itemgetter(4))

#         if(len(templates) == 10):
#         if True:
#             while True:
#                 root='results/{m}.png'
# #                 print(predictions[0][:4])
#                 x1, y1, x2, y2 = predictions[0][:4]
#                 tar = target.copy()
#                 image_plot = cv2.rectangle( tar, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2 )
#                 cv2.imwrite(root.format(m=predictions[0][4]), tar)

#                 break
                
        # TODO : add a parameter for get max of predictions data e.g. in this case it is 10.
#         with torch.no_grad():
#         torch.cuda.empty_cache()
        return predictions[:10]