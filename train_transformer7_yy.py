import argparse
import os
import time
import sys

sys.path.append('/home/drenego/CNN_3D/eye_test/mnist')

#from utee import misc
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np

import dataset
#import dataset3 as dataset3
# import dataset2 as dataset
#import model
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict


import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torch.nn.parallel.data_parallel import data_parallel
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

# from dataset.dataset import *
# from networks.network3 import *
# from networks.network4 import *
# from networks.network5 import *
# from networks.network6 import *
# from networks.network7 import *
from networks.network_res9 import *


# from networks.network8 import *
# from networks.lr_schedule import *
# from metrics.metric import *
# from utils.plot import *
from config import config

import warnings
warnings.filterwarnings("ignore")




# fflag=True  #earlier data
fflag=False  #from jinzhou




# torch.backends.cudnn.enabled = False


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

parser.add_argument('--lr-max', default=0.05, type=float)

parser.add_argument('--wd', type=float, default=0.005, help='weight decay')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 1e-3)')
parser.add_argument('--gpu', default=0, help='index of gpus to use')
parser.add_argument('--ngpu', type=int, default=1, help='number of gpus to use')
parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=100*50,  help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=1,  help='how many epochs to wait before another test')
parser.add_argument('--logdir', default='log/default', help='folder to save to the log')
parser.add_argument('--data_root', default='/tmp/public_dataset/pytorch/', help='folder to save the model')
parser.add_argument('--decreasing_lr', default='80,120', help='decreasing strategy')
args = parser.parse_args()
args.logdir = os.path.join(os.path.dirname(__file__), args.logdir)

# misc.logger.init(args.logdir, 'train_log')
# print = misc.logger.info

# from IPython import embed
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)   #gpu_id
meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
print(meminfo.used/(10**9))
print(meminfo.free/(10**9))



# select gpu
# args.gpu = misc.auto_select_gpu(utility_bound=0, num_gpu=args.ngpu, selected_gpus=args.gpu)
# args.ngpu = len(args.gpu)

# logger
misc.ensure_dir(args.logdir)
print("=================FLAGS==================")
for k, v in args.__dict__.items():
    print('{}: {}'.format(k, v))
print("========================================")

# seed
args.cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# data loader
# train_loader, test_loader = dataset.get(batch_size=args.batch_size, data_root=args.data_root, num_workers=1)

if fflag:
    dataset1=dataset.MyDataset('/home/drenego/CNN_3D/eye_test/mnist/data/x_data_uint8_2_train.npy', '/home/drenego/CNN_3D/eye_test/mnist/data/y_data_uint8_2_train.npy')
else:
    # dataset1=dataset.MyDataset('/home/drenego/CNN_3D/eye_test/mnist/data/data/x_data_uint8_choose_train.npy', '/home/drenego/CNN_3D/eye_test/mnist/data/data/y_data_uint8_choose_train.npy')
    dataset1=dataset.MyDataset('/home/drenego/CNN_3D/eye_test/mnist/data/data_sep/x_div1_all_train.npy','/home/drenego/CNN_3D/eye_test/mnist/data/data_sep/y_div1_all_train.npy')

train_loader= DataLoader(dataset1, batch_size=args.batch_size  , shuffle=True, pin_memory=True)
print(dataset)

if fflag:
    dataset2=dataset.MyDataset('/home/drenego/CNN_3D/eye_test/mnist/data/x_data_uint8_2_test.npy', '/home/drenego/CNN_3D/eye_test/mnist/data/y_data_uint8_2_test.npy')
else:
    # dataset2=dataset.MyDataset('/home/drenego/CNN_3D/eye_test/mnist/data/data/x_data_uint8_choose_test.npy', '/home/drenego/CNN_3D/eye_test/mnist/data/data/y_data_uint8_choose_test.npy')
    dataset2=dataset.MyDataset('/home/drenego/CNN_3D/eye_test/mnist/data/data_sep/x_div1_all_test.npy', '/home/drenego/CNN_3D/eye_test/mnist/data/data_sep/y_div1_all_test.npy')


test_loader= DataLoader(dataset2, batch_size=args.batch_size  , shuffle=True, pin_memory=True)
print(dataset)


aa= np.load('/home/drenego/CNN_3D/eye_test/mnist/data/data_sep/y_div1_all_test.npy',allow_pickle=True) 
# print(aa.shape)
m1=np.abs(aa[0]).max()
m2=np.abs(aa[1]).max()
m3=np.abs(aa[2]).max()

# m4=aa[0]+0.5*aa[1]
print(  [m1,m2,m3]   ) 
mm=torch.Tensor([ m1,m2,m3]).view(1,3)


m1=1.
m2=1.
m3=1.

# model



backbone = models.resnet18(pretrained=False)

# backbone = torch.load(os.path.join('./checkpoints', config.checkpoint))

model = ResNet18(backbone, num_classes=1)
# print(model)

# if not fflag:

#     model = torch.load(os.path.join('./checkpoints', 'ResNet18_S.pth.pth'))

# m = model_zoo.load_url('mnist_model_g_corr.pkl',model_dir='/home/drenego/DeepLearning/pytorch-playground-master/corr_stl/log')
# # m = model_zoo.load_url('stl10_model_g.pkl',model_dir='/home/drenego/DeepLearning/pytorch-playground-master/gan_stl/log')
# state_dict = m.state_dict() if isinstance(m, nn.Module) else m
# assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
# model.load_state_dict(state_dict)



# model = model.mnist(input_dims=784, n_hiddens=[256, 256], n_class=10)
# # model = torch.nn.DataParallel(model, device_ids= range(args.ngpu))
if args.cuda:
    model.cuda()



# if config.freeze:
#     # for p in model.cov1.parameters(): p.requires_grad = False

#     for p in model.backbone.layer1.parameters(): p.requires_grad = False
#     for p in model.backbone.layer2.parameters(): p.requires_grad = False
#     for p in model.backbone.layer3.parameters(): p.requires_grad = False
    # for p in model.backbone.layer4.parameters(): p.requires_grad = False

if fflag:

    criterion =  nn.MSELoss().cuda()
else:

    criterion =  nn.L1Loss().cuda()




# optimizer
# optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
print('decreasing_lr: ' + str(decreasing_lr))
best_acc, old_file = 0, None
t_begin = time.time()

if not os.path.exists('./log'):
    os.makedirs('./log')
log = open('./log/log.txt', 'a')

log.write('-'*30+'\n')
log.write('model:{}\nnum_classes:{}\nnum_epoch:{}\nlearning_rate:{}\nim_width:{}\nim_height:{}\niter_smooth:{}\n'.format(
            config.model, config.num_classes, config.num_epochs, config.lr, 
            config.width, config.height, config.iter_smooth))

# # load checkpoint
# if config.resume:
#     model = torch.load(os.path.join('./checkpoints', 'ResNet18_S.pth'))

# train
sum = 0
train_loss_sum = 0
train_top1_sum = 0
max_val_acc = 0
train_draw_acc = []
val_draw_acc = []
precision_50=0

if fflag:
    jj=[4,5,10,11,16,17]
else:
    jj=[4,5,8,9,10,11]


# lr_schedule = lambda t: np.interp([t], [0, args.epochs*2//5, args.epochs*4//5, args.epochs], 
                                #   [0, args.lr_max, args.lr_max/20.0, 0])[0]

# optimizer = optim.AdamW(model.parameters(), lr=args.lr_max, weight_decay=args.wd)



try:
    # ready to go
    for epoch in range(args.epochs): 
        ep_start = time.time()

        lr = step_lr(epoch)


        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                     lr=lr, betas=(0.9, 0.999), weight_decay=0.0002)

        model.train()
        top1_sum = 0

        train_loss_sum=0
        train_loss_sum1=0
        iii=0
        abs_errors=torch.zeros(1,1)

        if epoch in decreasing_lr:
            optimizer.param_groups[0]['lr'] *= 0.1
        for batch_idx, (data, target) in enumerate(train_loader):

            # for jj in [4,5,10,11,16,17]:

            indx_target = target.clone().data.cuda()
            # lr = lr_schedule(epoch + (batch_idx + 1)/len(train_loader))
            # optimizer.param_groups[0].update(lr=lr)


            # print(data)
            data=data[:,jj]
            # target=target/mm
            # cv2.imshow('img',data[0,0].data.cpu().numpy())
            # cv2.waitKey(1000)
            # target=target[:,0]+1*target[:,1]
            target= (  target[:,0].reshape(target.shape[0],1) ).detach()/m1
            # print(target)

            # print( data.shape  )
            # print( target.shape  )
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)


            output = model(data)

            # output = torch.round( ( model(data)  +1  )/0.125 )*0.125
            # torch.round( ( output  +1  )/0.125 )*0.125
             # print( output.shape)
            # print(  target.shape)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # top1 = accuracy(output.data, target.data, topk=(1,))
            train_loss_sum1 += loss.data.cpu().numpy()
            train_loss_sum +=(torch.abs(output-target) **2).sum().data.cpu().numpy()
            # print( train_loss_sum1 )
            # print( train_loss_sum//args.batch_size )
            iii+=data.shape[0]
            sum += 1
                # top1_sum += top1[0]
            # print(sum*args.batch_size)
            # print(len(train_loader))
            # abs_errors=torch.cat( [abs_errors, torch.abs(output-target).cpu().data] ,0)
            abs_errors=torch.cat( [abs_errors, torch.abs(output-indx_target[:,1].reshape(-1,1)).cpu().data] ,0)

            if (batch_idx+1) % config.iter_smooth == 0:
                print('Epoch [%d/%d], Iter [%d/%d], lr: %f, Loss: %.4f,Loss_mse: %.4f'
                       %(epoch, config.num_epochs, batch_idx+1, len(train_loader)//args.batch_size, 
                       lr, train_loss_sum/sum/args.batch_size*m1,train_loss_sum1/sum *m1))
                log.write('Epoch [%d/%d], Iter [%d/%d], lr: %f, Loss: %.4f,Loss_mse: %.4f\n'
                           %(epoch, config.num_epochs, batch_idx+1, len(train_loader)//args.batch_size, 
                           lr, train_loss_sum/sum/args.batch_size*m1, train_loss_sum1/sum*m1))
                sum = 0
                # iii=0
                # train_loss_sum = 0


        abs_errors=abs_errors[1:].numpy()*m1

        # print(abs_errors)
        # print(len( abs_errors))
        m=abs_errors.shape[0]
        # print( abs_errors.shape   )
        # print(len(test_loader) )
        # print(len(train_loader) )
        # print(iii)

        # abs_errors=np.array(abs_errors)
        count_leq350_25 = 0
        count_leq350_50 = 0
        count_leq350_100 = 0
        for i in range(m ):
            if abs_errors[i,0] <= 1:
                count_leq350_100 += 1
            if abs_errors[i,0] <= 0.5:
                count_leq350_50 += 1
            if abs_errors[i,0] <= 0.25:
                count_leq350_25 += 1

        precision_25 = count_leq350_25 / m
        precision_50 = count_leq350_50 / m
        precision_100 = count_leq350_100 / m

        # print('precision_25: ', precision_25, 'precision_50: ', precision_50, 'precision_100: ', precision_100)

        print( 'Train_predict_25 %.4f, 50: %.4f,  100: %.4f '
                %(precision_25, precision_50, precision_100) )



        # train_draw_acc.append(train_loss_sum/len(train_loader))
        train_draw_acc.append(train_loss_sum/iii*m1)
        
        epoch_time = (time.time() - ep_start) / 60.
        # print( train_loss_sum1 /len(train_loader)*8)
        # print( train_loss_sum /len(train_loader))
        # print(iii)



        #misc.model_snapshot(model, os.path.join(args.logdir, 'latest.pth'))

        if epoch % args.test_interval == 0:
            model.eval()
            test_loss = 0
            correct = 0
            abs_errors=torch.zeros(1,1)
            abs_errors2=torch.zeros(1,1)
            jjj=0

            for data, target in test_loader:
                indx_target = target.clone().data.cuda()


                data=data[:,jj]
                # target=target/mm
                
                abs_errors2=torch.cat( [abs_errors2, torch.abs(target[:,1]+0.5*target[:,2] ).cpu().data.reshape(target.shape[0],1)] ,0)

                # target=target/mm
                # target=target[:,0]+1*target[:,1]
                # target=target[:,0].reshape(target.shape[0],1)
                target= (  target[:,0].reshape(target.shape[0],1) ).detach()/m1

                if args.cuda:
                    data, target = data.cuda(), target.cuda()



                # for jj in [4,5,10,11,16,17]:

                # data=data[:,4,].reshape(data.shape[0],1,data.shape[2],data.shape[3]).repeat(1,3,1,1 )


                data, target = Variable(data, volatile=True), Variable(target)

                output = model(data)
                # output = torch.round( ( model(data)  +1  )/0.125 )*0.125


                loss = criterion(output, target)
                # test_loss += loss.data.cpu().numpy()
                test_loss+= torch.abs(output-target ).sum().data.cpu().numpy()

                # abs_errors=torch.cat( [abs_errors, torch.abs(output-target).cpu().data*mm] ,0)

                
                # abs_errors=torch.cat( [abs_errors, torch.abs(output-target).cpu().data] ,0)
                abs_errors=torch.cat( [abs_errors, torch.abs(output-indx_target[:,1].reshape(-1,1)).cpu().data] ,0)





                jjj+=data.shape[0]
                # print(data.shape)
                # correct += pred.cpu().eq(indx_target).sum()


            # eval
            val_time_start = time.time()
            # val_loss, val_top1 = eval(model, dataloader_valid, criterion)
            val_draw_acc.append(test_loss/ jjj*m1)
            val_time = (time.time() - val_time_start) / 60.

            print('Epoch [%d/%d], Val_Loss: %.4f,  Train_Loss: %.4f,val_time: %.4f s'
                   %(epoch+1, config.num_epochs, test_loss/ jjj*m1,train_loss_sum/iii*m1, val_time*60))
            print('epoch time: {}s'.format(epoch_time*60))

            # print( abs_errors2 )

            abs_errors=abs_errors[1:].numpy()*m1

            # print(abs_errors)
            # print(len( abs_errors))
            m=abs_errors.shape[0]
            # print( abs_errors.shape   )
            # print(len(test_loader) )
            # print(len(train_loader) )
            # print(iii)



            # abs_errors=np.array(abs_errors)
            count_leq350_25 = 0
            count_leq350_50 = 0
            count_leq350_55 = 0
            count_leq350_60 = 0
            count_leq350_65 = 0
            count_leq350_70 = 0
            count_leq350_75 = 0
            count_leq350_80 = 0
            count_leq350_85 = 0
            count_leq350_90 = 0
            count_leq350_95 = 0
            count_leq350_100 = 0
            
            count_leq350=0
            
            for i in range(m ):
                if abs_errors2[i,0]<3.5:
                    count_leq350 +=1
                        
                    if abs_errors[i,0] <= 1:
                        count_leq350_100 += 1
                    if abs_errors[i,0] <= 0.95:
                        count_leq350_95 += 1
                    if abs_errors[i,0] <= 0.9:
                        count_leq350_90 += 1
                    if abs_errors[i,0] <= 0.85:
                        count_leq350_85 += 1
                    if abs_errors[i,0] <= 0.8:
                        count_leq350_80 += 1
                    if abs_errors[i,0] <= 0.75:
                        count_leq350_75 += 1
                    if abs_errors[i,0] <= 0.7:
                        count_leq350_70 += 1
                    if abs_errors[i,0] <= 0.65:
                        count_leq350_65 += 1
                    if abs_errors[i,0] <= 0.6:
                        count_leq350_60 += 1
                    if abs_errors[i,0] <= 0.55:
                        count_leq350_55 += 1

                    if abs_errors[i,0] <= 0.5:
                        count_leq350_50 += 1
                    if abs_errors[i,0] <= 0.25:
                        count_leq350_25 += 1

            precision_leq350_25 = count_leq350_25 / count_leq350+1e-6
            precision_leq350_50 = count_leq350_50 / count_leq350+1e-6

            precision_leq350_55  = count_leq350_55   / count_leq350+1e-6
            precision_leq350_60  = count_leq350_60   / count_leq350+1e-6
            precision_leq350_65  = count_leq350_65   / count_leq350+1e-6
            precision_leq350_70  = count_leq350_70   / count_leq350+1e-6
            precision_leq350_75  = count_leq350_75   / count_leq350+1e-6
            precision_leq350_80  = count_leq350_80   / count_leq350+1e-6
            precision_leq350_85  = count_leq350_85   / count_leq350+1e-6
            precision_leq350_90  = count_leq350_90   / count_leq350+1e-6
            precision_leq350_95  = count_leq350_95   / count_leq350+1e-6


            precision_leq350_100 = count_leq350_100 / count_leq350+1e-6

            # print('precision_25: ', precision_25, 'precision_50: ', precision_50, 'precision_100: ', precision_100)

            print( 'Test_predict_Leq350__!!!!50 %.4f, 55: %.4f, 60: %.4f, 65: %.4f,70: %.4f, 75: %.4f, 80: %.4f, 85: %.4f, 90: %.4f, 95: %.4f,  100: %.4f , count: %.4f'
                   %(precision_leq350_50, precision_leq350_55, precision_leq350_60, precision_leq350_65, \
                     precision_leq350_70, precision_leq350_75, precision_leq350_80, precision_leq350_85, \
                        precision_leq350_90, precision_leq350_95, precision_leq350_100,count_leq350) )
            


            count_geq350_25 = 0
            count_geq350_50 = 0

            count_geq350_50 = 0
            count_geq350_55 = 0
            count_geq350_60 = 0
            count_geq350_65 = 0
            count_geq350_70 = 0
            count_geq350_75 = 0
            count_geq350_80 = 0
            count_geq350_85 = 0
            count_geq350_90 = 0
            count_geq350_95 = 0

            count_geq350_100 = 0
            
            count_geq350=0
            
            for i in range(m ):
                if abs_errors2[i,0]>=3.5:
                    count_geq350+=1
                        
                    if abs_errors[i,0] <= 1:
                        count_geq350_100 += 1


                    if abs_errors[i,0] <= 0.95:
                        count_geq350_95 += 1
                    if abs_errors[i,0] <= 0.9:
                        count_geq350_90 += 1
                    if abs_errors[i,0] <= 0.85:
                        count_geq350_85 += 1
                    if abs_errors[i,0] <= 0.8:
                        count_geq350_80 += 1
                    if abs_errors[i,0] <= 0.75:
                        count_geq350_75 += 1
                    if abs_errors[i,0] <= 0.7:
                        count_geq350_70 += 1
                    if abs_errors[i,0] <= 0.65:
                        count_geq350_65 += 1
                    if abs_errors[i,0] <= 0.6:
                        count_geq350_60 += 1
                    if abs_errors[i,0] <= 0.55:
                        count_geq350_55 += 1




                    if abs_errors[i,0] <= 0.5:
                        count_geq350_50 += 1
                    if abs_errors[i,0] <= 0.25:
                        count_geq350_25 += 1

            precision_geq350_25 = count_geq350_25 / count_geq350+1e-6
            precision_geq350_50 = count_geq350_50 / count_geq350+1e-6



            precision_geq350_55  = count_geq350_55   / count_geq350+1e-6
            precision_geq350_60  = count_geq350_60   / count_geq350+1e-6
            precision_geq350_65  = count_geq350_65   / count_geq350+1e-6
            precision_geq350_70  = count_geq350_70   / count_geq350+1e-6
            precision_geq350_75  = count_geq350_75   / count_geq350+1e-6
            precision_geq350_80  = count_geq350_80   / count_geq350+1e-6
            precision_geq350_85  = count_geq350_85   / count_geq350+1e-6
            precision_geq350_90  = count_geq350_90   / count_geq350+1e-6
            precision_geq350_95  = count_geq350_95   / count_geq350+1e-6


            precision_geq350_100 = count_geq350_100 / count_geq350+1e-6

            # print('precision_25: ', precision_25, 'precision_50: ', precision_50, 'precision_100: ', precision_100)

            # print( 'Test_predict_geq350__!!!!50 %.4f, 50: %.4f, 100: %.4f , count: %.4f  '
            #        %(precision_geq350_50, precision_geq350_50, precision_geq350_100,count_geq350 ) )
            
            print( 'Test_predict_Geq350__!!!!50 %.4f, 55: %.4f, 60: %.4f, 65: %.4f, 70: %.4f, 75: %.4f, 80: %.4f, 85: %.4f, 90: %.4f, 95: %.4f,  100: %.4f , count: %.4f'
                   %(precision_geq350_50, precision_geq350_55, precision_geq350_60, precision_geq350_65, \
                     precision_geq350_70, precision_geq350_75, precision_geq350_80, precision_geq350_85, \
                        precision_geq350_90, precision_geq350_95, precision_geq350_100,count_geq350) )
            



            if precision_leq350_50 > best_acc:
                # new_file = os.path.join(args.logdir, 'best-{}.pth'.format(epoch))
                # misc.model_snapshot(model, new_file, old_file=old_file, verbose=True)
                best_acc = precision_leq350_50
                # old_file = new_file
                print('Taking snapshot...')
                if not os.path.exists('./checkpoints'):
                    os.makedirs('./checkpoints')
                if fflag:
                    torch.save(model, '{}/{}.pth'.format('checkpoints','ResNet18_S.pth'))
                else:
                    torch.save(model, '{}/{}.pth'.format('checkpoints','ResNet18_S_new_div1_all.pth'))



            # if val_top1[0].data > max_val_acc:
            #     max_val_acc = val_top1[0].data
            #     print('Taking snapshot...')
            #     if not os.path.exists('./checkpoints'):
            #         os.makedirs('./checkpoints')
            #     torch.save(model, '{}/{}.pth'.format('checkpoints', config.model))

            log.write('Epoch [%d/%d], Val_Loss: %.4f, val_time: %.4f s\n'
                       %(epoch+1, config.num_epochs, test_loss/ jjj*m1, val_time*60))
        draw_curve(train_draw_acc, val_draw_acc)


    log.write('-'*30+'\n')
    log.close()




except Exception as e:
    import traceback
    traceback.print_exc()
finally:
    print("Total Elapse: {:.2f}, Best Result: {:.3f}%".format(time.time()-t_begin, best_acc))


