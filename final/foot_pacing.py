import cv2     # for capturing videos
import os
import matplotlib.pyplot as plt    # for plotting the images
import math
from sklearn import preprocessing
import numpy as np    # for mathematical operations
import pickle
import torch
import random
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import sys
import copy
from scipy.special import softmax
from torchvision import models
from sklearn.metrics import classification_report

from Resnet3d import resnet34,my_resnet,resnet18
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules.normalization import LayerNorm

import logging
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())
fh = logging.FileHandler('./logging_foot_pacing.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
fh.setFormatter(formatter)
fh.setLevel(logging.INFO) # or any level you want
logger.addHandler(fh)
logger.setLevel(logging.INFO) # or any level you want


# Turn videos to frames
def get_frames(predict_videoFile = None):
    logger.info('Getting frames...')

    if not predict_videoFile:
        for (dirpath, dirnames, filenames) in os.walk('./pacing/marked_only_videos/'):
            for file in filenames:
                logger.info(file)
                file_path = os.path.join(dirpath,file)
                count = 0
                cap = cv2.VideoCapture(file_path)   # capturing the video from the given path
                frameRate = cap.get(5) #frame rate
                logger.info('frame rate:'+str(frameRate))
                x=1
                frames = []
                while(cap.isOpened()):
                    frameId = cap.get(1) #current frame number
                    ret, frame = cap.read()
                    # logger.info(ret,frame)
                    if (ret != True):
                        break
                    count+=1
                    # if (frameId % math.floor(frameRate) == 0):
                    # logger.info(count)
                    framename = "frame%d.jpg" % count
                    out_dir = os.path.join('./pacing/marked_only_images/', file)
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    cv2.imwrite(os.path.join(out_dir,framename), frame)
                cap.release()
        logger.info ("Done!")
        return None

    else:
        count = 0
        cap = cv2.VideoCapture(predict_videoFile)   # capturing the video from the given path
        frameRate = cap.get(5) #frame rate
        logger.info('frame rate:'+str(frameRate))
        x=1
        frames = []
        while(cap.isOpened()):
            frameId = cap.get(1) #current frame number
            ret, frame = cap.read()
            # logger.info(ret,frame)
            if (ret != True):
                break
            count+=1
            # logger.info(count)
            frames.append(frame)
            # plt.imshow(frame)
            # plt.close()
        cap.release()
        logger.info ("Done!")
        return frames,frameRate


# preprocess the data and annotate each frame.
def prepare_data(predic_frames=None):
    logger.info('Preparing data...')
    if not predic_frames:
        for (dirpath, dirnames, filenames) in os.walk('./pacing/marked_only_videos/'):
            for file in filenames:
                logger.info(file)
                file_path = os.path.join(dirpath, file)
                frames, frame_rate = get_frames(file_path)
                X = []
                for count, img in enumerate(frames):
                    if img.shape != (480, 272, 3):
                        # logger.info(file,img.shape)
                        img = cv2.resize(img, (272, 480))
                        # print(type(img),img)
                        # logger.info(img.shape)
                        # cv2.imwrite('./test.jpg', img)
                        X.append(img)  # storing each image in array X

                X = np.asarray(X)


                # Manual Annotation
                if file == 'foot_pacing_test_1.markonly.mp4':
                    y = [0 for _ in range(1,35)]
                    y.extend([1 for _ in range(35,98)])
                    y.extend([0 for _ in range(98, 150)])
                    y.extend([1 for _ in range(150, 184)])
                    y.extend([0 for _ in range(184, 262)])
                    y.extend([1 for _ in range(262, 309)])
                    y.extend([0 for _ in range(309,len(X)+1)])

                elif file == 'foot_pacing_test_2.markonly.mp4':
                    y = [0 for _ in range(1, 43)]
                    y.extend([1 for _ in range(43, 96)])
                    y.extend([0 for _ in range(96, 103)])
                    y.extend([1 for _ in range(103, 165)])
                    y.extend([0 for _ in range(165, 218)])
                    y.extend([1 for _ in range(218, 245)])
                    y.extend([0 for _ in range(245, 288)])
                    y.extend([1 for _ in range(288, len(X) + 1)])


                elif file == 'foot_pacing_train_1.markonly.mp4':
                    y = [0 for _ in range(1, 55)]
                    y.extend([1 for _ in range(55, 88)])
                    y.extend([0 for _ in range(88, 119)])
                    y.extend([1 for _ in range(119, 155)])
                    y.extend([0 for _ in range(155, 225)])
                    y.extend([1 for _ in range(225, len(X) + 1)])


                elif file == 'foot_pacing_train_2.markonly.mp4':
                    y = [0 for _ in range(1, 43)]
                    y.extend([1 for _ in range(43, 118)])
                    y.extend([0 for _ in range(118, 193)])
                    y.extend([1 for _ in range(193, 263)])
                    y.extend([0 for _ in range(263, len(X) + 1)])

                elif file == 'foot_pacing_train_3.markonly.mp4':
                    y = [0 for _ in range(1, 63)]
                    y.extend([1 for _ in range(63, 95)])
                    y.extend([0 for _ in range(95, 177)])
                    y.extend([1 for _ in range(177, 211)])
                    y.extend([0 for _ in range(211, 291)])
                    y.extend([1 for _ in range(291, 312)])
                    y.extend([0 for _ in range(312, 384)])
                    y.extend([1 for _ in range(384, 407)])
                    y.extend([0 for _ in range(407, len(X) + 1)])

                elif file == 'foot_pacing_train_4.markonly.mp4':
                    y = [0 for _ in range(1, 61)]
                    y.extend([1 for _ in range(61, 102)])
                    y.extend([0 for _ in range(102, 132)])
                    y.extend([1 for _ in range(132, 169)])
                    y.extend([0 for _ in range(169, 250)])
                    y.extend([1 for _ in range(250, len(X) + 1)])

                elif file == 'foot_pacing_train_5.markonly.mp4':
                    y = [0 for _ in range(1, 50)]
                    y.extend([1 for _ in range(50, 90)])
                    y.extend([0 for _ in range(90, 119)])
                    y.extend([1 for _ in range(119, 150)])
                    y.extend([0 for _ in range(150, 283)])
                    y.extend([1 for _ in range(283, 356)])
                    y.extend([1 for _ in range(356, len(X) + 1)])

                elif file == 'foot_pacing_train_6.markonly.mp4':
                    y = [0 for _ in range(1, 58)]
                    y.extend([1 for _ in range(58, 97)])
                    y.extend([0 for _ in range(97, 139)])
                    y.extend([1 for _ in range(139, 172)])
                    y.extend([0 for _ in range(172, 271)])
                    y.extend([1 for _ in range(271, 305)])
                    y.extend([0 for _ in range(305, 320)])
                    y.extend([1 for _ in range(320, 390)])
                    y.extend([0 for _ in range(390, len(X) + 1)])


                else:
                    continue

                assert len(X) == len(y)
                out_dir = './pacing/marked_only_data'
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                y = np.array(y)
                pickle.dump(X, open(os.path.join(out_dir,'X_'+file.split('.')[0]),'wb'),protocol=4)
                pickle.dump(y, open(os.path.join(out_dir,'y_'+file.split('.')[0]),'wb'), protocol=4)
                if 'test' in file:
                    x_axis = [i / frame_rate for i in range(len(y))]
                    plt.plot(x_axis, y)
                    plt.savefig('./pacing/'+file.split('.')[0]+'_gold_label.png')
                    plt.close()

    else:
        X = []
        for count,img in enumerate(predic_frames):
            if img.shape != (480, 272, 3):
                # logger.info(file,img.shape)
                img = cv2.resize(img, (272, 480))
                # logger.info(img.shape)
                # cv2.imwrite('./test.jpg', img)
                X.append(img)  # storing each image in array X



        X = np.asarray(X)
        # print(X.shape)

    logger.info('Done!')
    return X





def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()



def diff_first_frame(X):
    a = len(X)
    first_frames = X[0].view(1,3,416,240).expand(a,3,416,240)
    # logger.info(first_frames[0],first_frames[1])
    # logger.info(first_frames.size())
    diff_X = X - first_frames
    # logger.info(diff_X.size())
    # logger.info(diff_X[0],diff_X[1])
    return diff_X

def diff_adjacent_frame(X):
    a = len(X)
    # logger.info(a)
    diff_X = copy.deepcopy(X)
    for i in range(a):
        if i == 0:
            diff_X[i] = diff_X[i] - X[i]
        else:
            diff_X[i] = diff_X[i] - X[i-1]

    return diff_X

class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Net(nn.Module):
    def __init__(self,sample_rate):
        super(Net, self).__init__()
        self.sample_rate = sample_rate
        # Conv 3D
        self.resnet3d =resnet34(sample_size =  64,sample_duration =16, num_classes=512)



        # # Pretrained 2D
        # self.resnet = models.resnet34(pretrained = True)
        # self.fc1 = nn.Linear(1000, 512)

        # Common CNN
        # self.conv1 = nn.Conv2d(3, 6, 16)
        # self.pool = nn.MaxPool2d(5, 5)
        # self.conv2 = nn.Conv2d(6, 16, 16)
        # self.fc1 = nn.Linear(16 * 6 * 13, 512)
        # self.BN1 = nn.BatchNorm2d(6)
        # self.BN2 = nn.BatchNorm2d(16)

        # Transformer
        # self.nlayers = 2
        # self.pos_encoder = PositionalEncoding(512, dropout=0.2)
        # encoder_layer = TransformerEncoderLayer(512, 4, 128, 0.6, activation="relu")
        # encoder_norm = LayerNorm(512)
        # self.transformer_encoder = TransformerEncoder(encoder_layer, self.nlayers, encoder_norm)


        # # RNN
        self.nlayers = 1
        self.rnn = nn.LSTM(
            512,
            256,
            num_layers=self.nlayers,
            dropout=0.0 if self.nlayers == 1 else 0.5,
            bidirectional=True,
            batch_first=True,
        )
        self.fc2 = nn.Linear(512, 32)
        self.fc3 = nn.Linear(32, 2)
        self.drop_out = nn.Dropout(0.5)


    def forward(self, x):
        # logger.info(x.size())
        # TODO: change to 3d

        # Conv3d
        if not self.training:
            x = x.unsqueeze(0)
        x = x.permute(0,4,1,2,3)
        x = self.resnet3d(x)
        x = F.relu(x)


        # # ResNet2d
        # if self.training:
        #     x = x.reshape(-1,x.size(2),x.size(3),x.size(4))
        # x = x.permute(0, 3, 1, 2)
        # x = self.resnet(x)
        # if self.training:
        #     x = F.relu(self.fc1(x)).reshape(-1,self.sample_rate,512)
        # else:
        #     x = F.relu(self.fc1(x)).reshape(1, -1, 512)


        # # Common CNN
        # res_out_dim = x.sise()[-1]
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.drop_out(self.BN1(x))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = self.drop_out(self.BN2(x))
        # x = x.view(-1, 16 * 6 * 13)


        # RNN
        rnn_output, hidden = self.rnn(x)
        rnn_output = rnn_output.squeeze()
        x = self.drop_out(rnn_output)

        # Transformer
        # x = x* math.sqrt(512)
        # x = self.pos_encoder(x)
        # x = self.transformer_encoder(x)

        x = F.relu(self.fc2(x))
        x = self.drop_out(x)
        x = self.fc3(x).reshape(-1,2)
        return x



# Data augmentation utils
# https://medium.com/@thimblot/data-augmentation-boost-your-image-dataset-with-few-lines-of-python-155c2dc1baec

import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util

def random_rotation(image_array: ndarray):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array)

def horizontal_flip(image_array: ndarray):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]

def vertical_flip(image_array: ndarray):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return np.flipud(image_array)

# dictionary of the transformations functions we defined earlier
def train_aug(image_to_transform):
    available_transformations = {
        'rotate': random_rotation,
        # 'noise': random_noise,
        'horizontal_flip': horizontal_flip,
        # 'vertical_flip': vertical_flip
    }

    # random num of transformations to apply
    num_transformations_to_apply = random.randint(1, len(available_transformations))

    num_transformations = 0
    transformed_image = None
    while num_transformations <= num_transformations_to_apply:
        # choose a random transformation to apply for a single image
        key = random.choice(list(available_transformations))
        transformed_image = available_transformations[key](image_to_transform)
        num_transformations += 1
    return transformed_image





def sampling(X_list,y_list,start,sample_rate):
    X_samples,y_samples = [],[]
    for i in range(len(X_list)):
        X,y = X_list[i],y_list[i]
        # print(len(X),len(X)//sample_rate)
        for j in range(len(X)//sample_rate-1):
            # print(X[start+j*sample_rate:start+(j+1)*sample_rate].shape)
            X_samples.append(X[start+j*sample_rate:start+(j+1)*sample_rate])
            y_samples.append(y[start+j*sample_rate:start+(j+1)*sample_rate])
        # add the final pieces
        # print(X[len(X)- sample_rate:len(X)].shape)
        X_samples.append(X[len(X)- sample_rate:len(X)])
        y_samples.append(y[len(X)- sample_rate:len(X)])
    # print(len(X_samples))
    shuffle_index = np.random.permutation(len(X_samples))
    X_result = np.asarray(X_samples)[shuffle_index]
    y_result = np.asarray(y_samples)[shuffle_index]
    return X_result,y_result





def Train_LRCN():
    data_type = 'foot_pacing'
    train_count = 6
    test_count = 2
    max_epoch = 50
    sample_rate = 16
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net(sample_rate)
    logger.info(model)
    weights = [3, 1]
    class_weights = torch.FloatTensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    # load training data
    logger.info('Loading data....')
    X_train = []
    y_train = []
    for m in range(1, train_count+1):
        frames, _ = get_frames(predict_videoFile='./pacing/marked_only_videos/foot_pacing_train_%d.markonly.mp4'% ( m))
        assert frames
        X = prepare_data(frames)
        y = pickle.load(open('./pacing/marked_only_data/y_%s_train_%d' % (data_type, m), 'rb'))
        assert len(X) == len(y)
        X_train.append(X)
        y_train.append(y)

    # Add more negative samples
    for m in range(1, 13):
        if m in [1,2,3,7,10,11,12]:
            frames, _ = get_frames(predict_videoFile='./withdrawing/marked_only_videos/foot_withdrawing_train_%d.markonly.mp4'% ( m))
            assert frames
            X = prepare_data(frames)
            y = np.zeros(len(X))
            X_train.append(X)
            y_train.append(y)

    for m in range(1, 15):
        if m in [1,4,7,8]:
            frames, _ = get_frames(predict_videoFile='./turning/marked_only_videos/foot_turning_away_train_%d.markonly.mp4'% ( m))
            assert frames
            X = prepare_data(frames)
            y = np.zeros(len(X))
            X_train.append(X)
            y_train.append(y)

    logger.info(sum([sum(y) for y in y_train])/sum([len(y) for y in y_train]))

    # load test data
    X_test = []
    y_test = []

    # Use new test data
    for m in range(1, test_count + 1):
        frames, _ = get_frames(predict_videoFile='./pacing/marked_only_videos/foot_pacing_test_%d.markonly.mp4' % (m))
        assert frames
        X = prepare_data(frames)
        y = pickle.load(open('./pacing/marked_only_data/y_%s_test_%d' % (data_type, m), 'rb'))
        assert len(X) == len(y)
        X_test.append(torch.tensor(X, dtype=torch.float))
        y_test.append(torch.tensor(y, dtype=torch.long))

    # Add more negative samples
    for m in range(1, 5):
        if m in [1,2,3]:
            frames, _ = get_frames(predict_videoFile='./withdrawing/marked_only_videos/foot_withdrawing_test_%d.markonly.mp4'% ( m))
            assert frames
            X = prepare_data(frames)
            y = np.zeros(len(X))
            X_test.append(torch.tensor(X,dtype=torch.float))
            y_test.append(torch.tensor(y,dtype=torch.long))

    for m in range(1, 4):
        if m in [1,2]:
            frames, _ = get_frames(predict_videoFile='./turning/marked_only_videos/foot_turning_away_test_%d.markonly.mp4'% ( m))
            assert frames
            X = prepare_data(frames)
            y = np.zeros(len(X))
            X_test.append(torch.tensor(X,dtype=torch.float))
            y_test.append(torch.tensor(y,dtype=torch.long))


    best_f1 = 0
    loss_total = 0
    lens = [len(X_train[i]) - sample_rate for i in range(len(X_train))]
    for epoch in range(max_epoch):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        logger.info('Training......')
        model = model.to(device)
        model.train()
        model.zero_grad()
        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.

        # start = random.randint(0, sample_rate)
        # curr_x_train, curr_y_train = sampling(X_train,y_train,start,sample_rate)
        # # print(len(curr_x_train))
        # # print(curr_x_train.shape,curr_y_train.shape)
        # running_loss = 0.0
        # for batch_i in range(math.ceil(len(curr_x_train)/batch_size)):
        #     x_batch = torch.tensor(curr_x_train[batch_i * batch_size:(batch_i + 1) * batch_size],dtype=torch.float)
        #     y_batch = torch.tensor(curr_y_train[batch_i * batch_size:(batch_i + 1) * batch_size],dtype=torch.long).reshape(-1)

        running_loss = 0.0
        for batch_i in range(int(max(lens) / sample_rate * 4)):

            starts = [random.randint(0, length) for length in lens]
            # logger.info('Random Starts:'+str(starts))
            a = np.asarray([X_train[i][starts[i]:starts[i] + sample_rate] for i in range(len(X_train))])
            x_batch = torch.tensor(a, dtype=torch.float)
            y_batch = torch.tensor(
                np.asarray([y_train[i][starts[i]:starts[i] + sample_rate] for i in range(len(X_train))]),
                dtype=torch.long)

            shuffle_index = np.random.permutation(len(lens))
        #
        #     # Shuffle
        #     x_batch_split = x_batch[shuffle_index]
        #     y_batch_split = y_batch[shuffle_index].reshape(-1)
        #

            # Batch too big, spit
            batch_size = int(len(lens) / 2)
            for split_i in range(2):
                # Shuffle
                x_batch_split = x_batch[shuffle_index[split_i * batch_size:(split_i + 1) * batch_size]]
                y_batch_split = y_batch[shuffle_index[split_i * batch_size:(split_i + 1) * batch_size]].reshape(-1)

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = model(x_batch_split.to(device, dtype=torch.float))
                loss = criterion(outputs, y_batch_split.to(device, dtype=torch.long))
                loss.backward()
                optimizer.step()
                loss_total += loss.item()

                # print statistics
                running_loss += loss.item()
                if batch_i % 5 == 0 and batch_i:
                    logger.info('[%d, %5d, %5d] loss: %.3f' %
                                (epoch + 1, batch_i + 1, split_i, running_loss / 5))
                    running_loss = 0.0
        # test
        logger.info("Testing..")
        model.eval()
        with torch.no_grad():
            y_preds = []
            y_tests = []
            for test_i,test in enumerate(X_test):
                # print(test.shape)
                model = model.to('cpu')
                model_out = model(test.to('cpu',dtype=torch.float)).numpy()
                y_pred = model_out.argmax(axis=1)
                y_preds.extend(y_pred)
                y_tests.extend(y_test[test_i].numpy())

            logger.info(classification_report(y_tests, y_preds))
            report = classification_report(y_tests, y_preds,output_dict=True)
            if (report['1']['f1-score'])>best_f1 and (model_out.argmax(axis=1).sum()!=len(model_out)):
                torch.save(model, './save/%s'%data_type)
                best_f1 = report['1']['f1-score']
                logger.info('Best F1:'+str(best_f1))

    logger.info('Finished Training')




def Predict_LRCN(data_type = 'foot_pacing' ,test_video='./pacing/marked_only_videos/foot_pacing_test_1.markonly.mp4'):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')

    file_name = test_video.split('/')[-1].split('.')[0]
    logger.info(file_name)
    frames,frame_rate = get_frames(predict_videoFile=test_video)
    if not frames:
        print('No video found!')
        exit()
    X_test = torch.tensor(prepare_data(frames),dtype=torch.float)
    print(X_test.shape)
    model = torch.load('./save/%s'%data_type, map_location=device)
    model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    model.to(device)
    with torch.no_grad():
        model_out = model(X_test.to(device, dtype=torch.float)).to("cpu").numpy()
        scores = softmax(model_out,axis=1)
    pred_scores = []
    results_dic = {}
    results_dic[file_name] = []
    for i,score in enumerate(scores):
        results_dic[file_name].append([str(i/frame_rate),str(score[1])])
        pred_scores.append(score[1])

    logger.info('Saving results to figure ./pacing/%s_timeLabel_%s.png'%(file_name,data_type))
    # logger.info(pred_scores)
    x_axis = [i/frame_rate for i in range(len(pred_scores))]
    plt.plot(x_axis,pred_scores)
    plt.ylim(-0.1, 1.1)
    plt.savefig('./pacing/%s_%s.png'%(file_name,data_type))
    # plt.show(block=True)
    # plt.close()





if __name__ == '__main__':
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
    # First, get frames of the body-marked videos
    # _ = get_frames()
    # Then, pre-processing the frames (images)
    # _ = prepare_data()
    # Train the model
    # Train_LRCN()


    # Notice for demo:
    # 1. Put test videos in './marked/ directory'
    # 2. Make sure that the models are located in './save/' directory
    # 3. The input of the model is the 'data_type' and the 'test_video' file location
    # 4. Data_types: foot_pacing; foot_withdrawing; foot_turning_away;
    # 5. After prediction, the results are saved in 'frameLabel.json' and 'FILENAME_pred_label.png'
    # Predict the results
    import argparse

    parser = argparse.ArgumentParser(description='Test vedio')
    parser.add_argument('--type', default='foot_pacing',
                        help='3 options: foot_withdrawing, foot_pacing, foot_turning_away')
    parser.add_argument('--video', default='./pacing/marked_only_videos/foot_pacing_test_1.markonly.mp4',
                        help='path to test video')
    # parser.add_argument('--video', default='./test_marked_only/4400_4.markOnly.mp4',
    #                     help='path to test video')

    args = parser.parse_args()
    Predict_LRCN(data_type=args.type, test_video=args.video)

