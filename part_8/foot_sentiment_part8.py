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
from scipy.special import softmax
from torchvision import models



# Turn videos to frames
def get_frames(predict_videoFile = None):
    print('Getting frames...')

    if not predict_videoFile:
        for (dirpath, dirnames, filenames) in os.walk('./marked/'):
            for file in filenames:

                file_path = os.path.join(dirpath,file)
                count = 0
                cap = cv2.VideoCapture(file_path)   # capturing the video from the given path
                frameRate = cap.get(5) #frame rate
                print('frame rate:',frameRate)
                x=1
                frames = []
                while(cap.isOpened()):
                    frameId = cap.get(1) #current frame number
                    ret, frame = cap.read()
                    # print(ret,frame)
                    if (ret != True):
                        break
                    count+=1
                    # if (frameId % math.floor(frameRate) == 0):
                    # print(count)
                    framename = "frame%d.jpg" % count
                    out_dir = os.path.join('./pic/', file)
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    cv2.imwrite(os.path.join(out_dir,framename), frame)
                cap.release()
        print ("Done!")
        return None

    else:
        count = 0
        cap = cv2.VideoCapture(predict_videoFile)   # capturing the video from the given path
        frameRate = cap.get(5) #frame rate
        print('frame rate:',frameRate)
        x=1
        frames = []
        while(cap.isOpened()):
            frameId = cap.get(1) #current frame number
            ret, frame = cap.read()
            # print(ret,frame)
            if (ret != True):
                break
            count+=1
            # print(count)
            frames.append(frame)
            # plt.imshow(frame)
            # plt.close()
        cap.release()
        print ("Done!")
        return frames,frameRate


# preprocess the data and annotate each frame.
def prepare_data(predic_frames=None):
    print('Preparing data...')
    if not predic_frames:
        for (dirpath, dirnames, _) in os.walk('./pic/'):
            for dir in dirnames:
                print(dir)
                for (_, _, filenames) in os.walk(os.path.join(dirpath,dir)):
                    X = []
                    for i in range(len(filenames)):
                        file = os.path.join(dirpath,dir,'frame%d.jpg'%(i+1))
                        img = plt.imread(file)
                        if img.shape != (240, 416, 3):
                            # print(file,img.shape)
                            img = cv2.resize(img, (416, 240))
                            # print(img.shape)
                            # cv2.imwrite('./test.jpg', img)

                        X.append(img)  # storing each image in array X
                        # print(file, img.shape)
                    X = np.array(X)  # converting list to array
                    # Manually add labels to y
                    if dir == 'foot_withdrawing_train6.mark.mp4':
                        y = [0 for _ in range(1,12)]
                        y.extend([1 for _ in range(12,31)])
                        y.extend([0 for _ in range(31,len(X)+1)])

                    elif dir == 'foot_turning_away_train2.mark.mp4':
                        y = [0 for _ in range(1, 7)]
                        y.extend([1 for _ in range(7, 26)])
                        y.extend([0 for _ in range(26, len(X)+1)])

                    elif dir == 'foot_pacing_train6.mark.mp4':
                        y = [0 for _ in range(1, 20)]
                        y.extend([1 for _ in range(20, 103)])
                        y.extend([0 for _ in range(103, len(X)+1)])

                    elif dir == 'foot_withdrawing_train2.mark.mp4':
                        y = [1 for _ in range(1, 20)]
                        y.extend([0 for _ in range(20, len(X)+1)])

                    elif dir == 'foot_pacing_train3.mark.mp4':
                        y = [0 for _ in range(1, 19)]
                        y.extend([1 for _ in range(19, 101)])
                        y.extend([0 for _ in range(101, len(X)+1)])

                    elif dir == 'foot_withdrawing_train4.mark.mp4':
                        y = [0 for _ in range(1, 41)]
                        y.extend([1 for _ in range(41, 61)])
                        y.extend([0 for _ in range(61, len(X)+1)])

                    elif dir == 'foot_pacing_train2.mark.mp4':
                        y = [0 for _ in range(1, 41)]
                        y.extend([1 for _ in range(41, 61)])
                        y.extend([0 for _ in range(61, len(X)+1)])

                    elif dir == 'foot_turning_away_train4.mark.mp4':
                        y = [0 for _ in range(1, 30)]
                        y.extend([1 for _ in range(30, 46)])
                        y.extend([0 for _ in range(46, len(X)+1)])

                    elif dir == 'foot_pacing_train5.mark.mp4':
                        y = [0 for _ in range(1, 14)]
                        y.extend([1 for _ in range(14, len(X)+1)])

                    elif dir == 'foot_withdrawing_test2.mark.mp4':
                        y = [0 for _ in range(1, 97)]
                        y.extend([1 for _ in range(97, 128)])
                        y.extend([0 for _ in range(128, len(X)+1)])

                    elif dir == 'foot_pacing_test2.mark.mp4':
                        y = [0 for _ in range(1, 7)]
                        y.extend([1 for _ in range(7, 111)])
                        y.extend([0 for _ in range(111, len(X)+1)])

                    elif dir == 'foot_pacing_train4.mark.mp4':
                        y = [1 for _ in range(1, 101)]
                        y.extend([0 for _ in range(101, len(X)+1)])

                    elif dir == 'foot_pacing_test1.mark.mp4':
                        y = [0 for _ in range(1, 31)]
                        y.extend([1 for _ in range(31, 113)])
                        y.extend([0 for _ in range(113, len(X)+1)])

                    elif dir == 'foot_turning_away_train1.mark.mp4':
                        y = [0 for _ in range(1, 16)]
                        y.extend([1 for _ in range(16, 37)])
                        y.extend([0 for _ in range(37, len(X)+1)])

                    elif dir == 'foot_withdrawing_test1.mark.mp4':
                        y = [0 for _ in range(1, 53)]
                        y.extend([1 for _ in range(53, 78)])
                        y.extend([0 for _ in range(78, len(X)+1)])

                    elif dir == 'foot_withdrawing_test3.mark.mp4':
                        y = [0 for _ in range(1, 5)]
                        y.extend([1 for _ in range(5, 37)])
                        y.extend([0 for _ in range(37, len(X)+1)])

                    elif dir == 'foot_withdrawing_train1.mark.mp4':
                        y = [0 for _ in range(1, 27)]
                        y.extend([1 for _ in range(27, 114)])
                        y.extend([0 for _ in range(114, len(X)+1)])

                    elif dir == 'foot_withdrawing_train5.mark.mp4':
                        y = [0 for _ in range(1, 32)]
                        y.extend([1 for _ in range(32, 52)])
                        y.extend([0 for _ in range(52, len(X)+1)])

                    elif dir == 'foot_withdrawing_train3.mark.mp4':
                        y = [0 for _ in range(1, 50)]
                        y.extend([1 for _ in range(50, 68)])
                        y.extend([0 for _ in range(68, len(X)+1)])

                    elif dir == 'foot_pacing_train1.mark.mp4':
                        y = [0 for _ in range(1, 43)]
                        y.extend([1 for _ in range(43, len(X)+1)])

                    elif dir == 'foot_turning_away_train3.mark.mp4':
                        y = [0 for _ in range(1, 11)]
                        y.extend([1 for _ in range(11, 33)])
                        y.extend([0 for _ in range(33, len(X) + 1)])

                    elif dir == 'foot_turning_away_test.mark.mp4':
                        y = [0 for _ in range(1, 17)]
                        y.extend([1 for _ in range(17, 43)])
                        y.extend([0 for _ in range(43, len(X) + 1)])

                    assert len(X) == len(y)
                    y = np.array(y)
                    pickle.dump(X, open('./data/'+'X_'+dir.split('.')[0],'wb'),protocol=4)
                    pickle.dump(y, open('./data/'+'y_'+dir.split('.')[0],'wb'), protocol=4)
                    if 'test' in dir:
                        plt.plot(y)
                        plt.savefig(dir.split('.')[0]+'_gold_label.png')
                        plt.close()

    else:
        X = []
        for img in predic_frames:
            if img.shape != (240, 416, 3):
                # print(file,img.shape)
                img = cv2.resize(img, (416, 240))
                # print(img.shape)
                # cv2.imwrite('./test.jpg', img)
                X.append(img)  # storing each image in array X

        X = np.asarray(X)

    print('Done')
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
    first_frames = X[0].view(1,3,240,416).expand(a,3,240,416)
    # print(first_frames[0],first_frames[1])
    # print(first_frames.size())
    diff_X = X - first_frames
    # print(diff_X.size())
    # print(diff_X[0],diff_X[1])
    return diff_X

def diff_adjacent_frame(X):
    a = len(X)
    # print(a)
    diff_X = X.clone()
    for i in range(a):
        if i == 0:
            diff_X[i] = diff_X[i] - X[i]
        else:
            diff_X[i] = diff_X[i] - X[i-1]

    # print(diff_X[0],diff_X[1],diff_X[2],diff_X[2].size())
    # print(first_frames[0],first_frames[1])
    # print(first_frames.size())
    # diff_X = X - first_frames
    # print(diff_X.size())
    # print(diff_X[0],diff_X[1])
    # print(X.shape)
    # import matplotlib
    # X_ = diff_X.numpy()
    # for j in range(X.shape[0]):
    #     print(X_[j])
    #     matplotlib.image.imsave('./images/outfile%d.jpg'%j, X_[j].reshape(240,416,3))
    return diff_X



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet = models.resnet34(pretrained = True)
        self.conv1 = nn.Conv2d(3, 6, 16)
        self.pool = nn.MaxPool2d(5, 5)
        self.conv2 = nn.Conv2d(6, 16, 16)
        # self.fc1 = nn.Linear(16 * 6 * 13, 512)
        self.fc1 = nn.Linear(1000, 512)
        self.fc2 = nn.Linear(512, 32)
        self.fc3 = nn.Linear(32, 2)
        self.drop_out = nn.Dropout(0.5)
        self.BN1 = nn.BatchNorm2d(6)
        self.BN2 = nn.BatchNorm2d(16)

    def forward(self, x):
        # print(x.size())
        x = self.resnet(x)
        # x = self.pool(F.relu(self.conv1(x)))
        # print(x.size())
        # x = self.drop_out(self.BN1(x))
        # # print(x.size())
        # x = self.pool(F.relu(self.conv2(x)))
        # # print(x.size())
        # x = self.drop_out(self.BN2(x))
        # # print(x.size())
        # x = x.view(-1, 16 * 6 * 13)
        # print(x.size())
        x = F.relu(self.fc1(x))
        # print(x.size())
        x = self.drop_out(x)
        x = F.relu(self.fc2(x))
        # print(x.size())
        x = self.drop_out(x)
        x = self.fc3(x)
        # print(x.size())
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



# from torchvision import transforms as tfs
# def train_augmentation(x):
#     im_aug = tfs.Compose([
#         tfs.RandomHorizontalFlip(),
#         tfs.RandomVerticalFlip(),
#         tfs.RandomRotation(60),
#         tfs.Normalize([0, 0, 0], [1, 1, 1])
#     ])
#     x = im_aug(x)
#     return x
#
# def test_augmentation(x):
#     im_aug = tfs.Compose([
#         tfs.Normalize([0, 0, 0], [1, 1, 1])
#     ])
#     x = im_aug(x)
#     return x


def Train_CNN():
    data_type = 'foot_withdrawing'
    train_count = 6
    test_count = 3
    batch_size = 16
    max_epoch = 300
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Only CNN
    model = Net()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    # load training data

    X_train = torch.zeros(0, 3, 240, 416)
    y_train = torch.zeros(0, dtype=torch.long)
    for m in range(1, train_count + 1):
        X = torch.tensor(pickle.load(open('./data/X_%s_train%d' % (data_type, m), 'rb')), dtype=torch.float).view(-1, 3,
                                                                                                                  240,
                                                                                                                  416)
        y = torch.tensor(pickle.load(open('./data/y_%s_train%d' % (data_type, m), 'rb')), dtype=torch.long)
        X = diff_adjacent_frame(X)


        X_train = torch.cat((X_train, X), dim=0)
        y_train = torch.cat((y_train, y), dim=0)

    # Add more negative samples
    X = torch.tensor(pickle.load(open('./data/X_%s_train%d' % ('foot_turning_away', 1), 'rb')), dtype=torch.float).view(-1, 3,
                                                                                                              240,
                                                                                                              416)
    y = torch.tensor(np.zeros(len(X)), dtype=torch.long)
    X = diff_adjacent_frame(X)
    X_train = torch.cat((X_train, X), dim=0)
    y_train = torch.cat((y_train, y), dim=0)

    X = torch.tensor(pickle.load(open('./data/X_%s_train%d' % ('foot_withdrawing', 1), 'rb')), dtype=torch.float).view(
        -1, 3,
        240,
        416)
    y = torch.tensor(np.zeros(len(X)), dtype=torch.long)

    X = diff_adjacent_frame(X)
    X_train = torch.cat((X_train, X), dim=0)
    y_train = torch.cat((y_train, y), dim=0)



    # load test data
    X_test = torch.zeros(0, 3, 240, 416)
    y_test = torch.zeros(0, dtype=torch.long)
    for m in range(1, test_count + 1):
        X = torch.tensor(pickle.load(open('./data/X_%s_test%d' % (data_type, m), 'rb')), dtype=torch.float).view(-1, 3,
                                                                                                                  240,
                                                                                                                  416)
        y = torch.tensor(pickle.load(open('./data/y_%s_test%d' % (data_type, m), 'rb')), dtype=torch.long)
        X = diff_adjacent_frame(X)
        X_test = torch.cat((X_test, X), dim=0)
        y_test = torch.cat((y_test, y), dim=0)

    # print(X_train.size(),y_train.size(),X_test.size(),y_test.size())
    num_samples = len(X_train)
    num_batches = int(num_samples / batch_size)
    print(num_samples, num_batches)
    # print('X_%s_train%d'%(data_type,train_count))
    best_acc = 0
    loss_total = 0
    for epoch in range(max_epoch):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.train()
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.

        # Shuffle
        shuffle_index = np.random.permutation(num_samples)
        curr_x_train = X_train[shuffle_index]
        curr_y_train = y_train[shuffle_index]

        running_loss = 0.0
        for i in range(num_batches):
            # get the inputs; data is a list of [inputs, labels]
            x_batch = curr_x_train[i * batch_size:(i + 1) * batch_size]
            y_batch = curr_y_train[i * batch_size:(i + 1) * batch_size]

            # Data Augmentation
            # x_batch_aug = []
            # x_batch = x_batch.numpy()
            # for x in range(x_batch.shape[0]):
            #     # TODO: should I make channel last?
            #     # TODO: normalize the arrary?
            #     x_batch_aug.append(train_aug(x_batch[x].reshape(240,416,3)).reshape(3,240,416))
            # x_batch = torch.tensor(np.asarray(x_batch_aug), dtype=torch.float)
            # print(x_batch.shape)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(x_batch.to(device,dtype=torch.float))
            loss = criterion(outputs, y_batch.to(device,dtype=torch.long))
            loss.backward()
            optimizer.step()
            loss_total+=  loss.item()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

            # test
        print("Testing..")
        model.eval()
        with torch.no_grad():
            model_out = model(X_test.to(device,dtype=torch.float)).to("cpu").numpy()
            # print(model_out.argmax(axis=1))
            # print(y_test)
            correct = (model_out.argmax(axis=1) == y_test.numpy()).sum()
            total = len(model_out)
            print("{}, {}/{}\n".format(correct / total, correct, total))
            if (correct/total)>best_acc and (model_out.argmax(axis=1).sum() !=0):
                torch.save(model, './save/cnn_model_neg_resnet1_%s'%data_type)
                best_acc = correct / total
                print('Best Acc:',best_acc)

    print('Finished Training')




def Predict_CNN(data_type = 'foot_withdrawing' ,test_video='./marked/foot_withdrawing_test2.mark.mp4'):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')

    file_name = test_video.split('/')[-1].split('.')[0]
    print(file_name)
    frames,frame_rate = get_frames(predict_videoFile=test_video)
    X_test = torch.tensor(prepare_data(frames),dtype=torch.float)

    model = torch.load('./save/cnn_model_neg_resnet_%s'%data_type, map_location=device)
    model.eval()

    # for debugging!
    # X_test = torch.tensor(pickle.load(open('./data/X_%s%d'%(file_name,1), 'rb')), dtype=torch.float)
    # y = torch.tensor(pickle.load(open('./data/y_%s%d' % (file_name,1), 'rb')), dtype=torch.long)
    X_test = X_test.view(-1, 3, 240, 416)
    X_test = diff_adjacent_frame(X_test)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    model.to(device)
    with torch.no_grad():
        model_out = model(X_test.to(device, dtype=torch.float)).to("cpu").numpy()
        # print(model_out.shape)
        scores = softmax(model_out,axis=1)
        # print(scores)
        # for debugging
        # correct = (model_out.argmax(axis=1) == y.numpy()).sum()
        # total = len(model_out)
        # print("{}, {}/{}\n".format(correct / total, correct, total))
    pred_scores = []
    results_dic = {}
    results_dic[file_name] = []
    for i,score in enumerate(scores):
        results_dic[file_name].append([str(i/frame_rate),str(score[1])])
        pred_scores.append(score[1])
    print('Saving results to timeLabel.json')
    with open('./timeLabel.json','a') as f:
        json.dump(results_dic,f)
        f.write('\n')
    print('Saving results to figure timeLabel.png')
    # print(pred_scores)
    x_axis = [i/frame_rate for i in range(len(pred_scores))]
    plt.plot(x_axis,pred_scores)
    plt.savefig('timeLabel.png')
    # plt.show(block=True)
    # plt.close()





if __name__ == '__main__':
    # First, get frames of the body-marked videos
    # _ = get_frames()
    # Then, pre-processing the frames (images)
    # _ = prepare_data()
    # Train the model
    # Train_CNN()

    # Predict_CNN(data_type='foot_withdrawing', test_video='./marked/8_types_of_actions_trim.mark.mp4')


    # Notice for demo:
    # 1. Put test videos in './marked/ directory'
    # 2. Make sure that the models are located in './save/' directory
    # 3. The input of the model is the 'data_type' and the 'test_video' file location
    # 4. Data_types: foot_pacing; foot_withdrawing; foot_turning_away;
    # 5. After prediction, the results are saved in 'timeLabel.json' and 'timeLabel.png'
    # Predict the results
    import argparse

    parser = argparse.ArgumentParser(description='Test vedio')
    parser.add_argument('--type', default='foot_withdrawing',
                        help='3 options: foot_withdrawing, foot_pacing, foot_turning_away')
    parser.add_argument('--video', default='./marked/8_types_of_actions_trim.mark.mp4',
                        help='path to test video')

    args = parser.parse_args()
    Predict_CNN(data_type=args.type, test_video=args.video)

