import cv2     # for capturing videos
import matplotlib.pyplot as plt    # for plotting the images
import math
from sklearn import preprocessing
# %matplotlib inline
import numpy as np    # for mathematical operations
import pickle
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json

def get_frames():
    count = 0
    videoFile = "pacing.mark.mkv"
    cap = cv2.VideoCapture(videoFile)   # capturing the video from the given path
    frameRate = cap.get(5) #frame rate
    print(frameRate)
    x=1
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        # print(ret,frame)
        if (ret != True):
            break
        count+=1
        # if (frameId % math.floor(frameRate) == 0):
        print(count)
        filename ="frame%d.jpg" % count
        cv2.imwrite('./pic/'+filename, frame)
    cap.release()
    print ("Done!")


def prepare_data():
    X = []  # creating an empty array
    for i in range(1032):
        # print(i)
        img = plt.imread('./pic/frame%d.jpg'%i)
        # print(img.shape)
        X.append(img)  # storing each image in array X
    X = np.array(X)  # converting list to array
    y = [0 for _ in range(41)]
    y.extend([1 for _ in range(169)])
    y.extend([0 for _ in range(35)])
    y.extend([1 for _ in range(202)])
    y.extend([0 for _ in range(53)])
    y.extend([1 for _ in range(189)])
    y.extend([0 for _ in range(86)])
    y.extend([1 for _ in range(207)])
    y.extend([0 for _ in range(50)])
    y = np.array(y)
    assert len(X)==len(y)
    X_train,X_test = X[:700],X[700:]
    y_train,y_test = y[:700],y[700:]
    pickle.dump(X_train, open('X_train','wb'),protocol=4)
    pickle.dump(X_test, open('X_test','wb'), protocol=4)
    pickle.dump(y_train, open('y_train','wb'), protocol=4)
    pickle.dump(y_test, open('y_test','wb'),protocol=4)
    plt.plot(y_test)
    plt.savefig('gold_label.png')







class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 16)
        self.pool = nn.MaxPool2d(5, 5)
        self.conv2 = nn.Conv2d(6, 16, 16)
        # self.conv3 = nn.Conv2d(16, 32, 16)
        # self.conv4 = nn.Conv2d(32, 64, 16)
        self.fc1 = nn.Linear(16 * 11 * 5, 512)
        self.fc2 = nn.Linear(512, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.pool(x)
        # print(x.size())
        x = self.pool(F.relu(self.conv1(x)))
        # print(x.size())
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.size())
        # x = self.pool(F.relu(self.conv3(x)))
        # print(x.size())
        # x = self.pool(F.relu(self.conv4(x)))
        # print(x.size())
        x = x.view(-1, 16 * 11 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



def Train_CNN():
    # load data
    X_train = torch.tensor(pickle.load(open('X_train','rb')))
    X_test = torch.tensor(pickle.load(open('X_test', 'rb')))
    y_train = torch.tensor(pickle.load(open('y_train', 'rb')))
    y_test = torch.tensor(pickle.load(open('y_test', 'rb')))
    print(X_train.size(),X_test.size())
    X_train = X_train.view(-1,3,1920,1080)
    X_test = X_test.view(-1,3,1920,1080)
    print(X_train.size(), X_test.size())
    batch_size = 16
    max_epoch = 30
    num_samples = len(X_train)
    num_batches = int(num_samples / batch_size)
    # train model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net()
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    best_acc = 0
    for epoch in range(max_epoch):  # loop over the dataset multiple times
        # Shuffle
        net.train()
        shuffle_index = np.random.permutation(num_samples)
        curr_x_train = X_train[shuffle_index]
        curr_y_train = y_train[shuffle_index]

        running_loss = 0.0
        for i in range(num_batches):
            # get the inputs; data is a list of [inputs, labels]
            x_batch = curr_x_train[i * batch_size:(i + 1) * batch_size]
            y_batch = curr_y_train[i * batch_size:(i + 1) * batch_size]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(x_batch.to(device,dtype=torch.float))
            loss = criterion(outputs, y_batch.to(device,dtype=torch.long))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
        # test
        print("Testing..")
        net.eval()
        # if not vis is None:
        #     visdom_windows = None
        with torch.no_grad():
            correct = 0.0
            total = 0.0
            for j in tqdm(range(len(X_test))):
                # if not vis is None and i == 0:
                #     visdom_windows = plot_weights(model,visdom_windows,b,vocab,vis)
                sample = X_test[j].view(-1,3,1920,1080)
                model_out = net(sample.to(device,dtype=torch.float)).to("cpu").numpy()
                correct += (model_out.argmax(axis=1) == y_test[j].numpy()).sum()
                total += 1
            print("{}, {}/{}\n".format(correct / total, correct, total))
            if (correct/total)>best_acc:
                torch.save(net, './cnn_model')


    print('Finished Training')

def Predict_CNN():
    model = torch.load('./cnn_model')
    model.eval()
    X_test = torch.tensor(pickle.load(open('X_test', 'rb')))
    X_test = X_test.view(-1, 3, 1920, 1080)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        # for j in range(len(X_test)):
        sample = X_test.view(-1, 3, 1920, 1080)
        model_out = model(sample.to(device, dtype=torch.float)).to("cpu").numpy()
        scores = preprocessing.normalize(model_out)
    pred_scores = []
    results_dic = {}
    results_dic['foot_pacing'] = []
    for i,score in enumerate(scores):
        results_dic['foot_pacing'].append([str(i),str(score[1])])
        pred_scores.append(score[1])
    # print(results_dic)
    json.dump(results_dic,open('./frameLabel.json','w'))
    plt.plot(pred_scores)
    plt.savefig('pred_label.png')





if __name__ == '__main__':
    # First, get frames of the body-marked videos
    # get_frames()
    # Then, pre-processing the frames (images)
    # prepare_data()
    # Train the model
    # Train_CNN()
    # Predict the results
    Predict_CNN()