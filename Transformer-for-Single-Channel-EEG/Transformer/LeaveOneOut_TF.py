import torch
import scipy.io as sio
import numpy as np
from sklearn.metrics import accuracy_score
import torch.optim as optim
from Transformer.TransformerModel import TransformerModel

torch.cuda.empty_cache()
torch.manual_seed(0)


def run():
    # load data from the file
    filename = r'../dataset.mat'

    tmp = sio.loadmat(filename)
    xdata=np.array(tmp['EEGsample'])
    label=np.array(tmp['substate'])
    subIdx=np.array(tmp['subindex'])

    label.astype(int)
    subIdx.astype(int)

    samplenum=label.shape[0]

    # there are 11 subjects in the dataset. Each sample is 3-seconds data from 30 channels with sampling rate of 128Hz.
    channelnum=30
    subjnum=11
    samplelength=3
    sf=128

    # define the learning rate, batch size and epoches
    lr=1e-2
    batch_size = 50
    n_epoch =6

    # ydata contains the label of samples
    ydata=np.zeros(samplenum,dtype=np.longlong)

    for i in range(samplenum):
        ydata[i]=label[i]

    # only channel 28 is used, which corresponds to the Oz channel
    selectedchan=[28]

    # update the xdata and channel number
    xdata=xdata[:,selectedchan,:]
    channelnum=len(selectedchan)

    # the result stores accuracies of every subject
    results=np.zeros(subjnum)


    # it performs leave-one-subject-out training and classfication
    # for each iteration, the subject i is the testing subject while all the other subjects are the training subjects.
    for i in range(1,subjnum+1):

        # form the training data
        trainindx=np.where(subIdx != i)[0]
        xtrain=xdata[trainindx]
        x_train = xtrain.reshape(xtrain.shape[0],1,channelnum, samplelength*sf)
        y_train=ydata[trainindx]

        # form the testing data
        testindx=np.where(subIdx == i)[0]
        xtest=xdata[testindx]
        x_test = xtest.reshape(xtest.shape[0], 1,channelnum, samplelength*sf)
        y_test=ydata[testindx]


        train = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

        # load the TF model to deal with 1D EEG signals
        my_net = TransformerModel().double().cpu()

        optimizer = optim.Adam(my_net.parameters(), lr=lr)
        loss_class = torch.nn.NLLLoss().cuda()

        for p in my_net.parameters():
            p.requires_grad = True

        # train the classifier
        for epoch in range(n_epoch):
            for j, data in enumerate(train_loader, 0):
                inputs, labels = data

                # restructure the input data
            #   input_data = inputs.view(samplelength*sf, -1, channelnum).cpu()

                #input_data = inputs.view(samplelength * sf, -1).cpu()
                input_data = inputs.view(-1, samplelength * sf).cpu()

                #class_label = labels.cpu()
                class_label = labels.view(-1, 1).cpu()

                class_label = labels.squeeze().cpu()

                # print("Input shape:", input_data.shape)
                # print("Labels shape:", class_label.shape)

                my_net.zero_grad()
                my_net.train()

                class_output = my_net(input_data)

                # print("Output shape:", class_output.shape)
                err_s_label = loss_class(class_output, class_label)
                err = err_s_label

                err.backward()
                optimizer.step()

        # test the results
        my_net.train(False)
        with torch.no_grad():
            x_test =  torch.DoubleTensor(x_test).cpu()
            # restructure the input data
            #x_test = x_test.view(samplelength*sf, -1, channelnum)
            x_test = x_test.view(-1, samplelength * sf)

            answer = my_net(x_test)
            probs=answer.cpu().numpy()
            preds= probs.argmax(axis = -1)
            acc=accuracy_score(y_test, preds)

            print(acc)
            results[i-1]=acc

    print('mean accuracy:', np.mean(results))

if __name__ == '__main__':
    run()
