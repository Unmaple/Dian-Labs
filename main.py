import torch
import torch.nn as nn

from torch.utils import data
from tqdm import tqdm

from tvid import TvidDataset
from detector import Detector
from utils import compute_iou
from PIL import Image
import torchvision.transforms as tr
#from matplotlib import pyplot as plt


lr = 5e-3
batch = 32
epochs = 100
device = "cuda" if torch.cuda.is_available() else "cpu"
#print(device)
iou_thr = 0.5
GAMMA = 0.965
#maxacc = 0

def train_epoch(model, dataloader, criterion: dict, optimizer,
                scheduler, epoch, device):
    model.train()
    # bar = tqdm()
    # bar.set_description(f'epoch {epoch:2}')
    correct, total = 0, 0
    #dropout = nn.functional.dropout()
    #x_drop = dropout(x)
    for x, y in dataloader:
        temp_tr_loss_sum = 0
        # TODO Implement the train pipeline.
        x = x.to(device)


        #x = nn.functional.dropout(x,p=0.1)


        y = y.to(device)
        temp_y_pred = model(x)
        temp_loss = criterion['cls'](temp_y_pred, y)
        optimizer.zero_grad()
        temp_loss.backward()
        optimizer.step()
        temp_tr_loss_sum += temp_loss.cpu().item()
        arg = temp_y_pred.argmax(dim=1)
        y = y.argmax(dim = 1)
        correct += (arg == y).sum().cpu().item()
        #temp_num += y.shape[0]
        # temp_batch_count += 1
    #test_acc = evaluate_accuracy(te_iter, model)
    # print("Epoch %d, loss %.4f, training acc %.3f, test ass %.3f, time %.1f s" %
    #       (epoch + 1, temp_tr_loss_sum / temp_batch_count, temp_tr_acc_sum / temp_num, test_acc,
    #        time.time() - temp_start_time))
        ...
        # unloader = tr.ToPILImage()(x[0].squeeze())
        # print(y)
        # unloader.show()

        total += batch

        # End of todo

        # bar.set_postfix_str(f'lr={scheduler.get_last_lr()[0]:.4f} acc={correct / total * 100:.2f}loss={temp_tr_loss_sum / total:.5f}')
    scheduler.step()
    return correct / total * 100.0



def test_epoch(model, dataloader, device, epoch):
    model.eval()
    with torch.no_grad():
        correct, correct_cls, total = 0, 0, 0
        for X, y in dataloader:

            # TODO Implement the test pipeline.
            x = X.to(device)
            y = y.to(device)
            temp_y_pred = model(x)
            correct += (temp_y_pred.argmax(dim=1) == y.argmax(dim = 1)).sum().cpu().item()
            ...
            total += batch
            #unloader = tr.ToPILImage()(X[0].squeeze())

            #print(temp_y_pred.argmax(dim=1)[0],y.argmax(dim = 1)[0])
            #unloader.show()
            # End of todo

        #print(f' val acc: {correct / total * 100:.2f}')
        return correct / total * 100.0


def main():
    trainloader = data.DataLoader(TvidDataset(root='./data/tiny_vid', mode='train'),
                                  batch_size=batch, shuffle=True, num_workers=4)
    testloader = data.DataLoader(TvidDataset(root='./data/tiny_vid', mode='test'),
                                 batch_size=batch, shuffle=True, num_workers=4)
    model = Detector(backbone='resmodel50', lengths=(2048 * 4 * 4, 2048, 512),
                     num_classes=5,norm_layer = nn.BatchNorm2d).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                weight_decay=1e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=GAMMA,
                                                last_epoch=-1)
    criterion = {'cls': nn.CrossEntropyLoss(), 'box': nn.L1Loss()}
    maxacc = 0
    maxaccepoch = 0
    bar = tqdm(range(epochs))

    for epoch in bar:
        bar.set_description(f'epoch {epoch:2}')
        traacc = train_epoch(model, trainloader, criterion, optimizer,
                    scheduler, epoch, device)
        tesacc = test_epoch(model, testloader, device, epoch)

        if tesacc > maxacc :
            maxacc = tesacc
            maxaccepoch = epoch
        bar.set_postfix_str(f'tracc={traacc:.2f} treacc={tesacc:.2f} maxacc={maxacc:.2f} maxaccepoch={maxaccepoch}')

    print(maxacc)
    #maxacc = 0

def test():
    while(1):
        main()
if __name__ == '__main__':
    test()
