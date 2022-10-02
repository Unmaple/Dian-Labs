import os
from torch.utils import data
import transforms
import numpy as np
import torch
import torchvision.transforms as tr

# tivid 共5种label，每个label提取150个用作训练，30个用作检测
Ori_bbox = 128 * 128
Target = {'car':1, 'bird':2, 'turtle':3, 'dog':4, 'lizard':5}
num_classes = 5
Trans = 1
Trainnum = 150 * num_classes
Testnum = 30 * num_classes
LOADFILE = 1


class TvidDataset(data.Dataset):

    # TODO Implement the dataset class inherited
    # from `torch.utils.data.Dataset`.
    # tips: Use `transforms`.
    def __init__(self,root,mode):

        self.root = root
        self.mode = mode
        self.image = []
        load = transforms.LoadImage()
        ten = transforms.ToTensor()
        flip = transforms.RandomHorizontalFlip(1)
        crop = transforms.RandomCrop((64,64),)  # 训练使用的数据
        rdsize = transforms.RandomResize(0.5, 0.8)
        Nor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        gray = tr.Grayscale(num_output_channels=3)
        #Co = transforms.ColorJitter

        self.trainIn = torch.full([1,3,128,128],0)
        self.trainBB = torch.full([Trans * Trainnum+1,4],0)
        self.trainLa = torch.full([Trans * Trainnum+1,num_classes],0,dtype= float)
        self.tr = self.trainIn, self.trainLa
        # 测试使用的数据
        self.testIn = torch.full([1, 3, 128, 128], 0)
        self.testBB = torch.full([Testnum + 1, 4], 0)
        self.testLa = torch.full([Testnum + 1,num_classes], 0,dtype= float)
        self.te = self.testIn, self.testLa
        if LOADFILE:
            self.trainIn = torch.load("trainIn.pth")
            self.trainBB = torch.load("trainBB.pth")
            self.trainLa = torch.load("trainLa.pth")
            self.testIn = torch.load("testIn.pth")
            self.testBB = torch.load("testBB.pth")
            self.testLa = torch.load("testLa.pth")


        else:
            count = 1
            count1 = 1
            brightness = (0.5, 2.0)
            contrast = (0.5, 2.0)
            saturation = (0.5, 2.0)
            hue = (-0.2,0.2)
            color = tr.ColorJitter(brightness, contrast, saturation, hue)

            degrees = (-90, 90)
            translate = (0, 0.2)
            scale = (0.8, 1)
            fillcolor = (0, 0, 0)
            TRAF = tr.RandomAffine(degrees=degrees, translate=translate, scale=scale, fillcolor=fillcolor)
            for object in Target.keys():
                root1 = self.root + '/' + object
                file = open(root1 + '_gt.txt', "r")
                label_draft = file.read()
                file.close()
                # print(label_draft)
                label_draft = label_draft.replace('\n',' ')
                lines=label_draft.split()
                print(object)
                lines = list(map(int, lines))

                data = np.array(lines)
                data = data.reshape(-1,5)


                for i in range(1,151):
                    file_num = str(i).zfill(6)
                    file_path = root1 + '\\' + file_num + '.jpeg'
                    oribbox = [data[i][1], data[i][2], data[i][3], data[i][4]]
                    image, bbox = load(file_path, oribbox)
                    #image.show()


                    image1, bbox1 = ten(image,bbox)
                    #image1 = gray(image1)
                    self.trainIn = torch.cat((self.trainIn,torch.unsqueeze(image1,dim = 0)),dim = 0)
                    #print(image.size, self.trainIn[0].size,count)
                    self.trainBB[count] = bbox1
                    self.trainLa[count][Target[object]-1] = 1.0
                    count += 1
                    ## unloader = tr.ToPILImage()(image1)
                    ## unloader.show()
                    #
                    # image2, bbox2 = flip(image, bbox)
                    # image2, bbox2 = ten(image2, bbox2)
                    # image2, bbox2 = Nor(image2, bbox2)
                    # self.trainIn = torch.cat((self.trainIn, torch.unsqueeze(image2, dim=0)), dim=0)
                    # # print(image.size, self.trainIn[0].size,count)
                    # self.trainBB[count] = bbox2
                    # self.trainLa[count][Target[object]-1] = 1.0
                    # count += 1

                    # for k in range(3):
                    #     image3 = color(image)
                    #     bbox3 = bbox
                    #     #image3 = transforms.pad_if_smaller(image3,128)
                    #     # image3.show()
                    #     image3, bbox3 = ten(image3, bbox3)
                    #
                    #     self.trainIn = torch.cat((self.trainIn, torch.unsqueeze(image3, dim=0)), dim=0)
                    #     # print(image.size, self.trainIn[0].size,count)
                    #     self.trainBB[count] = bbox3
                    #     self.trainLa[count][Target[object]-1] = 1.0
                    #     count += 1
                    #
                    # for k in range(3):
                    #     image4 = TRAF(image)
                    #     bbox4 = bbox
                    #     # image3 = transforms.pad_if_smaller(image3,128)
                    #     #image4.show()
                    #     image4, bbox4 = ten(image4, bbox4)
                    #     image4, bbox4 = Nor(image4, bbox4)
                    #
                    #     self.trainIn = torch.cat((self.trainIn, torch.unsqueeze(image4, dim=0)), dim=0)
                    #     # print(image.size, self.trainIn[0].size,count)
                    #     self.trainBB[count] = bbox4
                    #     self.trainLa[count][Target[object] - 1] = 1.0
                    #     count += 1



                    # image3, bbox3 = rdsize(image, bbox)
                    # image3 = transforms.pad_if_smaller(image3,128)
                    # image3, bbox3 = ten(image3, bbox3)
                    # self.trainIn = torch.cat((self.trainIn, torch.unsqueeze(image3, dim=0)), dim=0)
                    # # print(image.size, self.trainIn[0].size,count)
                    # self.trainBB[count] = bbox3
                    # self.trainLa[count][Target[object]-1] = 1.0

                    # if i % 20 == 4:
                    #     print(self.trainLa[count])
                    #
                    # count += 1


                for i in range(151, 181):
                    file_num = str(i).zfill(6)
                    file_path = root1 + '\\' + file_num + '.jpeg'
                    oribbox = [data[i][1], data[i][2], data[i][3], data[i][4]]
                    image, bbox = load(file_path, oribbox)

                    image1, bbox1 = ten(image, bbox)
                    #image1 = gray(image1)
                    #image2, bbox2 = F.hflip(image)
                    self.testIn = torch.cat((self.testIn, torch.unsqueeze(image1, dim=0)), dim=0)
                    #print(image.size, self.trainIn[0].size, count)
                    self.testBB[count1] = bbox1
                    self.testLa[count1][Target[object]-1] = 1.0

                # if i % 10 == 2:
                #     print(self.testLa[count1])
                #     image.show()
                    count1 += 1


                # print(image,self.trainIn[0])
                # print(file_path)

        # if self.mode == 'train':
        #     z = np.zeros((self.trainLa.size(0), num_classes), )
        #     for j in range(1, self.trainLa.size(0)):
        #         z[j][self.trainLa[j]-1] = 1
        #     # print(np.sum(X),z)
        #     self.trainLa = z
        # else :
        #     z = np.zeros((self.testLa.size(0), num_classes), )
        #     for j in range(1, self.testLa.size(0)):
        #         z[j][self.testLa[j] - 1] = 1
        #     # print(np.sum(X),z)
        #     self.testLa = z
            torch.save(self.trainIn.to(torch.device('cpu')), "trainIn.pth")
            torch.save(self.trainBB.to(torch.device('cpu')), "trainBB.pth")
            torch.save(self.trainLa.to(torch.device('cpu')), "trainLa.pth")
            torch.save(self.testIn.to(torch.device('cpu')), "testIn.pth")
            torch.save(self.testBB.to(torch.device('cpu')), "testBB.pth")
            torch.save(self.testLa.to(torch.device('cpu')), "testLa.pth")
        return

    def __len__(self):
        if self.mode == 'train':
            return Trainnum * Trans
        else:
            return Testnum

    def __getitem__(self,idx):
        if self.mode == 'train':
            return self.trainIn[idx+1],self.trainLa[idx+1]
        else:
            return self.testIn[idx+1], self.testLa[idx+1]




            #print(root1)
    ...

    # End of todo


if __name__ == '__main__':

    dataset = TvidDataset(root=r'.\data\tiny_vid', mode='test')
    #import pdb; pdb.set_trace()
