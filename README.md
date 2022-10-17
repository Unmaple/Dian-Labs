# Dian-Labs
AIA曾明杰2201智实

-Warning：对应lab在本库的branches里  
-Learning Machinelearning in this Labs  
-学习python基础语法：9.24晚-9.25下午  
初次认识lab0：9.25晚  
学习文件处理和numpy数组：9.26空余时间（全天有课）  
使用knn算法（最终变成了1nn算法，即取最相近的样本对应的label）处理lab0：9.26晚  
修改lab0中出现的bug并且运行识别程序且达到96.31%准确率（10000个数据识别时间50min）：9.27早  
同时写出了一秒可以处理完毕的程序，但仅66%准确率  
学习神经网络的基本结构：9.27下午  
初次认识lab1结构：9.27晚   
学习神经网络层的实现并写出程序结构（需要补充的函数和对应传输的变量类型）：9.28上午  
重构lab1结构并使用微分法进行神经网络的学习：9，28下午，晚上  
使用效率极低的数值微分方法运行了7个epoch并达到41acc：9.29 凌晨  
重构lab1的model结构，实现误差反向传播法，并且debug（最终发现是softmax函数的有一个地方少加了一个log。。。使用数值微分可以学习但是反向运算会出错））：9.29 中午  
lab1运行10000个epoch，达到91.4%并学习了各类超参数的设置方式及初始化的技巧：9.29下午

10.17测试结果：  
同时使用官方提供的baseline和yolov5两条技术路线，其中yolov5路线率先达到0.9正确率（2022.10.17，7：00结果：91.54%）  
yolov5pipeline：  
使用readjson处理赛事提供的数据成yolov  
使用training的训练命令（即yolov5文件里的train.py  
使用training中的测试命令（即yolov5文件里的detect.py  
使用out.py将输出数据格式化成赛事提交的格式  
注意事项：batchsize不能调太大，不仅占用显存大还延缓模型收敛速度  
注意事项：有三个.yaml应当更改，--cfg，--data，--hyp，其中前两个为了改识别物体类别，最后是把图像水平翻转的图像增强去掉  
  
baseline-pipeline  
main.py函数  
（因验证集正确率不理想故没有继续往下做输出） 



10.15早上8点收到测试要求，得知信息：街景字符检测，初步想法：图像检测0-9，按水平坐标拼接起来。  
中午：得到两个技术方向，一个是yolov5为基础的图像检测模型，一个是自己写以torchvision的fasterrcnn为基础的图像检测  
晚上：尝试写数据读入和训练，因dataset不支持不定长的getitem且engine中train_one_epoch 运行出错而不知如何修改而终止，后尝试yolov5（此时关键参数：img320，batch64）并调整--data和--cfg进行训练）  
10.16早：训练完成，输出识别并上传，62.5正确率（现在分析应该是推理的时候选的参数文件选错了，因为那个时候的验证集正确率有80）  
中午：发现天池官方提供了baseline，是以图像识别为基础（resnet18）做的，且长度更小的字符串识别（以X代指没有这个数字）。按照baseline在验证集得到了56%正确率  
下午：参照各种方法（如减少输出字符串长度为4，提高网络复杂度，增加标签平滑，增加dropout层抑制过拟合），最终将正确率提高到64%，判断此方法可能无法在测试结束前优化到90%正确率  
晚上：查阅yolov5训练数据案例，发现有--img默认的（640），没有--cache的，--batch=3，测试了10epoch发现img默认效果良好（现在分析可能是由于图像变大显存放不下，于是将batch从64改成16，反而batch应该是效果最明显的超参数）上传得到的成绩为87.4%。尔后测试25epoch，几个loss下降得很明显，最终成绩是89.74%.最后通宵测试40epoch，同时在--hyp把图像水平翻转去掉了，达到91.4%正确率，正式收工。  
