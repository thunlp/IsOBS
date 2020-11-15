from tqdm import tqdm

from dataloader import *
from source.PrototypicalLoss import euclidean_dist

from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn.functional as F
import torch

def get_representation(opt,test_dataloader,model,full_size):
    model.eval()
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    test_iter_c=iter(test_dataloader)
    model_output=torch.empty(min(full_size,100),576)
    first=True
    for batch_c in test_iter_c:
        x_c=batch_c
        # print(x_c.size())
        x_c=x_c.to(device)
        if first:
            model_output=model(x_c).detach().to('cpu')
            first=False
        else:
            model_output=torch.cat((model_output,model(x_c).detach().to('cpu')),dim=0)
    # print("model_output",model_output.size())
    support_cpu=model_output.to('cpu')
    prototypes=support_cpu.mean(0)
    # print("prototpyes",prototypes.size())
    return prototypes

def test_model(options,model,n_classes):
    # 获取支撑集
    #n_classes=361;  # 如果跑oracle300就是361，oracle600是617.oracle1600是1621
    test_dataset_c = datasets.ImageFolder("./data_oct/train_"+options.oracle,transform=transform)  # train_300/train_600/train_1600
    prototypes=torch.empty(n_classes,576)
    head=0
    tail=0
    num=0
    while num<n_classes:
        while tail<len(test_dataset_c) and test_dataset_c[tail][1]==num:
            tail+=1
        dataset_c=torch.empty(tail-head,3,50,50)
        for i in range(head,tail):
            dataset_c[i-head]=test_dataset_c[i][0].unsqueeze(0)
        prototypes[num]=get_representation(options,DataLoader(dataset_c,batch_size=600),model,len(dataset_c))
        head=tail
        num+=1
    # print("prototypes",prototypes.size())

    # 做测试
    model.eval()
    device = 'cuda:0' if torch.cuda.is_available() and options.cuda else 'cpu'
    test_dataset = datasets.ImageFolder("./data_oct/test_"+options.oracle,transform=transform)  # test_300/test_600/test_1600
    correct=0
    correct_3=0
    correct_5=0
    correct_10=0
    whole_num=len(test_dataset)
    test_dataloader=DataLoader(test_dataset,batch_size=600)
    test_iter=iter(test_dataloader)
    # 因为CUDA Memory有限，所以分批处理，最后再合起来
    for batch in tqdm(test_iter):
        x, y = batch  # x:数据，y:label
        x = x.to(device)
        # print("x",x.size())
        model_output=model(x).detach().to('cpu')
        answer_cpu=y
        dists=euclidean_dist(model_output,prototypes)
        # print("dists",dists.size())
        # Top1准确率
        log_p_y=F.log_softmax(-dists,dim=1).max(1)
        # print(answer_cpu.size(),log_p_y[1].size())
        correct+=answer_cpu.eq(log_p_y[1]).sum()  # [1]表示取得是index，因为第0维是value
        # Topk准确率
        k=3
        _,log_p_y_k=torch.topk(F.log_softmax(-dists,dim=1),k=k,dim=1)
        answer_cpu=answer_cpu.view(-1,1)  # 从[n]调整为[n,1]，就可以和[n,k]的log_p_y_k比较了
        correct_3+=(answer_cpu==log_p_y_k).sum().item()
        k=5
        _,log_p_y_k=torch.topk(F.log_softmax(-dists,dim=1),k=k,dim=1)
        answer_cpu=answer_cpu.view(-1,1)  # 从[n]调整为[n,1]，就可以和[n,k]的log_p_y_k比较了
        correct_5+=(answer_cpu==log_p_y_k).sum().item()
        k=10
        _,log_p_y_k=torch.topk(F.log_softmax(-dists,dim=1),k=k,dim=1)
        answer_cpu=answer_cpu.view(-1,1)  # 从[n]调整为[n,1]，就可以和[n,k]的log_p_y_k比较了
        correct_10+=(answer_cpu==log_p_y_k).sum().item()

    print("result: {} correct in {}. (accuracy: {}%)".format(correct,whole_num,correct*100/float(whole_num)))
    print("top-{}: {} correct in {}. (accuracy: {}%)".format(3,correct_3,whole_num,correct_3*100/float(whole_num)))
    print("top-{}: {} correct in {}. (accuracy: {}%)".format(5,correct_5,whole_num,correct_5*100/float(whole_num)))
    print("top-{}: {} correct in {}. (accuracy: {}%)".format(10,correct_10,whole_num,correct_10*100/float(whole_num)))