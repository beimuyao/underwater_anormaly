"""
inference GANOMALY

. Example: Run the following command from the terminal.
    run inference.py                             \
        --model ganomaly                        \
        --dataset UCSD_Anomaly_Dataset/UCSDped1 \
        --batchsize 32                          \
        --isize 256                         \
        --nz 512                                \
        --ngf 64                               \
        --ndf 64
"""
# from __future__ import print_function
import numpy as np
import pandas as pd
import os
import torch
import csv
from options import Options
from lib.networks import NetG
from nauta.dataset import get_dataset
from datetime import datetime
from lib.evaluate import roc

def evaluate(gt, pred):
    """
    计算二分类的准确率、精确率、召回率
    gt: list 或 numpy array, 真实标签 (0/1)
    pred: list 或 numpy array, 预测标签 (0/1)
    """
    gt = np.array(gt)
    pred = np.array(pred)

    TP = np.sum((gt == 1) & (pred == 1))
    TN = np.sum((gt == 0) & (pred == 0))
    FP = np.sum((gt == 0) & (pred == 1))
    FN = np.sum((gt == 1) & (pred == 0))

    accuracy = (TP + TN) / len(gt)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"TP": TP, "TN": TN, "FP": FP, "FN": FN,
            "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1}

def inference(dataloader, gt, device, netG, threshold=None):
    anomaly_scores = []
    filenames = []
    with torch.no_grad():
        for data in dataloader['test']:
            # 假设 dataset 返回 (input, filename) 或 (input, label, filename)
            if len(data) == 3:
                inputs,labels, fnames = data
            else:
                inputs = data[0]
                fnames = [f"sample_{i}" for i in range(inputs.size(0))]

            inputs = inputs.to(device)

            fake, latent_i, latent_o = netG(inputs)
            # 异常分数
            score = torch.mean((latent_i - latent_o) ** 2, dim=1)
            anomaly_scores.append(score.cpu())
            filenames.extend(fnames)

    anomaly_scores = torch.cat(anomaly_scores, dim=0).numpy()
    
    # 如果没有给阈值，用正常样本分布自动设置
    if threshold is None:
        score_list = anomaly_scores.squeeze().tolist()
        best_roc, threshold = roc( gt,score_list)
        print("Auto threshold set to:", threshold)
        

    # 判定正常 / 异常
    results = []
    for fname, score in zip(filenames, anomaly_scores):
        label = 1 if score > threshold else 0
        results.append({'filename': fname, 'score': float(score), 'label': label})
  

    return results

def main():
    #参数
    opt = Options().parse() 
    device = torch.device("cuda:0" if opt.device == 'gpu' else "cpu")
   
    # LOAD DATA
    dataloader = get_dataset(opt) 
    metadata = pd.read_csv(opt.test_metadata)
    gt = list(metadata.label)
    gt = [1 if x != 0 else 0 for x in gt]
    # LOAD MODEL
    netg = NetG(opt).to(device)

    with torch.no_grad():
        # Load the weights of netg and netd.
        # if opt.load_weights:
        if True:
            # path = "./output/{}/{}/train/weights/netG.pth".format(self.name.lower(), self.opt.dataset)
            pretrained_dict = torch.load(opt.model_path)['state_dict']

            try:
                netg.load_state_dict(pretrained_dict)
            except IOError:
                raise IOError("netG weights not found")
            print('   Loaded weights.')

    netg.eval()
    
    # #推理
    results = inference(dataloader,gt, device, netg)

    #保存结果
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(opt.outf,'test',time_str)
    if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
    csv_name = os.path.join(out_dir, 'inference_results.csv')
    file_name = os.path.join(out_dir, 'result.txt')
    os.makedirs(opt.outf, exist_ok=True)
    with open(csv_name, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'score', 'label'])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print("Inference done. Results saved to", csv_name)

    metadata = pd.read_csv(csv_name)
    pred = list(metadata.label)
    res = evaluate(gt, pred)
    print(res)
    with open(file_name,'w',newline='') as f:
        for k,v in res.items():
            f.write(f"{k}:{v}\n")

    

if __name__ == '__main__':
    main()