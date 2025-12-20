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
import os
import torch
import csv
from options import Options
from lib.networks import NetG
from nauta.dataset import get_dataset


def inference(dataloader, device, netG, threshold=None):
    anomaly_scores = []
    filenames = []

    with torch.no_grad():
        for data in dataloader:
            # 假设 dataset 返回 (input, filename) 或 (input, label, filename)
            if len(data) == 2:
                inputs, fnames = data
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
        mean = np.mean(anomaly_scores)
        std = np.std(anomaly_scores)
        threshold = mean + 3 * std
        print("Auto threshold set to:", threshold)

    # 判定正常 / 异常
    results = []
    for fname, score in zip(filenames, anomaly_scores):
        label = 'abnormal' if score > threshold else 'normal'
        results.append({'filename': fname, 'score': float(score), 'label': label})

    return results

def main():
    #参数
    opt = Options().parse() 
    device = torch.device("cuda:0" if opt.device == 'gpu' else "cpu")
    # LOAD DATA

    ######dataloader有问题 返回的是train和test


    dataloader = get_dataset(opt) 
    # LOAD MODEL
    netg = NetG(opt).to(device)

    if opt.load_weights:
        checkpoint = torch.load(opt.model_path, map_location=device)
        netg.load_state_dict(checkpoint['state_dict'])
        print("Loaded trained weights from:", opt.model_path)

    netg.eval()
    
    #推理
    results = inference(dataloader, device, netg, threshold=opt.threshold)

    #保存结果
    csv_file = os.path.join(opt.outf, 'inference_results.csv')
    os.makedirs(opt.outf, exist_ok=True)
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'score', 'label'])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print("Inference done. Results saved to", csv_file)

if __name__ == '__main__':
    main()