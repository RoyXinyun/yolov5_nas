import argparse
from pathlib import Path
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from get_flops_feat import get_flops, get_layer_name, get_feat_size, get_feat_size_2

FILE = Path(__file__).resolve()

edge_computation = []

def find_optimal_split(layers, feat, flops, B, edge_computility, cloud_computility=10600):
    # N = 10
    N = len(layers)
    edge_delay = [0] * N
    cloud_delay = [0] * N
    total_delay = [0] * N 

    edge_power = edge_computility  # GFLOPs
    cloud_power = cloud_computility  # GFLOPs

    # compute edge delay
    edge_delay[0] = flops[0] / edge_power
    for i in range(1, N):
        t = flops[i] / edge_power
        edge_delay[i] = edge_delay[i-1] + t

    # compute cloud delay
    cloud_delay[N-1] = 0
    for i in range(N-2, -1, -1):
        t = flops[i+1] / cloud_power
        cloud_delay[i] = cloud_delay[i+1] + t

    # find min split point
    min_total_delay = float('inf')
    best_cut = -1
    min_edge = float('inf')
    min_cloud = float('inf')
    min_trans = float('inf')
    for i in range(N-1):
        trans_delay = feat[i] / B
        total_delay[i] = edge_delay[i] + trans_delay + cloud_delay[i]
        if total_delay[i] < min_total_delay:
            min_total_delay = total_delay[i]
            best_cut = i

            min_edge = edge_delay[i]
            min_trans = trans_delay
            min_cloud = cloud_delay[i]

    return best_cut, min_total_delay, min_edge, min_trans, min_cloud

def read_data(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [[float(i.strip().split(' ')[0]),
             float(i.strip().split(' ')[1])] for i in lines]

def get_bandwidth():
    data = np.array(read_data('./visualization/trace_322.txt'))
    # data[:, 1] /= 1024
    data[:, 1] /= 1000  # Mbps kbps/1000 = Mbps
    bandwidths = data[0:50, 1]
    return bandwidths

def plot_results(x, y, name, edge_computilities):
    plt.xlabel('Bandwidth', fontsize=15)
    plt.ylabel(name, fontsize=15)

    for i in range(len(y)):
        plt.plot(x,y[i],linestyle='-', linewidth=1, label=str(edge_computilities[i])+'GFLOPS')

    plt.xticks(ticks=x, labels=[str(i) for i in x])  # 设置刻度和标签
    plt.legend()
    plt.savefig('results3/{}.png'.format(name))
    plt.clf()


def main():
    per_layer = get_layer_name()
    per_layer_feature = get_feat_size_2()
    print(per_layer_feature)
    per_layer_flops = get_flops()
    # print("flops")
    # print(per_layer_flops)
    # print()
    # bandwidths = get_bandwidth()
    # bandwidths = [0.001, 0.01, 0.1, 1, 10, 40, 70, 100]  # Mbps
    bandwidths = [1, 2, 3, 4, 5, 6, 7, 8, 16, 56, 84]  # Mbps

    # edge_computilities = [0.2, 3, 326.4]  # GFLOPS
    # cloud_computility = 10600 # GFLOPS
    # edge_computilities = [200, 1300, 6000, 11000]  # GFLOPS
    # cloud_computility = 35700 # GFLOPS 3090
    edge_computilities = [1333]  # GFLOPS
    cloud_computility = 19500 # GFLOPS a100
    results = []
    y_split = []
    y_latency = []
    for e in edge_computilities:
        y_sp = []
        y_lat = []
        for b in bandwidths:
            r = {}
            best_cut, min_total_delay, min_edge, min_trans, min_cloud = find_optimal_split(per_layer, per_layer_feature, per_layer_flops, b, e, cloud_computility)
            r['best_cut'] = per_layer[best_cut]    # "{:.3f}".format(num)
            r['min_total_delay'] = "{:.3f}".format(min_total_delay)
            r['bandwidth_MB'] = b
            r['edge_computility'] = e

            r['min_edge'] = "{:.3f}".format(min_edge)
            r['min_trans'] = "{:.3f}".format(min_trans)
            r['min_cloud'] = "{:.3f}".format(min_cloud)
            results.append(r)

            y_sp.append(best_cut)
            y_lat.append(min_total_delay)

        y_split.append(y_sp)
        y_latency.append(y_lat)

    # print(results)


    with open("results3/results.json", 'w') as f:
        f.write(json.dumps(results))
    
    plot_results(bandwidths, y_split, "split", edge_computilities)
    plot_results(bandwidths, y_latency, "latency", edge_computilities)

if __name__ == "__main__":
    main()