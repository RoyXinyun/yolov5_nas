from matplotlib import pyplot as plt
import numpy as np


def linear():
    x = [1, 0.81, 0.64, 0.49, 0.36, 0.25]
    y = [0.552, 0.506, 0.494, 0.469, 0.431, 0.365]
    pow_n = 1
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, pow_n))(np.unique(x)))
    print(np.poly1d(np.polyfit(x, y, pow_n)))
    plt.scatter(x, y, c='g')
    plt.savefig('p_acc.png')


def read_data(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [[float(i.strip().split(' ')[0]),
             float(i.strip().split(' ')[1])] for i in lines]


def get_acc(x):
    return 0.2202 * x + 0.3392


def get_params(bw, limit=70):
    size_o = 80 * 80 * 34 / 8 / 1024 / 1024  # 0.025MB
    size_p = (bw / 8) * (limit - 47.2) / 1000  # kps/8/1024 = MB/s
    return min(size_p / size_o, 1)


def plot(x, y, index, name):
    y_label = ['Bandwidth(Mbps)', 'Parameters', 'Accuracy', 'FPS']
    color = ['r', 'g', 'b', 'y']

    for i in range(len(y)):
        plt.xlabel('Timestamp', fontsize=15)
        plt.ylabel(y_label[index], fontsize=15)
        plt.plot(x, y[i], c=color[index])
        plt.savefig('scheduler/{}_{}.png'.format(
            y_label[index], name))
        plt.clf()


def get_xy(acc_limit, lantancy_limit):
    t = []
    bw = []
    param = []
    acc = []
    fps = []
    data = np.array(read_data('trace_322.txt'))
    data[:, 1] /= 1024
    data[:, 1] /= 8
    for i in data:
        t.append(i[0])
        bw.append(i[1])
        param_ = get_params(i[1], lantancy_limit)
        acc_ = get_acc(param_)
        if acc_ >= 0.552 * acc_limit:
            param.append(param_)
            acc.append(acc_)
            # print(param_*80*80*34*1000/1024/1024/i[1])
            fps.append(
                1000 /
                (47.2 + param_ * 80 * 80 * 34 * 1000 / 1024 / 1024 / i[1]))
        else:
            param.append(None)
            acc.append(None)
            fps.append(None)
    return t, bw, param, acc, fps


acc_limits = [0.5, 0.7]
lantancy_limits = [60, 80, 100]
for acc in acc_limits:
    for lantancy in lantancy_limits:
        t, bw, param, acc_, fps = get_xy(acc, lantancy)
        plot(t, [bw], 0, str(acc) + '_' + str(lantancy))
        plot(t, [param], 1, str(acc) + '_' + str(lantancy))
        plot(t, [acc_], 2, str(acc) + '_' + str(lantancy))
        plot(t, [fps], 3, str(acc) + '_' + str(lantancy))
