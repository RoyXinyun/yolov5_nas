from matplotlib import pyplot as plt
import matplotlib.patches as mpatches


def process(output_feat_size):
    results = []
    for i, value in enumerate(output_feat_size):
        if i <= 4:
            results.append(value)
        elif i <= 6:
            results.append(value + output_feat_size[4])
        elif i <= 10:
            results.append(value + output_feat_size[4] + output_feat_size[6])
        elif i == 11:
            results.append(value + output_feat_size[4] + output_feat_size[6] +
                           output_feat_size[10])
        elif i <= 14:
            results.append(value + output_feat_size[4] + output_feat_size[10])
        elif i == 15:
            results.append(value + output_feat_size[4] + output_feat_size[10] +
                           output_feat_size[14])
        elif i <= 17:
            results.append(value + output_feat_size[10] + output_feat_size[14])
        elif i == 18:
            results.append(value + output_feat_size[17] +
                           output_feat_size[10] + output_feat_size[14])
        elif i <= 20:
            results.append(value + output_feat_size[17] + output_feat_size[10])
        elif i == 21:
            results.append(value + output_feat_size[17] +
                           output_feat_size[10] + output_feat_size[20])
        elif i < 24:
            results.append(value + output_feat_size[17] + output_feat_size[20])
    return results


def show_split_size(output_feat_size, layer_name):
    y = process(output_feat_size)
    y = [3 * 640 * 640] + y
    y = [i * 4 for i in y]

    color_map = [['#4F7479'], ['#412B3C'], ['#C55544'], ['#EAD5BA']]
    color = color_map[0] * 6 + color_map[1] * 2 + \
        color_map[2] * 4 + color_map[3] + \
        color_map[2] * 3 + color_map[3] + \
        color_map[2] * 2 + color_map[3] + \
        color_map[2] * 2 + color_map[3] + \
        color_map[2] * 2

    plt.figure(figsize=(30, 15), dpi=200)
    plt.xlabel('Layer Name', fontsize=20)
    plt.ylabel('Data Size(B)', fontsize=20)

    plt.bar(layer_name, y, color=color)
    labels = ['Single Branch', 'Double Branch', 'Three Branche', 'Four Branch']
    patches = [
        mpatches.Patch(color=color_map[i][0], label="{:s}".format(labels[i]))
        for i in range(len(color_map))
    ]

    plt.legend(handles=patches, loc='upper right', fontsize=20)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14, rotation=300)
    plt.savefig('split_feat_size.png')


def show_output_size(output_feat_size, layer_name):
    y = output_feat_size
    y = [3 * 640 * 640] + y
    y = [i * 4 for i in y]

    plt.figure(figsize=(30, 15), dpi=200)
    plt.xlabel('Layer Name', fontsize=30)
    plt.ylabel('Data Size(B)', fontsize=30)
    plt.bar(layer_name, y, color='#4F7479')

    plt.xticks(fontsize=14, rotation=300)
    plt.savefig('feat_size.png')

def show_output_flops(output_feat_size, layer_name):
    y = output_feat_size

    params = []
    flops = []
    for x in y:
        params.append(x[0])
        flops.append(x[1])
    print(len(flops),len(layer_name))
    plt.figure(figsize=(30, 15), dpi=200)
    plt.xlabel('Layer Name', fontsize=30)
    plt.ylabel('MACs(MMACs)', fontsize=30)
    plt.bar(layer_name, flops, color='#4F7479')

    plt.xticks(fontsize=14, rotation=300)
    plt.savefig('flops_size.png')

def show_output_params(output_feat_size, layer_name):
    y = output_feat_size

    params = []
    flops = []
    for x in y:
        params.append(x[0])
        flops.append(x[1])
    print(len(flops),len(layer_name))
    plt.figure(figsize=(30, 15), dpi=200)
    plt.xlabel('Layer Name', fontsize=30)
    plt.ylabel('Params(k)', fontsize=30)
    plt.bar(layer_name, params, color='#4F7479')

    plt.xticks(fontsize=14, rotation=300)
    plt.savefig('params_size.png')

if __name__ == '__main__':
    output_feat_size = [
        32 * 320 * 320,
        32 * 320 * 320,  # 0, 1
        64 * 160 * 160,
        128 * 80 * 80,  # 2, 3
        128 * 80 * 80,
        256 * 40 * 40,  # 4, 5
        256 * 40 * 40,
        512 * 20 * 20,  # 6,7
        512 * 20 * 20,
        512 * 20 * 20,  # 8, 9
        512 * 20 * 20,
        256 * 40 * 40,  # 10, 11
        512 * 40 * 40,
        256 * 40 * 40,  # 12, 13
        128 * 40 * 40,
        128 * 80 * 80,  # 14, 15
        256 * 80 * 80,
        128 * 80 * 80,  # 16, 17
        128 * 40 * 40,
        256 * 40 * 40,  # 18, 19
        128 * 40 * 40,
        512 * 20 * 20,  # 20, 21
        512 * 20 * 20,
        512 * 20 * 20,  # 22, 23
    ]

    output_layer_size = [   # params k MACs M
        [3.52, 360.45],
        [18.56, 475.14], # 1
        [18.82, 481.69],
        [73.98, 473.5], # 3
        [115.71, 740.56],
        [295.42, 472.68], # 5
        [625.15, 1000],
        [1180, 472.27], # 7
        [1.18, 473.09],
        [656.9, 263.07], # 9
        [131.58, 52.63],
        [0, 0.4096], # 11
        [0, 0],
        [361.98, 579.17], # 13
        [33.02, 52.84],
        [0, 0.819], # 15
        [0, 0],
        [90.88, 581.63], # 17
        [147.71, 236.34],
        [0 ,0], # 19
        [296.45, 474.32],
        [590.34, 236.13], # 21
        [0, 0],
        [1180, 473.09], # 23
        [229.25, 367.71]
    ]

    layer_name = [
        'Conv_0', 'Conv_1', 'C3_2', 'Conv_3', 'C3_4', 'Conv_5',
        'C3_6', 'Conv_7', 'C3_8', 'SPPF_9', 'Conv_10', 'Upsample_11',
        'Concat_12', 'C3_13', 'Conv_14', 'Upsample_15', 'Concat_16', 'C3_17',
        'Conv_18', 'Concat_19', 'C3_20', 'Conv_21', 'Concat_22', 'C3_23', 'Detect_24'
    ]

    # show_split_size(output_feat_size, layer_name)
    # show_output_size(output_feat_size, layer_name)
    show_output_flops(output_layer_size, layer_name)
    show_output_params(output_layer_size, layer_name)