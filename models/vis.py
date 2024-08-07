import matplotlib.pyplot as plt
import numpy as np
import cv2


def vis_feature_dis(inputs):
    """
    Visualize the distribution of feature map
    """
    feature = inputs.cpu().clone().detach().numpy()[0]
    for i in range(feature.shape[0]):
        plt.figure(i)
        feat = feature[i].flatten()*100
        # feat = feat[feat != 0]

        plt.hist(feat, bins=20)
        plt.title('Feature map {}'.format(i))
        plt.show()
        plt.savefig('feature_map_dis/{}.png'.format(str(i)))


def vis_feature(inputs, name='input'):
    """
    Visualize feature map
    """
    feature = inputs.cpu().clone().detach().numpy()[0]
    for i in range(feature.shape[0]):
        tmp = feature[i]
        # tmp *= 255
        # tmp = cv2.applyColorMap(np.uint8(tmp), cv2.COLORMAP_JET)
        plt.imshow(tmp, cmap='gray')
        plt.axis('off')
        plt.imsave('feature_map/{}_{}.png'.format(i, name), tmp)
        # cv2.imwrite('feature_map/{}_{}.png'.format(i, name), tmp)
