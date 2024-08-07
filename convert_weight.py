import torch
import sys

path1 = sys.argv[1]
split_point = int(sys.argv[2])
branch_num = int(sys.argv[3])
model_name = sys.argv[4]
a = torch.load(path1+'.pt')

model_k = a['model'].state_dict().keys()
state_dict = a['model'].state_dict()
state_dict_new = {}
for i in list(model_k):

    if '.' not in i[6:8] and int(i[6:8]) >= split_point:
        new_i = i[0:6] + str(int(i[6:8]) + branch_num+1) + i[8:]
        state_dict_new[new_i] = state_dict[i].clone()
        print('origin name:{}, shape:{}, update name:{}, shape:{}'.format(
            i, state_dict[i].shape, new_i, state_dict_new[new_i].shape))
    elif '.' in i[6:8] and int(i[6:7]) >= split_point:
        new_i = i[0:6] + str(int(i[6:7]) + branch_num+1) + i[7:]
        state_dict_new[new_i] = state_dict[i].clone()
        print('origin name:{}, shape:{}, update name:{}, shape:{}'.format(
            i, state_dict[i].shape, new_i, state_dict_new[new_i].shape))
    else:
        state_dict_new[i] = state_dict[i].clone()
        print('name:{}, shape:{}'.format(i, state_dict[i].shape))
print(state_dict_new.keys())
torch.save(state_dict_new, path1+'.'+model_name+'.pt')
