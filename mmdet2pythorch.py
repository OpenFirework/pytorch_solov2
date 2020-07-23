import torch

model_old = torch.load("./pretrained/SOLOv2_LIGHT_448_R34_3x.pth")

model_old_dict = model_old['state_dict']

model_new = torch.load('solov2_new.pth')
old_keys = []
for key in list(model_old_dict.keys()):
    old_keys.append(key)

index = 0
for key in list(model_new.keys()):
    old_key = old_keys[index]
    print(key,old_key)
    model_new[key] = model_old_dict[old_key]
    index = index + 1
torch.save(model_new,"./pretrained/solov2_448_r34_epoch_36.pth")
