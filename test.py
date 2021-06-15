import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt


# with open("/Users/marc-antoine/Desktop/logfile.log") as f:
#     l = f.readlines()
#
# geno = []
# flag = 0
# for i in l:
#     if 'genotype' in i:
#         if flag % 3 == 0:
#             geno.append(i)
#         flag += 1
# reduce = []
# normal = []
# for g in geno:
#     index = g.index("reduce")
#     normal.append(g[:index])
#     reduce.append(g[index:])
#
# lables = ['max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5']
#
#
# for step, n in enumerate(reduce):
#     data = []
#     for l in lables:
#         data.append(n.count(l))
#     p = plt.figure(figsize=(10, 10), dpi=80)
#     plt.bar(lables, data)
#     plt.title("Reduce epoch: {}".format(step))
#     plt.savefig("/Users/marc-antoine/Desktop/reduce/{}.png".format(step))
#     plt.close(p)

# import imageio
#
#
# def create_gif(image_list, gif_name, duration=0.35):
#     frames = []
#     for image_name in image_list:
#         frames.append(imageio.imread(image_name))
#     imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
#     return
#
#
# def main():
#     os.chdir("/Users/marc-antoine/Desktop/reduce/")
#     image_list = os.listdir()
#     image_list.sort()
#     image_list.remove('.DS_Store')
#     print(image_list)
#     gif_name = 'reduce.gif'
#     duration = 0.35
#     create_gif(image_list, gif_name, duration)
#
#
# if __name__ == '__main__':
#     main()
print(3//4)
