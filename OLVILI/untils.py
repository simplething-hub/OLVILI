import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from args import parameter_parser
config = {
    "font.family": 'Times New Roman',  # 设置字体类型
    "axes.unicode_minus": False  # 解决负号无法显示的问题
}
plt.rcParams.update(config)
args = parameter_parser()

def read_losses(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        losses = []
        for line in lines:
            # 移除字符串两端的空白字符，包括空格和换行符
            clean_line = line.strip()
            # 分割字符串，并移除每个部分末尾的逗号
            values = clean_line.split(', ')
            for value in values:
                # 移除每个值末尾的逗号（如果有）
                value = value.rstrip(',')
                # 将分割后的浮点数添加到损失列表中
                losses.append(float(value))
        return losses
    #COIL,
# 100leaves,Hdigit,handwritten,HW,Mfeat,UCI
dataset = 'Hdigit'
filename_loss = f'D:/codes/python/Attention_for_MvClustering/losses/{dataset}_loss1.txt'
losses = read_losses(filename_loss)
n = args.epoch

# 将损失值分组为三个一组
sr_losses = losses[::3]  # 每三个值中的第一个
knn_losses = losses[1::3]  # 每三个值中的第二个
l1_norms = losses[2::3]  # 每三个值中的第三个

# 确保每个变量中都有n个损失值
sr_losses = sr_losses[:n]
knn_losses = knn_losses[:n]
l1_norms = l1_norms[:n]

# 归一化损失值
scaler = MinMaxScaler(feature_range=(0, 1))
sr_losses_normalized = scaler.fit_transform(np.array(sr_losses).reshape(-1, 1)).flatten()
knn_losses_normalized = scaler.fit_transform(np.array(knn_losses).reshape(-1, 1)).flatten()
l1_norms_normalized = scaler.fit_transform(np.array(l1_norms).reshape(-1, 1)).flatten()

fontdict = {'family': 'Times New Roman', 'size': 30}

plt.figure(figsize=(11, 8), dpi=80)
plt.subplots_adjust(left=0.15, bottom=0.2, right=0.85, top=0.95)
x = np.arange(1, n+1)  # 生成1到n的x轴数据

# 主y轴
ax = plt.axes()
color = 'tab:blue'
ax.spines['right'].set_visible(False)
ax.set_ylabel('Loss Value', fontdict=fontdict)
ax.plot(x, sr_losses_normalized, color=color, linewidth=2, label='loss1')
ax.tick_params(axis='y', labelsize=20)

# 第二个y轴
ax2 = ax.twinx()
color = 'tab:green'
ax2.plot(x, knn_losses_normalized, color=color, linewidth=2, label='loss2')
ax2.tick_params(axis='y', labelsize=20)

# 第三个y轴
ax3 = ax.twinx()
color = 'tab:red'
ax3.plot(x, l1_norms_normalized, color=color, linewidth=2, label='loss3')
ax3.tick_params(axis='y', labelsize=20)
ax3.set_yticks([])  # 去掉最右边y轴的刻度

ax.set_xlabel('The number of iterations', fontdict=fontdict)
ax.tick_params(axis='x', labelsize=20)

# 添加网格线（虚线）
ax.grid(True, linestyle='--', which='both', color='gray', alpha=0.5)
ax2.grid(True, linestyle='--', which='both', color='gray', alpha=0.5)
ax3.grid(True, linestyle='--', which='both', color='gray', alpha=0.5)

# 添加图例
'''lines = [line for ax_ in [ax, ax2, ax3] for line in ax_.get_lines()]
labels = [line.get_label() for line in lines]
plt.legend(lines, labels, loc='upper right', prop=fontdict)
'''
plt.xlim(0, n)
plt.savefig(f'loss{dataset}.svg')
plt.show()
