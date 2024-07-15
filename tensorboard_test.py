from torch.utils.tensorboard import SummaryWriter
import numpy as np

# 示例数据
epochs = 50
train_loss = np.random.rand(epochs)
train_acc = np.random.rand(epochs)
val_loss = np.random.rand(epochs)
val_acc = np.random.rand(epochs)
learning_rate = np.linspace(0.01, 0.001, epochs)  # 示例学习率

# 创建日志目录
writer_train = SummaryWriter('runs/experiment_1/train')
writer_val = SummaryWriter('runs/experiment_1/val')
# 写入日志
for epoch in range(epochs):
    writer_train.add_scalar('Loss', train_loss[epoch], epoch)
    writer_val.add_scalar('Loss', val_loss[epoch], epoch)
    writer_train.add_scalar('Accuracy', train_acc[epoch], epoch)
    writer_val.add_scalar('Accuracy', val_acc[epoch], epoch)
    writer_train.add_scalar('Learning Rate', learning_rate[epoch], epoch)

writer_train.close()
writer_val.close()
