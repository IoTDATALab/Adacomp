import numpy as np
from prettytable import PrettyTable
from matplotlib import pyplot as plt

class ConfusionMatrix(object):

    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))  # 初始化混淆矩阵，元素都为0
        self.num_classes = num_classes  # 类别数量，本例数据集类别为5
        self.labels = labels  # 类别标签

    def update(self, preds, labels):
        for p, t in zip(preds, labels):  # pred为预测结果，labels为真实标签
            self.matrix[p, t] += 1  # 根据预测结果和真实标签的值统计数量，在混淆矩阵相应位置+1

    def summary(self):  # 计算指标函数
        # calculate accuracy
        sum_TP = 0
        # 计算测试样本的总数
        n = np.sum(self.matrix)
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]  # 混淆矩阵对角线的元素之和，也就是分类正确的数量
        acc = sum_TP / n  # 总体准确率
        print("the model accuracy is ", acc)

        # kappa
        sum_po = 0
        sum_pe = 0
        for i in range(len(self.matrix[0])):
            sum_po += self.matrix[i][i]
            row = np.sum(self.matrix[i, :])
            col = np.sum(self.matrix[:, i])
            sum_pe += row * col
        po = sum_po / n
        pe = sum_pe / (n * n)
        # print(po, pe)
        kappa = round((po - pe) / (1 - pe), 3)
        # print("the model kappa is ", kappa)

        # precision, recall, specificity
        table = PrettyTable()  # 创建一个表格
        table.field_names = ["", "Accuracy", "Precision", "Recall", "F1-score"]
        Precision, Recall, Specificity, F1_score = 0, 0, 0, 0
        for i in range(self.num_classes):  # 精确度、召回率、特异度的计算
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            precision = round(TP / (TP + FP), 6) if TP+ FP != 0 else 0.
            recall = round(TP / (TP + FN), 6) if TP + FN != 0 else 0.  # 每一类准确度
            specificity = round(TN / (TN + FP), 6) if TN + FP != 0 else 0.
            f1_score = round(2*(precision*recall)/(precision+recall), 6)
            Precision += precision
            Recall += recall
            Specificity += specificity
            F1_score += f1_score
        table.add_row(["value", acc, Precision/self.num_classes, Recall/self.num_classes, F1_score/self.num_classes])
        print(table)
        # return str(acc)
        return acc, Precision/self.num_classes, Recall/self.num_classes, F1_score/self.num_classes

    def plot(self):  # 绘制混淆矩阵
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix (acc=' + self.summary() + ')')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()

# class_indict = config.tomato_DICT
# #tomato_DICT = {'0': 'Bacterial_spot', '1': 'Early_blight', '2': 'healthy', '3': 'Late_blight', '4': 'Leaf_Mold'}
# # 标签名字列表
# confusion = ConfusionMatrix(num_classes=config.NUM_CLASSES, labels=label)
# #实例化混淆矩阵，这里NUM_CLASSES = 5
#
# with torch.no_grad():
#     model.eval()#验证
#     for j, (inputs, labels) in enumerate(val_data):
#         inputs = inputs.to(device)
#         labels = labels.to(device)
#         output = model(inputs)#分类网络的输出，分类器用的softmax,即使不使用softmax也不影响分类结果。
#         loss = loss_function(output, labels)
#         valid_loss += loss.item() * inputs.size(0)
#         ret, predictions = torch.max(output.data, 1)#torch.max获取output最大值以及下标，predictions即为预测值（概率最大），这里是获取验证集每个batchsize的预测结果
#                 #confusion_matrix
#         confusion.update(predictions.cpu().numpy(), labels.cpu().numpy())
#         confusion.plot()
#         confusion.summary()


