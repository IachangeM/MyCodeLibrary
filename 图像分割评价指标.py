
"""
接收一通道的分割图像数据(list)--需要保证对应一致，计算评价指标
默认已经根据threshold调整好了GT_predict图像

"""

import numpy as np
from PIL import Image


class Evaluation(object):
    def __init__(self, gtlist, problist, precision=5, threshold=0):
        # TODO: precision参数设置小数点后保留几位
        self.precision = precision
        # np.set_printoptions(precision=precision)

        self.gtlist = gtlist
        self.problist = problist
        self.threshold = threshold

        self._np_gtlist = []
        self._np_probls = []

        self._initialize()

        # TODO: 注册相同计算 不同叫法的函数
        self.get_recall = self.get_sensitivity
        self.get_TPR = self.get_sensitivity
        self.get_FPR = self.get_specificity

    def _initialize(self):
        for gt, gtprob in zip(self.gtlist, self.problist):
            self._np_gtlist.append(
                np.asarray(Image.open(gt), dtype=np.uint8)
            )
            self._np_probls.append(
                np.asarray(Image.open(gtprob), dtype=np.uint8)
            )

        try:
            assert np.min(self._np_probls) <= self.threshold <= np.max(self._np_probls)
            # TODO: 全部归一化到0, 1
            self._np_gtlist = list(map(lambda _: _ > 0, self._np_gtlist))
            self._np_probls = list(map(lambda _: _ > self.threshold, self._np_probls))
        except ValueError:
            ValueError("图像大小不一致")

    def get_accuracy(self, need_all=False):
        """
        正确率公式： AC = (TP+TN)/(TP+TN+FP+FN)
        # TP : True Positive , TN: True Negative, FP: False Positive, FN : False Negative,
        :return:预测结果和分割图一致的点的数目/全部的像素点数
        """
        all_points = np.prod(self._np_gtlist[0].shape)
        acc_list = []

        for gt, gtprob in zip(self._np_gtlist, self._np_probls):
            corr = np.sum(gt == gtprob)
            acc = corr*1.0/all_points
            acc_list.append(round(float(acc), self.precision))

        acc = round(float(np.mean(acc_list)), self.precision)
        if need_all:
            return acc, acc_list
        else:
            return acc

    def get_precision(self, need_all=False):
        """
        查准率公式： TP/(TP + FP)
        # TP : True Positive, FP : False Positive
        :param need_all: 为真的时候, 返回全部图像的对应测试结果
        :return:计算查准率
        """
        precision_list = []
        for gt, gtprob in zip(self._np_gtlist, self._np_probls):
            _TP = ((gtprob == 1)*1 + (gt == 1)*1) == 2      # TODO: 注意乘以1,不然两个张量相加还只是True/False 布尔值
            _FP = ((gtprob == 1)*1 + (gt == 0)*1) == 2
            _PC = float(np.sum(_TP)) / (float(np.sum(_TP + _FP)) + 1e-6)
            precision_list.append(round(float(_PC), self.precision))

        precision = round(float(np.mean(precision_list)), self.precision)
        if need_all:
            return precision, precision_list
        else:
            return precision

    def get_sensitivity(self, need_all=False):
        """
        查准率/召回率/TPR【真正例率】/敏感性 的计算公式： SE = TPR = TP/(TP + FN)
        # TP : True Positive , FN : False Negative
        :return:计算查全率/召回率==recall
        """
        recall_list = []
        for gt, gtprob in zip(self._np_gtlist, self._np_probls):
            _TP = ((gtprob == 1)*1 + (gt == 1)*1) == 2      # TODO: 注意乘以1,不然两个张量相加还只是True/False 布尔值
            _FN = ((gtprob == 0)*1 + (gt == 1)*1) == 2
            _SE = 1.0*np.sum(_TP) / (float(np.sum(_TP + _FN)) + 1e-6)
            recall_list.append(round(float(_SE), self.precision))

        recall = round(float(np.mean(recall_list)), self.precision)
        if need_all:
            return recall, recall_list
        else:
            return recall

    def get_specificity(self, need_all=False):
        """
        特异性specificity/假正例率FPR  的计算公式： SP = FPR = FP/(TN + FP)
        # TN : True Negative, FP : False Positive
        :param need_all: 为真的时候, 返回全部图像的对应测试结果
        :return:计算特异性,或者说假正例率
        """
        specificity_list = []
        for gt, gtprob in zip(self._np_gtlist, self._np_probls):
            _TN = ((gtprob == 0)*1 + (gt == 0)*1) == 2      # TODO: 注意乘以1,不然两个张量相加还只是True/False 布尔值
            _FP = ((gtprob == 1)*1 + (gt == 0)*1) == 2
            _SP = float(np.sum(_TN)) / (float(np.sum(_TN + _FP)) + 1e-6)
            specificity_list.append(round(float(_SP), self.precision))

        specificity = round(float(np.mean(specificity_list)), self.precision)
        if need_all:
            return specificity, specificity_list
        else:
            return specificity


if __name__ == '__main__':
    from glob import glob

    gtlist = sorted(glob("./groundtruth/*.gif"))
    gtproblist = sorted(glob("./result/*.gif"))

    evalution = Evaluation(gtlist, gtproblist)
    print(evalution.get_precision(need_all=True))

