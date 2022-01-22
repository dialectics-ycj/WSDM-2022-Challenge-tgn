import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score

from utils.get_predict_ans import get_predict_auc


class BaseRecoder:
    def __init__(self, logger=None, prefix=None, figs_dir=None, fig_infos_dir=None, title=None):
        self.logger = logger
        self.prefix = prefix
        self.figs_dir = figs_dir
        self.fig_infos_dir = fig_infos_dir
        self.title = title
        pass

    def clear(self):
        pass

    def draw(self):
        pass

    def record(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass


class PredictRecoder(BaseRecoder):
    def __init__(self, data_name, model_idx, logger, prefix, figs_dir, fig_infos_dir, title, version="normal"):
        super().__init__(logger, prefix, figs_dir, fig_infos_dir, title)
        self.data_name = data_name
        self.model_idx = model_idx
        self.scores_dict = defaultdict(list)
        self.aucs_dict = defaultdict(list)
        self.version = version

    def draw(self):
        fig, axs = plt.subplots(1, 2, figsize=(12.8, 4.8))
        # mean score on predict set
        axs[0].set_title("mean_score")
        element_num = 0
        for i, key in enumerate(self.scores_dict.keys()):
            axs[0].plot(key, data=self.scores_dict)
            element_num = len(self.scores_dict[key])
        axs[0].legend()
        if element_num < 15:
            x_major_locator = MultipleLocator(1)
            axs[0].xaxis.set_major_locator(x_major_locator)

        # auc on predict set
        axs[1].set_title("auc")
        for i, key in enumerate(self.aucs_dict.keys()):
            axs[1].plot(key, data=self.aucs_dict)
        if element_num < 15:
            x_major_locator = MultipleLocator(1)
            axs[0].xaxis.set_major_locator(x_major_locator)
        axs[1].legend()
        fig.suptitle(self.title)
        fig.savefig(os.path.join(self.figs_dir, self.title))
        plt.close(fig)

    def log(self):
        for name, _ in self.scores_dict.items():
            self.logger.info('{}-{} mean score: {}, auc: {}'.format(self.prefix, name, self.scores_dict[name][-1],
                                                                    self.aucs_dict[name][-1]))
        self.draw()

    def clear(self):
        self.scores_dict = defaultdict(list)
        self.aucs_dict = defaultdict(list)

    def record(self, predict_pros_dict, output_path_manager):
        for name, probs in predict_pros_dict.items():
            predict_aucs = get_predict_auc(self.data_name, probs, model_idx=self.model_idx,
                                           output_path_manager=output_path_manager)
            self.aucs_dict[name].append(predict_aucs["auc_{}".format(self.data_name)])
            self.scores_dict[name].append(np.mean(probs))
        self.log()


class Recoder(BaseRecoder):
    def __init__(self, has_loss=None, logger=None, prefix=None, figs_dir=None, fig_infos_dir=None, title=None):
        super().__init__(logger, prefix, figs_dir, fig_infos_dir, title)
        self.has_loss = has_loss
        self.scores = []
        self.precisions = []
        self.recalls = []
        self.f1s = []
        self.aps = []
        self.aucs = []
        self.losses = []

    def clear(self):
        self.scores = []
        self.precisions = []
        self.recalls = []
        self.f1s = []
        self.aps = []
        self.aucs = []
        self.losses = []

    def draw(self):
        """
        color -> score precision recall f1, ap, auc, loss
        :return:
        """
        fig, axs = plt.subplots(1, 1, figsize=(6.4, 4.8))
        axs.plot(self.scores, 'b-', label="score")
        axs.plot(self.precisions, 'g-', label="precision")
        axs.plot(self.recalls, 'r-', label="recall")
        axs.plot(self.f1s, 'c-', label="f1")
        axs.plot(self.aps, 'm-', label="ap")
        axs.plot(self.aucs, 'y-', label="auc")
        if self.has_loss:
            axs.plot(self.losses, 'k-', label="loss")
        axs.legend()
        if len(self.scores) < 15:
            x_major_locator = MultipleLocator(1)
            axs.xaxis.set_major_locator(x_major_locator)
        fig.suptitle(self.title)
        fig.savefig(self.figs_dir + f'/{self.title}')
        plt.close(fig)


class BatchRecoder(Recoder):
    def __init__(self, threshold, has_loss, logger=None, prefix=None, figs_dir=None, fig_infos_dir=None,
                 title=None):
        super().__init__(has_loss, logger, prefix, figs_dir, fig_infos_dir, title)
        self.threshold = threshold
        self.num_records = 0

    def record(self, labels, scores, loss=None):
        self.num_records += 1
        labels_pred = scores > self.threshold
        self.scores.append(np.mean(scores))
        self.precisions.append(precision_score(labels, labels_pred, zero_division=0))
        self.recalls.append(recall_score(labels, labels_pred))
        self.f1s.append(f1_score(labels, labels_pred))
        self.aps.append(average_precision_score(labels, scores))
        self.aucs.append(roc_auc_score(labels, scores))
        if self.has_loss:
            self.losses.append(loss)

    def log(self):
        self.logger.info('{} mean score: {}, f1: {}'.format(self.prefix, self.scores[-1], self.f1s[-1]))
        self.logger.info('{} p: {}, recall: {}'.format(self.prefix, self.precisions[-1], self.recalls[-1]))
        self.logger.info('{} ap: {}, auc: {}'.format(self.prefix, self.aps[-1], self.aucs[-1]))
        if self.has_loss:
            self.logger.info('{} loss: {}'.format(self.prefix, self.losses[-1]))
        self.draw()


class SetBatchesRecoder(Recoder):
    def __init__(self, set_size, has_loss, logger, prefix, figs_dir, fig_infos_dir, title):
        """

        :param set_size: if set_size is zero, it will summarize all infos from recoder when active record
        """
        super().__init__(has_loss, logger, prefix, figs_dir, fig_infos_dir, title)
        self.set_size = set_size

    def record(self, recoder):
        self.scores.append(np.mean(recoder.scores[-self.set_size:]))
        self.precisions.append(np.mean(recoder.precisions[-self.set_size:]))
        self.recalls.append(np.mean(recoder.recalls[-self.set_size:]))
        self.f1s.append(np.mean(recoder.f1s[-self.set_size:]))
        self.aps.append(np.mean(recoder.aps[-self.set_size:]))
        self.aucs.append(np.mean(recoder.aucs[-self.set_size:]))
        if self.has_loss:
            self.losses.append(np.mean(recoder.losses[-self.set_size:]))

    def log(self):
        self.logger.info('{} mean score: {}, mean f1: {}'.format(self.prefix, self.scores[-1], self.f1s[-1]))
        self.logger.info('{} mean p: {}, mean recall: {}'.format(self.prefix, self.precisions[-1], self.recalls[-1]))
        self.logger.info('{} mean ap: {}, mean auc: {}'.format(self.prefix, self.aps[-1], self.aucs[-1]))
        if self.has_loss:
            self.logger.info('{} mean loss: {}'.format(self.prefix, self.losses[-1]))
        self.draw()
