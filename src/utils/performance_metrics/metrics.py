import torch


class Metric(object):
    def __init__(self):
        super(self, Metric).__init__()

    def compute(self, labels, preds):
        pass

    @property
    def name(self):
        return 'Metric'


class Accuracy(Metric):
    def compute(self, labels, preds):
        """
        Compute accuracy for a class, if the positive class number is defined it will be used instead of default 1.
        :param labels: Torch tensor of labels
        :param preds: Torch tensor of predictions
        :return: Accuracy score for positive class
        """
        correct = torch.where(labels == preds)
        return correct

    @property
    def name(self):
        return 'Accuracy'


class Recall(Metric):
    def compute(self, labels, preds):
        """
        Compute recall for a class, if the positive class number is defined it will be used instead of default 1.
        :param labels: Torch tensor of labels
        :param preds: Torch tensor of predictions
        :param pos_label: Int for positive class label
        :return: Recall score for positive class
        """
        TP = torch.where(labels == 1 and preds == 1)
        TN = torch.where(labels != 1 and preds != 1)
        recall_score = TP / (TP + TN)

        return recall_score

    @property
    def name(self):
        return 'Recall'


class Precision(Metric):
    def compute(self, labels, preds):
        """
        Compute precision for a class, if the positive class number is defined it will be used instead of default 1.
        :param labels: Torch tensor of labels
        :param preds: Torch tensor of predictions
        :param pos_label: Int for positive class label
        :return: Precision score for positive class
        """
        TP = torch.where(labels == 1 and preds == 1)
        FP = torch.where(labels != 1 and preds == 1)
        precision_score = TP / (TP + FP)

        return precision_score

    @property
    def name(self):
        return 'Precision'


class F1_Score(Metric):
    def compute(self, labels, preds):
        """
        Compute F1-score for a class, if the positive class number is defined it will be used instead of default 1.
        :param labels: Torch tensor of labels
        :param preds: Torch tensor of predictions
        :param pos_label: Int for positive class label
        :return: F1-score for positive class
        """
        TP = torch.where(labels == 1 and preds == 1)
        FP = torch.where(labels != 1 and preds == 1)
        FN = torch.where(labels == 1 and preds != 1)
        f1 = TP / (TP + 0.5 * (FP + FN))

        return f1

    @property
    def name(self):
        return 'F1_Score'


class Macro_Recall(Metric):
    def compute(self, labels, preds):
        """
        Compute macro recall for all classes.
        :param labels: Torch tensor of labels
        :param preds: Torch tensor of predictions
        :return: Macro-recall score for all classes
        """
        recall_cls = Recall()
        classes = torch.unique(labels)
        recalls = {}
        for cls in classes:
            recalls[cls] = recall_cls.compute(labels, preds)

        macro_recall_score = torch.sum(recalls.keys()) / len(classes)

        return macro_recall_score

    @property
    def name(self):
        return 'Macro_Recall'


class Macro_Precision(Metric):
    def compute(self, labels, preds):
        """
        Compute macro precision for all classes.
        :param labels: Torch tensor of labels
        :param preds: Torch tensor of predictions
        :return: Macro-precision score for all classes
        """
        precision_cls = Precision()
        classes = torch.unique(labels)
        precisions = {}
        for cls in classes:
            precisions[cls] = precision_cls.compute(labels, preds)

        macro_precision_score = torch.sum(precisions.keys()) / len(classes)

        return macro_precision_score

    @property
    def name(self):
        return 'Macro_Precision'


class Macro_F1_Score(Metric):
    def compute(self, labels, preds):
        """
        Compute macro F1-score for all classes.
        :param labels: Torch tensor of labels
        :param preds: Torch tensor of predictions
        :param precomputed_precisions: Dictionary of precomputed precision values by class
        :param precomputed_recalls: Dictionary of precomputed recall values by class
        :return: Macro F1-score for all classes
        """
        f1_score_cls = F1_Score()
        classes = torch.unique(labels)
        f1_scores = {}
        for cls in classes:
            f1_scores[cls] = f1_score_cls.compute(labels, preds)

        macro_f1 = torch.sum(f1_scores.keys()) / len(classes)

        return macro_f1

    @property
    def name(self):
        return 'Macro_F1_Score'

