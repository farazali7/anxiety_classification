from statistics import mode
import torch


def majority_vote_transform(preds, targets, voters, drop_last=False):
    """
    Transform input predictions and targets vectors to a majority-vote setting based on number of voters
    :param preds: Tensor of predictions
    :param targets: Tensor of targets/labels
    :param voters: Integer for number of voters in majority vote
    :param drop_last: Boolean to drop last group
    """
    grouped_preds = [preds[i:i+voters].detach().numpy() for i in range(0, len(preds), voters)]
    grouped_targets = [targets[i:i + voters].detach().numpy() for i in range(0, len(targets), voters)]

    if drop_last:
        if len(grouped_preds[-1]) < voters:
            grouped_preds = grouped_preds[:-1]
        if len(grouped_targets[-1]) < voters:
            grouped_targets = grouped_targets[:-1]

    mode_preds = [mode(pred) for pred in grouped_preds]
    mode_preds = torch.tensor(mode_preds, dtype=torch.int64)
    mode_targets = [mode(target) for target in grouped_targets]
    mode_targets = torch.tensor(mode_targets, dtype=torch.int64)

    return mode_preds, mode_targets
