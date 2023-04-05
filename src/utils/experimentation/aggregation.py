import torch


def aggregate_predictions(pred_list, pred_str='preds', target_str='targets'):
    """
    Aggregate a series of predictions from a list of dictionaries.
    :param pred_list: List of dictionaries for prediction outputs
    :param pred_str: String for predictions
    :param target_str: String for targets
    :return: Tuple of preds, targets tensors
    """
    preds = []
    targets =[]
    for pred_batch in pred_list:
        preds.append(pred_batch[pred_str])
        target_val = pred_batch[target_str]
        if target_val.ndim == 0:
            target_val = torch.unsqueeze(target_val, 0)
        targets.append(target_val)

    preds_tensor = torch.concatenate(preds)
    targets_tensor = torch.concatenate(targets)

    return preds_tensor, targets_tensor

