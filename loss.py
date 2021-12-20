import torch
import torch.nn as nn


def generator_loss(pred_outputs):
    loss = 0
    for pred_output in pred_outputs:
        loss += torch.mean((1 - pred_output) ** 2)
    return loss


def feature_loss(real_features, pred_features):
    loss = 0
    for real_features_sub, pred_features_sub in zip(real_features, pred_features):
        for real_feature, pred_feature in zip(real_features_sub, pred_features_sub):
            loss += nn.L1Loss()(real_feature, pred_feature)
    return loss


def discriminator_loss(real_outputs, pred_outputs):
    loss = 0
    for real_output, pred_output in zip(real_outputs, pred_outputs):
        loss += torch.mean((1 - real_output) ** 2) + torch.mean(pred_output ** 2)
    return loss
