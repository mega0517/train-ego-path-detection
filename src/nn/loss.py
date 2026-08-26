import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, prediction, target):
        target = target.flatten()
        prediction = prediction.view(target.shape[0], -1)
        return self.loss(prediction, target)


class TrainEgoPathRegressionLoss(nn.Module):
    def __init__(self, ylimit_loss_weight, perspective_weight_limit=None):
        super(TrainEgoPathRegressionLoss, self).__init__()
        self.ylimit_loss_weight = ylimit_loss_weight
        self.perspective_weight_limit = perspective_weight_limit
        self.unreduced_smae = nn.SmoothL1Loss(reduction="none", beta=0.005)
        self.batchaveraged_smae = nn.SmoothL1Loss(reduction="mean", beta=0.015)
        self.unreduced_ylim_smae = nn.SmoothL1Loss(reduction="none", beta=0.015)

    def trajectory_loss(self, traj_prediction, traj_target, ylim_target, reduce=True):
        traj_se = self.unreduced_smae(traj_prediction, traj_target)  # (B, 2, H)
        ylim_target_idx = ylim_target * (traj_target.size(2) - 1)  # (B,)
        range_matrix = torch.arange(
            traj_target.size(2), device=ylim_target_idx.device
        ).expand(traj_target.size(0), -1)  # (B, H)
        loss_mask = (range_matrix <= ylim_target_idx.unsqueeze(1)).float()  # (B, H)
        rail_width = traj_target[:, 1, :] - traj_target[:, 0, :]  # (B, H)
        weights = (loss_mask / rail_width).unsqueeze(dim=1)  # (B, 1, H)
        if self.perspective_weight_limit is not None:
            weights = torch.clamp(weights, max=self.perspective_weight_limit)
        traj_loss = (traj_se * weights).sum(dim=(1, 2)) / loss_mask.sum(dim=1)
        mask = ylim_target == 0
        traj_loss = traj_loss.masked_fill(mask, 0.0)
        return traj_loss.mean() if reduce else traj_loss  # () or (B,)

    def ylim_loss(self, ylim_prediction, ylim_target):
        return self.batchaveraged_smae(torch.sigmoid(ylim_prediction), ylim_target)

    def per_sample(self, prediction, target):
        """Same loss as forward(), but kept per sample: (B,) instead of a scalar.

        Multi-hypothesis training needs to compare hypotheses sample by sample.
        forward() is the mean of this, which is exactly its previous value since
        the mean of a sum is the sum of the means.
        """
        traj_target, ylim_target = target
        traj_prediction = prediction[:, :-1].view_as(traj_target)
        ylim_prediction = prediction[:, -1]
        traj_loss = self.trajectory_loss(
            traj_prediction, traj_target, ylim_target, reduce=False
        )  # (B,)
        ylim_loss = self.unreduced_ylim_smae(
            torch.sigmoid(ylim_prediction), ylim_target
        )  # (B,)
        return traj_loss + self.ylimit_loss_weight * ylim_loss

    def forward(self, prediction, target):
        return self.per_sample(prediction, target).mean()


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, prediction, target):
        prediction = torch.sigmoid(prediction)
        prediction = prediction.flatten(start_dim=1)  # (B, H * W)
        target = target.flatten(start_dim=1)
        zero_target_mask = target.sum(dim=1) == 0  # (B,)
        if zero_target_mask.any():
            prediction[zero_target_mask] = 1 - prediction[zero_target_mask]
            target[zero_target_mask] = 1 - target[zero_target_mask]
        intersection = (prediction * target).sum(dim=1)  # (B,)
        cardinality = (prediction + target).sum(dim=1)
        scores = 2 * intersection / cardinality
        return 1 - scores.mean()  # ()


class MultiHypothesisRegressionLoss(nn.Module):
    """Winner-takes-all loss for a K-hypothesis ego-path head.

    At a switch the ego-path is genuinely ambiguous: a single regressed path can
    only hedge between the branches (see Laurent §VI-C), which matches neither.
    With K hypotheses, only the one closest to the ground truth is trained on
    each sample ("hindsight" / WTA), so the heads specialise -- typically one per
    branch -- without ever needing a branch label.

    Pure WTA starves the losing heads of gradient and they drift or die, so the
    losers keep a small weight `epsilon`. A score head is trained (cross-entropy)
    to predict which hypothesis hindsight picked; at inference that score, or a
    temporal selector, chooses the path.
    """

    def __init__(self, base_loss, n_hypotheses, epsilon=0.05, score_weight=0.1):
        super(MultiHypothesisRegressionLoss, self).__init__()
        self.base_loss = base_loss
        self.n_hypotheses = n_hypotheses
        self.epsilon = epsilon
        self.score_weight = score_weight
        self.ce = nn.CrossEntropyLoss()

    @staticmethod
    def split(prediction, n_hypotheses):
        """(B, K*D + K) -> paths (B, K, D), scores (B, K)."""
        k = n_hypotheses
        scores = prediction[:, -k:]
        paths = prediction[:, :-k].reshape(prediction.size(0), k, -1)
        return paths, scores

    def forward(self, prediction, target):
        paths, scores = self.split(prediction, self.n_hypotheses)
        per_hyp = torch.stack(
            [self.base_loss.per_sample(paths[:, k], target) for k in range(self.n_hypotheses)],
            dim=1,
        )  # (B, K)
        best = per_hyp.argmin(dim=1)  # (B,)
        winner = per_hyp.gather(1, best.unsqueeze(1)).squeeze(1)  # (B,)
        losers = (per_hyp.sum(dim=1) - winner) / max(self.n_hypotheses - 1, 1)
        wta = (winner + self.epsilon * losers).mean()
        return wta + self.score_weight * self.ce(scores, best.detach())
