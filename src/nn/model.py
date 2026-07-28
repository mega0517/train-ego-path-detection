import math

import torch
import torch.nn as nn

from .backbone import EfficientNetBackbone, ResNetBackbone
from .decoder import UNetDecoder


class ClassificationNet(nn.Module):
    def __init__(
        self,
        backbone,
        input_shape,
        anchors,
        classes,
        pool_channels,
        fc_hidden_size,
        pretrained=False,
    ):
        """Initializes the train ego-path detection model for the classification method.

        Args:
            backbone (str): Backbone to use in the model (e.g. "resnet18", "efficientnet-b3", etc.).
            input_shape (tuple): Input shape (C, H, W).
            anchors (int): Number of horizontal anchors in the input image where the path is classified.
            classes (int): Number of classes (grid cells) for each anchor. Background class is not included.
            pool_channels (int): Number of output channels of the pooling layer.
            fc_hidden_size (int): Number of units in the hidden layer of the fully connected part.
            pretrained (bool, optional): Whether to use pretrained weights for the backbone. Defaults to False.
        """
        super(ClassificationNet, self).__init__()
        if backbone.startswith("efficientnet"):
            self.backbone = EfficientNetBackbone(
                version=backbone[13:], pretrained=pretrained,
                in_channels=input_shape[0],
            )
        elif backbone.startswith("resnet"):
            self.backbone = ResNetBackbone(
                version=backbone[6:], pretrained=pretrained,
                in_channels=input_shape[0],
            )
        else:
            raise NotImplementedError
        self.pool = nn.Conv2d(
            in_channels=self.backbone.out_channels[-1],
            out_channels=pool_channels,
            kernel_size=1,
        )  # stride=1, padding=0
        self.fc = nn.Sequential(
            nn.Linear(
                pool_channels
                * math.ceil(input_shape[1] / self.backbone.reduction_factor)
                * math.ceil(input_shape[2] / self.backbone.reduction_factor),
                fc_hidden_size,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(fc_hidden_size, anchors * (classes + 1) * 2),
        )

    def forward(self, x):
        x = self.backbone(x)[0]
        fea = self.pool(x).flatten(start_dim=1)
        clf = self.fc(fea)
        return clf


class RegressionNet(nn.Module):
    def __init__(
        self,
        backbone,
        input_shape,
        anchors,
        pool_channels,
        fc_hidden_size,
        pretrained=False,
    ):
        """Initializes the train ego-path detection model for the regression method.

        Args:
            backbone (str): Backbone to use in the model (e.g. "resnet18", "efficientnet-b3", etc.).
            input_shape (tuple): Input shape (C, H, W).
            anchors (int): Number of horizontal anchors in the input image where the path is regressed.
            pool_channels (int): Number of output channels of the pooling layer.
            fc_hidden_size (int): Number of units in the hidden layer of the fully connected part.
            pretrained (bool, optional): Whether to use pretrained weights for the backbone. Defaults to False.
        """
        super(RegressionNet, self).__init__()
        if backbone.startswith("efficientnet"):
            self.backbone = EfficientNetBackbone(
                version=backbone[13:], pretrained=pretrained,
                in_channels=input_shape[0],
            )
        elif backbone.startswith("resnet"):
            self.backbone = ResNetBackbone(
                version=backbone[6:], pretrained=pretrained,
                in_channels=input_shape[0],
            )
        else:
            raise NotImplementedError
        self.pool = nn.Conv2d(
            in_channels=self.backbone.out_channels[-1],
            out_channels=pool_channels,
            kernel_size=1,
        )  # stride=1, padding=0
        self.fc = nn.Sequential(
            nn.Linear(
                pool_channels
                * math.ceil(input_shape[1] / self.backbone.reduction_factor)
                * math.ceil(input_shape[2] / self.backbone.reduction_factor),
                fc_hidden_size,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(fc_hidden_size, anchors * 2 + 1),
        )

    def forward(self, x):
        x = self.backbone(x)[0]
        fea = self.pool(x).flatten(start_dim=1)
        reg = self.fc(fea)
        return reg



class MultiPathRegressionNet(nn.Module):
    """RegressionNet with K ego-path hypotheses instead of one, plus a score head.

    Where the ego-path is ambiguous -- most importantly at a facing switch, where
    the train may take either branch -- a single regressed path can only predict
    something between the branches, which is on neither track. Emitting K paths
    lets the network keep both options open; a winner-takes-all loss (see
    MultiHypothesisRegressionLoss) makes the heads specialise without needing any
    branch annotation, and the score head learns which one is right per frame.

    Everything up to the hidden fully connected layer is identical to
    RegressionNet, so a trained single-path model can warm-start this one.
    """

    def __init__(
        self,
        backbone,
        input_shape,
        anchors,
        pool_channels,
        fc_hidden_size,
        n_hypotheses=2,
        pretrained=False,
    ):
        super(MultiPathRegressionNet, self).__init__()
        if backbone.startswith("efficientnet"):
            self.backbone = EfficientNetBackbone(
                version=backbone[13:], pretrained=pretrained,
                in_channels=input_shape[0],
            )
        elif backbone.startswith("resnet"):
            self.backbone = ResNetBackbone(
                version=backbone[6:], pretrained=pretrained,
                in_channels=input_shape[0],
            )
        else:
            raise NotImplementedError
        self.n_hypotheses = n_hypotheses
        self.path_dim = anchors * 2 + 1
        self.pool = nn.Conv2d(
            in_channels=self.backbone.out_channels[-1],
            out_channels=pool_channels,
            kernel_size=1,
        )
        self.fc = nn.Sequential(
            nn.Linear(
                pool_channels
                * math.ceil(input_shape[1] / self.backbone.reduction_factor)
                * math.ceil(input_shape[2] / self.backbone.reduction_factor),
                fc_hidden_size,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(fc_hidden_size, n_hypotheses * self.path_dim + n_hypotheses),
        )

    def split_output(self, out):
        """(B, K*D + K) -> paths (B, K, D), scores (B, K)."""
        k = self.n_hypotheses
        return out[:, :-k].reshape(out.size(0), k, self.path_dim), out[:, -k:]

    def load_single_path_state(self, state, noise=0.01):
        """Warm-start from a trained RegressionNet checkpoint.

        Backbone, pooling and the hidden layer are copied as they are; the final
        layer is replicated once per hypothesis with a little noise, so the heads
        start from the trained single-path solution but are not exactly tied --
        identical heads would stay identical under the winner-takes-all loss,
        since only one of them is ever the winner.
        """
        state = {k.replace("_orig_mod.", "").replace("module.", ""): v
                 for k, v in state.items()}
        own = self.state_dict()
        copied = []
        for key, val in state.items():
            if key in own and own[key].shape == val.shape:
                own[key] = val.clone()
                copied.append(key)
        for suffix, out_slice in (("weight", None), ("bias", None)):
            src = state.get(f"fc.2.{suffix}")
            if src is None:
                continue
            dst = own[f"fc.2.{suffix}"]
            for k in range(self.n_hypotheses):
                lo = k * self.path_dim
                chunk = src.clone()
                chunk = chunk + noise * torch.randn_like(chunk) * chunk.std()
                dst[lo : lo + self.path_dim] = chunk
            dst[self.n_hypotheses * self.path_dim :] = 0.0  # scores start neutral
        self.load_state_dict(own)
        return len(copied)

    def forward(self, x):
        x = self.backbone(x)[0]
        fea = self.pool(x).flatten(start_dim=1)
        return self.fc(fea)


class SegmentationNet(nn.Module):
    def __init__(
        self,
        backbone,
        decoder_channels,
        pretrained=False,
        in_channels=3,
    ):
        """Initializes the train ego-path detection model for the segmentation method.

        Args:
            backbone (str): Backbone to use in the model (e.g. "resnet18", "efficientnet-b3", etc.).
            decoder_channels (tuple): Number of output channels of each decoder block.
            pretrained (bool, optional): Whether to use pretrained weights for the backbone. Defaults to False.
            in_channels (int, optional): Number of input channels (4 adds a prior-path channel). Defaults to 3.
        """
        super(SegmentationNet, self).__init__()
        if backbone.startswith("efficientnet"):
            self.encoder = EfficientNetBackbone(
                version=backbone[13:],
                out_levels=(1, 3, 4, 6, 8),
                pretrained=pretrained,
                in_channels=in_channels,
            )
        elif backbone.startswith("resnet"):
            self.encoder = ResNetBackbone(
                version=backbone[6:],
                out_levels=(1, 2, 3, 4, 5),
                pretrained=pretrained,
                in_channels=in_channels,
            )
        else:
            raise NotImplementedError
        self.decoder = UNetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
        )
        self.segmentation_head = nn.Conv2d(
            in_channels=decoder_channels[-1],
            out_channels=1,  # binary segmentation
            kernel_size=3,
            padding=1,
        )  # stride=1

    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(features)
        masks = self.segmentation_head(decoder_output)
        return masks


def _temporal_train(model, mode):
    """nn.Module.train() that keeps a frozen base net in eval mode (stable BatchNorm/dropout)."""
    nn.Module.train(model, mode)
    if not any(p.requires_grad for p in model.base.parameters()):
        model.base.eval()
    return model


class _SequenceVectorRefiner(nn.Module):
    """Output-level temporal refiner for the vector-headed methods.

    Consumes a sequence of per-frame base-net output vectors (the ego-paths
    predicted on the last T frames, stored in memory) and predicts a residual
    correction to the current (last) frame's prediction, so the path is tracked
    over time. The output projection is zero-initialized so that, before any
    temporal training, the refiner reproduces the base-net prediction exactly.
    """

    def __init__(self, output_dim, rnn_hidden, rnn_layers):
        super(_SequenceVectorRefiner, self).__init__()
        self.input_proj = nn.Linear(output_dim, rnn_hidden)
        self.rnn = nn.GRU(
            input_size=rnn_hidden,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
        )
        self.output_proj = nn.Linear(rnn_hidden, output_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, base_seq):  # (B, T, D) -> (B, D)
        emb = self.input_proj(base_seq)
        rnn_out, _ = self.rnn(emb)
        delta = self.output_proj(rnn_out[:, -1])
        return base_seq[:, -1] + delta


class ConvGRUCell(nn.Module):
    """Minimal convolutional GRU cell, used to refine segmentation logits over time."""

    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super(ConvGRUCell, self).__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv_zr = nn.Conv2d(
            in_channels + hidden_channels, 2 * hidden_channels, kernel_size, padding=padding
        )
        self.conv_h = nn.Conv2d(
            in_channels + hidden_channels, hidden_channels, kernel_size, padding=padding
        )

    def forward(self, x, h):  # x: (B, Cin, H, W), h: (B, Chid, H, W)
        z, r = torch.split(
            torch.sigmoid(self.conv_zr(torch.cat([x, h], dim=1))),
            self.hidden_channels,
            dim=1,
        )
        h_tilde = torch.tanh(self.conv_h(torch.cat([x, r * h], dim=1)))
        return (1 - z) * h + z * h_tilde


class RegressionNetRNN(nn.Module):
    def __init__(
        self,
        backbone,
        input_shape,
        anchors,
        pool_channels,
        fc_hidden_size,
        seq_len,
        rnn_hidden,
        rnn_layers,
        pretrained=False,
    ):
        """Temporal (RNN) variant of RegressionNet tracking the ego-path over a sequence of frames.

        Wraps a per-frame RegressionNet and refines the current frame's
        prediction from the sequence of the last `seq_len` predicted ego-paths.

        Args:
            backbone, input_shape, anchors, pool_channels, fc_hidden_size, pretrained:
                Forwarded to the underlying per-frame RegressionNet.
            seq_len (int): Number of consecutive frames kept in memory.
            rnn_hidden (int): Hidden size of the temporal GRU.
            rnn_layers (int): Number of stacked GRU layers.
        """
        super(RegressionNetRNN, self).__init__()
        self.seq_len = seq_len
        self.base = RegressionNet(
            backbone, input_shape, anchors, pool_channels, fc_hidden_size, pretrained
        )
        self.refiner = _SequenceVectorRefiner(anchors * 2 + 1, rnn_hidden, rnn_layers)

    def train(self, mode=True):
        return _temporal_train(self, mode)

    def base_forward(self, x):
        return self.base(x)

    def refine(self, base_seq):
        return self.refiner(base_seq)

    def forward(self, x):  # (B, T, C, H, W) -> (B, anchors * 2 + 1)
        b, t = x.shape[0], x.shape[1]
        base_out = self.base(x.reshape(b * t, *x.shape[2:]))
        return self.refiner(base_out.reshape(b, t, -1))


class ClassificationNetRNN(nn.Module):
    def __init__(
        self,
        backbone,
        input_shape,
        anchors,
        classes,
        pool_channels,
        fc_hidden_size,
        seq_len,
        rnn_hidden,
        rnn_layers,
        pretrained=False,
    ):
        """Temporal (RNN) variant of ClassificationNet. See RegressionNetRNN for the temporal contract."""
        super(ClassificationNetRNN, self).__init__()
        self.seq_len = seq_len
        self.base = ClassificationNet(
            backbone, input_shape, anchors, classes, pool_channels, fc_hidden_size, pretrained
        )
        self.refiner = _SequenceVectorRefiner(
            anchors * (classes + 1) * 2, rnn_hidden, rnn_layers
        )

    def train(self, mode=True):
        return _temporal_train(self, mode)

    def base_forward(self, x):
        return self.base(x)

    def refine(self, base_seq):
        return self.refiner(base_seq)

    def forward(self, x):  # (B, T, C, H, W) -> (B, anchors * (classes + 1) * 2)
        b, t = x.shape[0], x.shape[1]
        base_out = self.base(x.reshape(b * t, *x.shape[2:]))
        return self.refiner(base_out.reshape(b, t, -1))


class SegmentationNetRNN(nn.Module):
    def __init__(
        self,
        backbone,
        decoder_channels,
        seq_len,
        rnn_hidden,
        rnn_layers=1,
        pretrained=False,
    ):
        """Temporal (RNN) variant of SegmentationNet.

        Refines the current frame's mask logits with a ConvGRU running over the
        sequence of per-frame mask logits kept in memory. `rnn_layers` is accepted
        for config uniformity; a single ConvGRU cell is used.
        """
        super(SegmentationNetRNN, self).__init__()
        self.seq_len = seq_len
        self.base = SegmentationNet(backbone, decoder_channels, pretrained)
        self.convgru = ConvGRUCell(in_channels=1, hidden_channels=rnn_hidden)
        self.out_conv = nn.Conv2d(rnn_hidden, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def train(self, mode=True):
        return _temporal_train(self, mode)

    def base_forward(self, x):
        return self.base(x)

    def refine(self, base_seq):  # (B, T, 1, H, W) -> (B, 1, H, W)
        b, t = base_seq.shape[0], base_seq.shape[1]
        h = base_seq.new_zeros(
            (b, self.convgru.hidden_channels, base_seq.shape[3], base_seq.shape[4])
        )
        for i in range(t):
            h = self.convgru(base_seq[:, i], h)
        return base_seq[:, -1] + self.out_conv(h)

    def forward(self, x):  # (B, T, C, H, W) -> (B, 1, H, W)
        b, t = x.shape[0], x.shape[1]
        base_out = self.base(x.reshape(b * t, *x.shape[2:]))
        return self.refine(base_out.reshape(b, t, *base_out.shape[1:]))
