import argparse
from pathlib import Path
from datetime import datetime
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Average, Accuracy
import torch
from torch.utils import tensorboard
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from typing import Callable, Union
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce
from trg.networks import load_network
from trg.networks import ModelWithSE
from trg.dataset import Dataset


seed = 42


class ECE(Accuracy):

    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        is_multilabel: bool = False,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super(ECE, self).__init__(output_transform=output_transform, is_multilabel=is_multilabel, device=device)
        n_bins =10
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    @reinit__is_reduced
    def reset(self):
        self._num_confidences = torch.tensor([], device=self._device)
        self._num_corrects = torch.tensor([], device=self._device)
        self._ece = torch.tensor([0.0], device=self._device)
        super(ECE, self).reset()

    @reinit__is_reduced
    def update(self, output):
        self._check_shape(output)
        y_pred, y = output[0].detach(), output[1].detach()
        if y_pred.squeeze().ndimension() == 1:
            y_pred_two = torch.cat(((1 - y_pred.view(-1)).unsqueeze(1), y_pred.view(-1).unsqueeze(1)), 1)
            confidences, indices = torch.max(y_pred_two, 1)
            correct = torch.eq(indices, y.view(-1)).view(-1)
        else:
            confidences, indices = torch.max(y_pred, 1)
            correct = torch.eq(indices, y).view(-1)

        self._num_corrects = torch.cat((self._num_corrects, correct.to(self._device)))
        self._num_confidences = torch.cat((self._num_confidences, confidences.to(self._device)))


    @sync_all_reduce("_num_examples", "_num_correct")
    def compute(self) -> float:
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = self._num_confidences.gt(bin_lower.item()) * self._num_confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = self._num_corrects[in_bin].float().mean()
                avg_confidence_in_bin = self._num_confidences[in_bin].mean()
                self._ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                # print(self._ece, prop_in_bin, accuracy_in_bin, avg_confidence_in_bin)

        return self._ece.float().item()


def main(args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 3, "pin_memory": True} if use_cuda else {}

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # create log directory
    time_stamp = datetime.now().strftime("%y-%m-%d-%H-%M")
    description = "{}_dataset={},net={},batch_size={},lr={:.0e},{}".format(
        time_stamp,
        args.dataset.name,
        args.net,
        args.batch_size,
        args.lr,
        args.description,
    ).strip(",")
    logdir = args.logdir / description

    # create data loaders
    train_loader, val_loader = create_train_val_loaders(
        args.dataset, args.batch_size, args.val_split, kwargs
    )

    # build the network
    orig_model = load_network(args.model, torch.device("cpu"))
    scaled_model = ModelWithSE(orig_model).to(device)
    # define optimizer and metrics
    optimizer = torch.optim.Adam(
        [{'params': scaled_model.conv1.parameters()},
         {'params': scaled_model.conv2.parameters()},
         {'params': scaled_model.conv3.parameters()},
         {'params': scaled_model.conv4.parameters()}
         ], lr=args.lr)

    metrics = {
        "loss": Average(lambda out: out[3]),
        "accuracy": Accuracy(lambda out: (torch.round(out[1][0]), out[2][0])),
        "ECE": ECE(lambda out: (out[1][0], out[2][0])),
    }

    trainer = create_trainer(scaled_model, optimizer, loss_fn_se, metrics, device)

    # log training progress to the terminal and tensorboard
    ProgressBar(persist=True, ascii=True).attach(trainer)

    train_writer, val_writer = create_summary_writers(scaled_model, device, logdir)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_train_results(engine):
        epoch, metrics = trainer.state.epoch, trainer.state.metrics
        train_writer.add_scalar("loss", metrics["loss"], epoch)
        train_writer.add_scalar("accuracy", metrics["accuracy"], epoch)
        train_writer.add_scalar("ECE", metrics["ECE"], epoch)

    def default_score_fn(engine):
        score = engine.state.metrics['ECE']
        return 1 - score

    # # checkpoint model
    # checkpoint_handler = ModelCheckpoint(
    #     logdir,
    #     "vgn",
    #     n_saved=1,  # 最多保存新权重文件个数
    #     require_empty=True,
    # )

    best_checkpoint_handler = ModelCheckpoint(
        logdir,
        "vgn",
        n_saved=1,
        score_name="best_ece",
        score_function=default_score_fn,
        require_empty=True,
    )

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=1), best_checkpoint_handler, {args.net: scaled_model}  # 每隔一轮保存一次权重
    )

    # run the training loop
    trainer.run(val_loader, max_epochs=args.epochs)


def create_train_val_loaders(root, batch_size, val_split, kwargs):
    # load the dataset
    dataset = Dataset(root)
    # split into train and validation sets
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    # create loaders for both datasets
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs
    )
    return train_loader, val_loader


def prepare_batch(batch, device):
    tsdf, (label, rotations, width), index = batch
    tsdf = tsdf.to(device)
    label = label.float().to(device)
    rotations = rotations.to(device)
    width = width.to(device)
    index = index.to(device)
    return tsdf, (label, rotations, width), index


def select(out, index):
    qual_out, rot_out, width_out = out
    label = qual_out.squeeze()[index == 0]
    rot = rot_out.transpose(1, 2).transpose(2, 3).transpose(3, 4).contiguous()[index == 0]
    width = width_out.squeeze()[index == 0]
    return label, rot, width


def loss_fn_se(y_pred, y, index):
    label_pred, rotation_pred, width_pred = y_pred
    label, rotations, width = y
    real_label = label[index == 0]
    real_rotations = rotations[index == 0]
    real_width = width[index == 0]
    loss_qual = _qual_loss_fn(label_pred, real_label)
    loss = loss_qual
    return loss.mean(), (real_label, real_rotations, real_width)


def _qual_loss_fn(pred, target):
    return F.binary_cross_entropy(torch.sigmoid(pred), target, reduction="none")


def _rot_loss_fn(pred, target):
    loss0 = _quat_loss_fn(pred, target[:, 0])
    loss1 = _quat_loss_fn(pred, target[:, 1])
    return torch.min(loss0, loss1)


def _quat_loss_fn(pred, target):
    return 1.0 - torch.abs(torch.sum(pred * target, dim=1))


def _width_loss_fn(pred, target):
    return F.mse_loss(pred, target, reduction="none")


def create_trainer(net, optimizer, loss_fn, metrics, device):
    def _update(engine, batch):
        net.train()

        # forward
        x, y, index = prepare_batch(batch, device)
        y_pred = select(net(x), index)
        loss, real_y = loss_fn(y_pred, y, index)

        optimizer.zero_grad()
        # backward
        loss.backward()
        optimizer.step()

        return x, [torch.sigmoid(y_pred[0]), y_pred[1], y_pred[2]], real_y, loss

    trainer = Engine(_update)

    for name, metric in metrics.items():
        metric.attach(trainer, name)

    return trainer


def create_summary_writers(net, device, log_dir):
    train_path = log_dir / "train"
    val_path = log_dir / "validation"

    train_writer = tensorboard.SummaryWriter(train_path, flush_secs=60)
    val_writer = tensorboard.SummaryWriter(val_path, flush_secs=60)

    return train_writer, val_writer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--net", default="conv")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--logdir", type=Path, default="data/runs")
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    args = parser.parse_args()
    main(args)
