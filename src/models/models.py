import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from copy import deepcopy


def get_model(model_name, model_args, trainer_args, batch_specific_train=False, use_legacy=False):
    """
    Retrieve a module wrapped around specified model, initialized with proper arguments.
    :param model_name: String for underlying model architecture
    :param model_args: Dictionary of model architecture arguments
    :param trainer_args: Keyword arguments for model trainer such as learning rate, class weights, etc.
    :param batch_specific_train: Boolean for whether to user batch specific trainer or regular
    :return: Model
    """
    try:
        model_fn = eval(model_name)
        model_def = model_fn(model_args)
    except Exception as e:
        print(e)
        raise Exception(f'Given model name: {model_name} is not supported.')

    if use_legacy:
        model = LegacyModel(model_def, **trainer_args)
    elif batch_specific_train:
        model = BatchwiseTrainModel(model_def, **trainer_args)
    else:
        model = Model(model_def, **trainer_args)

    return model


def load_model_from_checkpoint(checkpoint_path, strict=True, **kwargs):
    """
    Load and return a model from a given checkpoint path.
    :param checkpoint_path: String for location of model
    :return: Model from checkpoint
    """
    model = Model.load_from_checkpoint(checkpoint_path=checkpoint_path, strict=strict, **kwargs)

    return model


class Model(pl.LightningModule):
    def __init__(self, model, learning_rate, class_weights, classes, metrics, fold=None, prev_optimizer_state=None):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.class_weights = class_weights
        self.classes = classes
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

        if fold is not None:
            self.fold = fold
            self.log_prefix = 'fold_' + str(self.fold) + '/'
        else:
            self.log_prefix = ''

        self.train_step_outputs = []
        self.val_step_outputs = []
        self.test_step_outputs = []

        self.prev_optimizer_state = prev_optimizer_state

        self.save_hyperparameters(ignore=['metrics'])

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.model(x)
        targets = torch.squeeze(y)
        loss = nn.CrossEntropyLoss(weight=self.class_weights)(outputs, targets)

        _, preds = torch.max(outputs.detach(), 1)

        # Log and accumulate
        res = {'loss': loss, 'preds': preds, 'targets': targets}
        self.train_step_outputs.append(res)

        train_metrics_out = self.train_metrics(preds, targets)
        self.log_metrics(train_metrics_out)
        self.log(self.log_prefix + 'train_loss', loss, prog_bar=True)

        return res

    def on_train_epoch_end(self):
        train_metrics_out = self.train_metrics.compute()
        self.log_metrics(train_metrics_out)
        self.train_metrics.reset()

        step_losses = [step_item['loss'] for step_item in self.train_step_outputs]
        epoch_loss = torch.stack(step_losses).mean()
        self.log(self.log_prefix + 'epoch_train_loss', epoch_loss)
        self.train_step_outputs.clear()  # free memory

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.model(x)
        targets = torch.squeeze(y)
        val_loss = nn.CrossEntropyLoss()(outputs, targets)

        _, preds = torch.max(outputs.detach(), 1)

        # Log and accumulate
        res = {'val_loss': val_loss, 'preds': preds, 'targets': targets}
        self.val_step_outputs.append(res)

        self.val_metrics.update(preds, targets)
        self.log(self.log_prefix + 'val_loss', val_loss, prog_bar=True)

        return res

    def on_validation_epoch_end(self):
        valid_metrics_out = self.val_metrics.compute()
        self.log_metrics(valid_metrics_out)
        self.val_metrics.reset()

        step_losses = [step_item['val_loss'] for step_item in self.val_step_outputs]
        epoch_loss = torch.stack(step_losses).mean()
        self.log(self.log_prefix + 'epoch_val_loss', epoch_loss)
        self.val_step_outputs.clear()  # free memory

    def test_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.model(x)
        targets = torch.squeeze(y)
        test_loss = nn.CrossEntropyLoss()(outputs, targets)

        _, preds = torch.max(outputs.detach(), 1)

        # Log and accumulate
        res = {'test_loss': test_loss, 'preds': preds, 'targets': targets}
        self.test_step_outputs.append(res)
        self.test_metrics.update(preds, targets)

        return res

    def on_test_epoch_end(self):
        test_metrics_out = self.test_metrics.compute()
        self.log_metrics(test_metrics_out)
        self.test_metrics.reset()

        self.test_step_outputs.clear()

    def predict_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.model(x)
        targets = torch.squeeze(y)

        _, preds = torch.max(outputs.detach(), 1)

        return {'preds': preds, 'targets': targets}

    def configure_optimizers(self):
        if self.prev_optimizer_state is not None:
            return self.configure_optimizers_ckpt()
        else:
            return self.configure_optimizers_fresh()

    def configure_optimizers_ckpt(self):
        optimizer = optim.Adam(self.model.parameters(), self.learning_rate)
        optimizer.load_state_dict(deepcopy(self.prev_optimizer_state))
        return optimizer

    def configure_optimizers_fresh(self):
        optimizer = optim.Adam(self.model.parameters(), self.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": optim.lr_scheduler.ReduceLROnPlateau(optimizer),
                "monitor": self.log_prefix + "val_loss",
                "interval": "epoch",
                "frequency": 1
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }

    def log_metrics(self, metrics_val_dict):
        """
        Delineate when logging MetricCollections for scalar and tensor-returning metrics.
        :param metrics_val_dict: Dictionary of metric values to log
        :return: None, simply logs to logger
        """
        scalar_dict = {}
        for metric_name in metrics_val_dict:
            metric_tensor = metrics_val_dict[metric_name]
            new_name = self.log_prefix + metric_name
            if metric_tensor.ndim == 0:  # Scalar
                scalar_dict[new_name] = metric_tensor
            else:
                for i, cls in enumerate(self.classes):
                    scalar_dict[new_name + '_' + cls] = metric_tensor[i]

        self.log_dict(scalar_dict)


class BatchwiseTrainModel(Model):
    def __init__(self, *args, **kwargs):
        super(BatchwiseTrainModel, self).__init__(*args, **kwargs)

        self.bn_weights_by_subject = {}
        state_dict = self.model.state_dict()
        batch_norm_dict = {}
        for key in state_dict:
            if "bnorm" in key:
                batch_norm_dict.update({key: state_dict[key]})
        self.default_bn_weights = deepcopy(batch_norm_dict)

        self.current_subject_batch = 0  # Initialize

    def training_step(self, batch, batch_idx):
        x, y = batch

        # Load subject-specific batch-norm weights
        unique_val = torch.unique(x[..., -1])
        assert len(unique_val) == 1, "Mixed batches not allowed."
        subject_num = int(unique_val)
        self.current_subject_batch = subject_num
        if subject_num in self.bn_weights_by_subject:
            bn_weights = self.bn_weights_by_subject[subject_num]
        else:
            bn_weights = deepcopy(self.default_bn_weights)

        self.model.load_state_dict(bn_weights, strict=False)

        x = x[..., :-1]
        outputs = self.model(x)
        targets = torch.squeeze(y)
        loss = nn.CrossEntropyLoss(weight=self.class_weights)(outputs, targets)

        _, preds = torch.max(outputs.detach(), 1)

        # Log and accumulate
        res = {'loss': loss, 'preds': preds, 'targets': targets}
        self.train_step_outputs.append(res)

        train_metrics_out = self.train_metrics(preds, targets)
        self.log_metrics(train_metrics_out)
        self.log(self.log_prefix + 'train_loss', loss, prog_bar=True)

        return res

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        # Save batch-norm stats for this subject
        state_dict = self.model.state_dict()
        batch_norm_dict = {}
        for key in state_dict:
            if "bnorm" in key:
                batch_norm_dict.update({key: state_dict[key]})
        self.bn_weights_by_subject[self.current_subject_batch] = deepcopy(batch_norm_dict)


class LegacyModel(nn.Module):
    def __init__(self, model, learning_rate, classes, class_weights=None, metrics=None, prev_optimizer_state=None):
        super(LegacyModel, self).__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.classes = classes
        self.class_weights = class_weights
        self.metrics = metrics
        self.prev_optimizer_state = prev_optimizer_state

    def fit(self, train_loader, epoch):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.prev_optimizer_state is not None:
            optimizer.load_state_dict(deepcopy(self.prev_optimizer_state))

        self.model.train()
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = self.model(inputs)
            targets = torch.squeeze(targets)
            loss = nn.CrossEntropyLoss(weight=self.class_weights)(outputs, targets)
            total_loss += loss
            _, predictions = torch.max(torch.nn.Softmax(dim=1)(outputs.detach()), 1)
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print('Epoch: {} {}/{} Training loss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(inputs),
                    len(train_loader.dataset),
                    loss
                ))

        print('Training loss: {:.6f}'.format(total_loss / len(train_loader)))

        return total_loss / len(train_loader)

    def evaluate(self, test_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = self.model(inputs)
                targets = torch.squeeze(targets)
                loss = nn.CrossEntropyLoss()(outputs, targets)
                total_loss += loss
                _, predictions = torch.max(torch.nn.Softmax(dim=1)(outputs), 1)
                correct += predictions.eq(targets.view_as(predictions)).sum()

        test_loss = total_loss / len(test_loader)

        test_percent_acc = 100. * correct / len(test_loader.dataset)
        print('Test loss: {:.6f}, Test accuracy: {}/{} ({:.1f}%)\n'.format(
            test_loss,
            correct,
            len(test_loader.dataset),
            test_percent_acc
        ))

        return test_percent_acc, test_loss

    def on_epoch_end(self, labels, preds):
        """
        Callbacks to perform at the end of an epoch
        """
        res = {}
        if self.metrics is not None:
            res['metrics'] = self.compute_metrics(labels, preds)

        return res

    # def load_state_dict(self, state_dict, strict=True):
    #     self.model.load_state_dict(state_dict, strict)

    def compute_metrics(self, labels, preds):
        """
        Callback to compute performance metrics
        :param labels: Torch tensor of labels
        :param preds: Torch tensor of model predictions
        """
        res = {}
        for metric in self.metrics:
            res[metric.name] = metric.compute(labels, preds)

        print('\n PERFORMANCE METRICS: \n')
        print(res)
        print('\n')

        return res


class CNN_ITER4(nn.Module):
    def __init__(self, model_cfg):
        super(CNN_ITER4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=16, kernel_size=3)
        self.bnormconv = nn.BatchNorm2d(num_features=16)
        self.hidden1 = nn.Linear(144, 256)
        self.bnorm1 = nn.BatchNorm1d(num_features=256)
        self.hidden2 = nn.Linear(256, 128)
        self.bnorm2 = nn.BatchNorm1d(num_features=128)
        self.output = nn.Linear(128, 3)
        self.output_activation = torch.nn.Sigmoid()
        self.dropout = nn.Dropout(model_cfg['dropout'])

    def forward(self, x):
        # x = torch.reshape(x, (x.shape[0], 5, 5, x.shape[-1]))
        x = torch.permute(x, (0, 3, 2, 1))  # Set channels to dim 1
        x = self.conv1(x)
        x = self.bnormconv(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)

        x = self.hidden1(x)
        x = self.bnorm1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.hidden2(x)
        x = self.bnorm2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.output(x)
        x = self.output_activation(x)

        return x


class CNN_ITER3(nn.Module):
    def __init__(self, model_cfg):
        super(CNN_ITER3, self).__init__()

        self._input_batch_norm = nn.BatchNorm2d(4, eps=1e-4)
        self._input_prelu = nn.PReLU(4)

        self._list_conv1_first_part = []
        self._list_conv2_first_part = []
        self._first_part_dropout1 = []
        self._first_part_dropout2 = []
        self._first_part_relu1 = []
        self._first_part_relu2 = []
        self._first_part_batchnorm1 = []
        self._first_part_batchnorm2 = []
        for i in range(2):
            self._list_conv1_first_part.append(nn.Conv2d(2, 12, kernel_size=(4, 3)))
            self._list_conv2_first_part.append(nn.Conv2d(12, 12, kernel_size=(1, 3)))

            self._first_part_dropout1.append(nn.Dropout2d(model_cfg['dropout']))
            self._first_part_dropout2.append(nn.Dropout2d(model_cfg['dropout']))

            self._first_part_relu1.append(nn.PReLU(12))
            self._first_part_relu2.append(nn.PReLU(12))

            self._first_part_batchnorm1.append(nn.BatchNorm2d(12, eps=1e-4))
            self._first_part_batchnorm2.append(nn.BatchNorm2d(12, eps=1e-4))

        self._list_conv1_first_part = nn.ModuleList(self._list_conv1_first_part)
        self._list_conv2_first_part = nn.ModuleList(self._list_conv2_first_part)
        self._first_part_dropout1 = nn.ModuleList(self._first_part_dropout1)
        self._first_part_dropout2 = nn.ModuleList(self._first_part_dropout2)
        self._first_part_relu1 = nn.ModuleList(self._first_part_relu1)
        self._first_part_relu2 = nn.ModuleList(self._first_part_relu2)
        self._first_part_batchnorm1 = nn.ModuleList(self._first_part_batchnorm1)
        self._first_part_batchnorm2 = nn.ModuleList(self._first_part_batchnorm2)

        self._conv3 = nn.Conv2d(12, 24, kernel_size=(2, 3))
        self._batch_norm_3 = nn.BatchNorm2d(24, eps=1e-4)
        self._prelu_3 = nn.PReLU(24)
        self._dropout3 = nn.Dropout2d(model_cfg['dropout'])

        self._fc1 = nn.Linear(96, 100)
        self._batch_norm_fc1 = nn.BatchNorm1d(100, eps=1e-4)
        self._prelu_fc1 = nn.PReLU(100)
        self._dropout_fc1 = nn.Dropout(model_cfg['dropout'])

        self._fc2 = nn.Linear(100, 100)
        self._batch_norm_fc2 = nn.BatchNorm1d(100, eps=1e-4)
        self._prelu_fc2 = nn.PReLU(100)
        self._dropout_fc2 = nn.Dropout(model_cfg['dropout'])

        self._output = nn.Linear(100, 3)

        self._output_activation = nn.Sigmoid()

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def forward(self, x):
        x = self._input_batch_norm(x)
        x = self._input_prelu(x)

        input_1 = x[:, :2, :, :]
        input_2 = x[:, 2:, :, :]

        first_branch = self.first_parallel(input_1, 0)
        second_branch = self.first_parallel(input_2, 1)

        first_merge_1 = first_branch + second_branch

        after_conv = self._dropout3(self._prelu_3(self._batch_norm_3(self._conv3(first_merge_1))))

        flatten_tensor = after_conv.view(-1, 96)

        fc1_output = self._dropout_fc1(self._prelu_fc1(self._batch_norm_fc1(self._fc1(flatten_tensor))))

        fc2_output = self._dropout_fc2(self._prelu_fc2(self._batch_norm_fc2(self._fc2(fc1_output))))

        out = self._output(fc2_output)
        out = self._output_activation(out)

        return out

    def first_parallel(self, input_to_give, index):
        conv1_first_part1 = self._list_conv1_first_part[index](input_to_give)
        batch_norm1_first_part1 = self._first_part_batchnorm1[index](conv1_first_part1)
        prelu1_first_part1 = self._first_part_relu1[index](batch_norm1_first_part1)
        dropout1_first_part1 = self._first_part_dropout1[index](prelu1_first_part1)

        conv1_first_part2 = self._list_conv2_first_part[index](dropout1_first_part1)
        batch_norm1_first_part2 = self._first_part_batchnorm2[index](conv1_first_part2)
        prelu1_first_part2 = self._first_part_relu2[index](batch_norm1_first_part2)
        dropout1_first_part2 = self._first_part_dropout2[index](prelu1_first_part2)

        return dropout1_first_part2


class CNN_ITER2(nn.Module):
    def __init__(self, model_cfg):
        super(CNN_ITER2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(2, 5))
        self.bnorm1 = nn.BatchNorm2d(num_features=32)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 3))
        self.prelu1 = nn.PReLU(32)
        self.dropout1 = nn.Dropout2d(model_cfg['dropout'])

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(2, 5))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 3))
        self.bnorm2 = nn.BatchNorm2d(64)
        self.prelu2 = nn.PReLU(64)
        self.dropout2 = nn.Dropout2d(model_cfg['dropout'])

        self.fc1 = nn.Linear(768, 500)
        self.bnorm3 = nn.BatchNorm1d(500)
        self.prelu3 = nn.PReLU(500)
        self.dropout3 = nn.Dropout(model_cfg['dropout'])

        self.output = nn.Linear(500, 3)
        self.output_activation = torch.nn.Sigmoid()

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))  # Set electrode channels to rows dim (1)
        x = torch.unsqueeze(x, 1)  # Singular channel image
        x = self.conv1(x)
        x = self.bnorm1(x)
        x = self.prelu1(x)
        x = self.dropout1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bnorm2(x)
        x = self.prelu2(x)
        x = self.dropout2(x)
        x = self.pool2(x)

        # Flatten
        x = x.view(-1, 768)

        x = self.fc1(x)
        x = self.bnorm3(x)
        x = self.prelu3(x)
        x = self.dropout3(x)

        x = self.output(x)
        x = self.output_activation(x)

        return x


class CNN(nn.Module):
    def __init__(self, model_cfg):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=14, out_channels=64, kernel_size=7)
        # self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.bnormconv1 = nn.BatchNorm2d(num_features=64)
        self.dropout1 = nn.Dropout2d(model_cfg['dropout'])

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7)
        # self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.bnormconv2 = nn.BatchNorm2d(num_features=128)
        self.dropout2 = nn.Dropout2d(model_cfg['dropout'])

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5)
        self.bnormconv3 = nn.BatchNorm2d(num_features=64)
        self.dropout3 = nn.Dropout2d(model_cfg['dropout'])

        self.hidden1 = nn.Linear(135424, 512)
        self.bnorm1 = nn.BatchNorm1d(num_features=512)

        self.hidden2 = nn.Linear(512, 256)
        self.bnorm2 = nn.BatchNorm1d(num_features=256)

        self.output = nn.Linear(256, 4)
        self.output_activation = torch.nn.Softmax()

        self.dropout = nn.Dropout(model_cfg['dropout'])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bnormconv1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bnormconv2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.bnormconv3(x)
        x = F.relu(x)
        x = self.dropout3(x)

        x = x.reshape(-1, 135424)

        x = self.hidden1(x)
        x = self.bnorm1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.hidden2(x)
        x = self.bnorm2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.output(x)
        x = self.output_activation(x)

        return x


class MLP_ITER2(nn.Module):
    def __init__(self, model_cfg):
        super(MLP_ITER2, self).__init__()
        input_size = 40
        self.hidden1 = nn.Linear(input_size, 512)
        self.hidden2 = nn.Linear(512, 256)
        self.hidden3 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(model_cfg['dropout'])
        self.output = nn.Linear(128, 3)
        self.output_activation = torch.nn.Sigmoid()

    def forward(self, x):
        x = torch.flatten(x, 1)

        x = self.hidden1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.hidden2(x)
        x = F.relu(x)

        x = self.hidden3(x)
        x = self.dropout(x)
        x = F.relu(x)

        x = self.output(x)
        x = self.output_activation(x)

        return x


class MLP(nn.Module):
    def __init__(self, model_cfg):
        super(MLP, self).__init__()
        input_size = 5
        self.hidden1 = nn.Linear(input_size, 128)
        self.hidden2 = nn.Linear(128, 64)
        self.hidden3 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(model_cfg['dropout'])
        self.output = nn.Linear(32, 4)

        # def init_weights(m):
        #     if isinstance(m, nn.Linear):
        #         torch.nn.init.xavier_uniform_(m.weight)
        #         m.bias.data.fill_(0)
        #
        # self.apply(init_weights)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.hidden1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = self.dropout(x)
        x = self.hidden3(x)
        x = F.relu(x)

        return self.output(x)
