from typing import List, Any, Dict

import torch
import tqdm
from torch.utils import data
import pandas as pd
import numpy as np
import time
import collections
from torch import nn
import copy

from classificationutils import ClassificationResults
from learning_utils import ImageDataset, LabeledImageDataset
from IPython.display import clear_output
from IPython.display import display

try:
    import cPickle as pickle
except ImportError:
    import pickle

import signal


class PrintLayer(nn.Module):
    def __init__(self, msg=""):
        super(PrintLayer, self).__init__()
        self.msg = msg

    def forward(self, x):
        # Do your print / debug stuff here
        print(self.msg, x.shape)
        return x


# Build the neural network, expand on top of nn.Module
class CNN(nn.Module):
    def __init__(self, dataset_sample=None, model_type=0,
                 input_dims=1, output_classes=2, dense_input_dims=576, padding=0):
        """

        Args:
            dataset_sample (LabeledImageDataset):
             If provided dataset automatically configures the model.
            model_type: One of 0 or 1 or 2.
            input_dims: The number of channels for the input images. Not needed if dataset_sample is provided.
            output_classes: How many output classes there are.
            dense_input_dims: The input dimensions of the dense part of this model. Not needed if dataset_sample is
                provided.
            padding:
        """
        super().__init__()
        assert model_type in [0, 1, 2]

        if dataset_sample:
            batch_sample = dataset_sample[0]
            image_sample, label_sample = batch_sample
            # a tensor image is CxHxW
            input_dims = image_sample.shape[0]

        if model_type == 0:
            self.convolutional = nn.Sequential(
                nn.Conv2d(input_dims, 32, padding=2, kernel_size=5),
                nn.BatchNorm2d(32),
                nn.MaxPool2d(kernel_size=(3, 3), stride=2),

                nn.Conv2d(32, 32, padding=2, kernel_size=5),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=3, padding=1, stride=2),

                nn.Conv2d(32, 64, padding=2, kernel_size=5),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=3, padding=1, stride=2),
            )
        elif model_type == 1:
            self.convolutional = nn.Sequential(
                nn.Conv2d(input_dims, 16, padding=2, kernel_size=5),
                nn.BatchNorm2d(16),
                nn.MaxPool2d(kernel_size=3, stride=2),

                nn.Conv2d(16, 32, padding=2, kernel_size=5),
                nn.BatchNorm2d(32),
                nn.MaxPool2d(kernel_size=3, stride=2),

                nn.Conv2d(32, 32, padding=2, kernel_size=5),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=3, padding=1, stride=2),

                nn.Conv2d(32, 64, padding=2, kernel_size=5),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=3, padding=1, stride=2),
            )
        elif model_type == 2:
            self.convolutional = nn.Sequential(
                nn.Conv2d(input_dims, 32, padding=2, kernel_size=5),
                nn.BatchNorm2d(32),
                nn.MaxPool2d(kernel_size=(3, 3), stride=2),
                nn.ReLU(),
                nn.Dropout(),

                nn.Conv2d(32, 64, padding=2, kernel_size=5),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Dropout(),
                nn.AvgPool2d(kernel_size=3, padding=1, stride=2),
            )
        elif model_type == 3:
            self.convolutional = nn.Sequential(
                nn.Conv2d(input_dims, 32, padding=2, kernel_size=5),
                nn.BatchNorm2d(32),
                nn.MaxPool2d(kernel_size=(3, 3), stride=2),
                nn.ReLU(),
                nn.Dropout(),

                nn.Conv2d(32, 32, padding=2, kernel_size=5),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Dropout(),

                nn.AvgPool2d(kernel_size=3, padding=1, stride=2),
                nn.Conv2d(32, 64, padding=2, kernel_size=5),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Dropout(),
                nn.AvgPool2d(kernel_size=3, padding=1, stride=2),
            )

        # determine the input dimensions needed for the dense part of the model.
        if dataset_sample:
            # In order for the image sample to work we must append a batch number dimension
            # HxWxC -> 1xHxWxC
            image_batch = image_sample[None, ...].cpu()
            batch_size = 1

            with torch.no_grad():
                # output shape is 1 x output C x out H x out W.
                # to get the dense input dimensions we get the convolutional output and
                # reshape so every dimension other than batch size is multiplied together.
                convolutional_output = self.convolutional(image_batch)
                dense_input_dims = convolutional_output.reshape(batch_size, -1).shape[-1]
        self.dense_input_dims = dense_input_dims

        if model_type == 0:
            self.dense = nn.Sequential(
                nn.Linear(dense_input_dims, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(),

                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(),

                nn.Linear(32, output_classes)
            )
        elif model_type in [1, 2]:
            self.dense = nn.Sequential(
                nn.Linear(dense_input_dims, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(),

                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(),

                nn.Linear(32, 16),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Dropout(),

                nn.Linear(16, 8),
                nn.BatchNorm1d(8),
                nn.ReLU(),

                nn.Linear(8, output_classes)
            )

    # define forward function
    def forward(self, t):
        batch_size = len(t)
        t = self.convolutional(t)
        t = t.reshape(batch_size, -1)
        t = self.dense(t)
        return t


# Helper class, help track loss, accuracy, epoch time, run time,
# hyper-parameters etc.
class TrainingTracker:
    recorded_models: Dict[int, collections.OrderedDict]
    train_classification_results: Dict[int, ClassificationResults]
    valid_classification_results: Dict[int, ClassificationResults]
    best_train_accuracy_classification_results: ClassificationResults
    best_valid_accuracy_classification_results: ClassificationResults
    additional_displays: List[object]

    def __init__(self, device, additional_displays=None):
        self.recorded_models = {}
        if additional_displays is None:
            additional_displays = []
        self.manually_stopped = False

        self.additional_displays = additional_displays
        self.epoch_count = 0
        self.epoch_start_time = None
        self.epoch_duration = None

        self.run_start_time = None
        self.run_duration = None
        self.run_data = []

        # track every loss and performance
        self.train_losses = {}
        self.valid_losses = {}

        self.train_accuracies = {}
        self.valid_accuracies = {}

        self.train_positive_accuracies = {}
        self.valid_positive_accuracies = {}

        self.train_negative_accuracies = {}
        self.valid_negative_accuracies = {}

        self.train_classification_results = {}
        self.valid_classification_results = {}

        # Track training performance metrics
        self.epoch_durations = []

        # training dataset
        self.best_train_loss = np.inf
        self.is_best_train_loss_recorded = False
        self.best_train_accuracy_epoch = 0

        self.best_train_accuracy = 0
        self.is_best_train_accuracy_recorded = False

        self._times_since_last_best_train_loss = 0
        self._times_since_last_best_train_accuracy = 0
        self.best_train_accuracy_classification_results = None

        # validation dataset
        self.best_valid_loss = np.inf
        self.is_best_valid_loss_recorded = False

        self.best_valid_accuracy = 0
        self.best_valid_accuracy_epoch = 0
        self.is_best_valid_accuracy_recorded = False
        self.best_valid_accuracy_classification_results = None

        self._times_since_last_best_valid_loss = 0
        self._times_since_last_best_valid_accuracy = 0

        # tracking every run count, run data, hyper-params used, time
        self.run_params = None
        self.run_start_time = None

        # Model is updated each epoch
        self.model = None

        # Recorded model is updated each time record_model() is called
        self.recorded_model = None
        self.recorded_model_weights = None
        self.recorded_model_epoch = None
        self.recorded_model_valid_accuracy = None
        self.recorded_model_valid_loss = None
        self.recorded_model_train_accuracy = None
        self.recorded_model_train_loss = None
        self.is_model_recorded = False

        self.recorded_model_train_classification_results = None
        self.recorded_model_valid_classification_results = None

        # Loaders and loss criterion used for this run
        self.trainset = None
        self.validset = None
        self.train_loader = None
        self.valid_loader = None
        self.criterion = None

        # 'cpu' or 'cuda'
        self.device = device

    def start_run(self, model, params, train_loader, valid_loader, criterion, trainset=None, validset=None):
        self.run_start_time = time.time()
        if 'do_early_stop' in params:
            self.do_early_stop = params['do_early_stop']
        if 'early_stop_patience' in params:
            self.early_stop_patience = params['early_stop_patience']

        self.run_params = params
        self.model = model

        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.trainset = trainset
        self.validset = validset

        self.criterion = criterion

    def end_run(self):
        self.run_duration = time.time() - self.run_start_time
        print("Run duration ", self.run_duration)

    def start_epoch(self):
        self.epoch_start_time = time.time()

    def end_epoch(self):
        self.epoch_count += 1
        self.epoch_duration = time.time() - self.epoch_start_time

        self.is_best_valid_accuracy_recorded = False
        self.is_best_valid_loss_recorded = False
        self.is_model_recorded = False

    def display_results(self):
        # Write into 'results' (OrderedDict) for all run related data
        results = collections.OrderedDict()
        results['e'] = self.epoch_count
        # record epoch loss and accuracy

        if self.train_losses:
            results['train loss'] = f'{self.train_losses[self.epoch_count]:.1E}'
        if self.train_accuracies:
            results['train acc'] = self.train_accuracies[self.epoch_count]
        if self.train_classification_results:
            results['train balanced acc'] = self.train_classification_results[self.epoch_count].balanced_accuracy
        if self.train_positive_accuracies:
            results['train pos acc'] = self.train_positive_accuracies[self.epoch_count]
        if self.train_negative_accuracies:
            results['train neg acc'] = self.train_negative_accuracies[self.epoch_count]

        if self.valid_losses:
            results['valid loss'] = f'{self.valid_losses[self.epoch_count]:.1E}'
        if self.valid_accuracies:
            results['valid acc'] = self.valid_accuracies[self.epoch_count]
        if self.valid_classification_results:
            results['valid balanced acc'] = self.valid_classification_results[self.epoch_count].balanced_accuracy
        if self.valid_positive_accuracies:
            results['valid pos acc'] = self.valid_positive_accuracies[self.epoch_count]
        if self.valid_negative_accuracies:
            results['valid neg acc'] = self.valid_negative_accuracies[self.epoch_count]

        # results['Best loss?'] = self.is_best_valid_loss_recorded
        # results['Best Acc?'] = self.is_best_valid_accuracy_recorded
        # results["Model Recorded?"] = self.is_model_recorded

        if self.do_early_stop:
            results['lst bst loss e'] = self._times_since_last_best_valid_loss
            results['lst bst acc  e'] = self._times_since_last_best_valid_accuracy

        run_parameters = collections.OrderedDict()
        for param_group in self.run_params['optimizer'].param_groups:
            run_parameters['lr'] = f'{param_group["lr"]:.2E}'
            run_parameters['wd'] = param_group["weight_decay"]
            results['lr'] = f"{param_group['lr']:.2E}"

        # Record hyper-params into 'results'
        for k, v in self.run_params.items():
            if k in ['batch_size',
                     'epochs', 'shuffle']:
                run_parameters[k] = v
            elif k == 'learning_rate_scheduler_patience':
                run_parameters['lr patience'] = v
            elif k == 'early_stop_patience':
                run_parameters['stop patience'] = v
            elif k == 'trainset':
                run_parameters['Trainset size'] = len(self.run_params[k])
            elif k == 'validset':
                run_parameters['Validset size '] = len(self.run_params[k])
            elif k not in ['trainset', 'validset', 'testset', 'optimizer', 'n_negatives_per_positive',
                           'lr', 'weight_decay', 'do_early_stop', 'evaluation_epochs']:
                # The columns I don't want to see each epoch
                results[k] = v

        self.run_data.append(results)
        run_parameters_df = pd.DataFrame.from_dict([run_parameters], orient='columns')
        run_performance_df = pd.DataFrame.from_dict(self.run_data, orient='columns')

        current_valid_performance_df = pd.DataFrame(
            collections.OrderedDict({
                'Best valid acc': self.best_valid_accuracy,
                'loss': f'{self.best_train_loss:.2E}',
                'valid pos acc': self.valid_positive_accuracies[self.best_valid_accuracy_epoch],
                'valid neg acc': self.valid_negative_accuracies[self.best_valid_accuracy_epoch],
                'train pos acc': self.train_positive_accuracies[self.best_valid_accuracy_epoch],
                'train neg acc': self.train_negative_accuracies[self.best_valid_accuracy_epoch],
                'epoch': self.best_train_accuracy_epoch,
            }), index=[0])

        current_train_performance_df = pd.DataFrame(
            collections.OrderedDict({
                'Best train acc': self.best_train_accuracy,
                'loss': f'{self.best_train_loss:.2E}',
                'valid pos acc': self.valid_positive_accuracies[self.best_train_accuracy_epoch],
                'valid neg acc': self.valid_negative_accuracies[self.best_train_accuracy_epoch],
                'train pos acc': self.train_positive_accuracies[self.best_train_accuracy_epoch],
                'train neg acc': self.train_negative_accuracies[self.best_train_accuracy_epoch],
                'epoch': self.best_train_accuracy_epoch,
            }), index=[0])

        # display epoch information and show progress
        with pd.option_context('display.max_rows', 7,
                               'display.max_colwidth', 30,
                               # 'display.float_format', '{:.2E}'.format,
                               'display.max_columns', None,
                               'display.width', 1000):  # more options can be specified also
            clear_output()
            for additional_display in self.additional_displays:
                if type(additional_display) == dict or type(additional_display) == collections.OrderedDict:
                    display(pd.DataFrame(additional_display, index=[0]))
                else:
                    display(additional_display)

            display(current_valid_performance_df)
            display(current_train_performance_df)

            display(run_parameters_df)
            display(run_performance_df)

    # noinspection PyUnresolvedReferences
    @torch.no_grad()
    def load(self, filename, input_dims):
        model = CNN(input_dims=input_dims)
        model.load_state_dict(torch.load(filename))
        model.eval()

    @torch.no_grad()
    def save(self, output_directory, v=False):
        """ Saves the recorded model as {output_name}.pt among other info files.

        Makes {output_name}.pt, {output_name}.txt with recorded epoch loss, accuracy and other run parameters
        and {output_name}_run_parameters.txt with the hyper parameters that the network was ran (learning_rate, patience
        e.t.c)

        Args:
            output_directory (string): The output directory name.
        """
        import os.path
        import pathlib
        pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)

        if v:
            print(f'Saving to {output_directory}')

        # https: // pytorch.org / tutorials / beginner / saving_loading_models.html  # save-load-state-dict-recommended
        for props_name, props in self.recorded_models.items():
            output_file = os.path.join(output_directory, f'{props_name}_model.pt')
            torch.save(props['model'].state_dict(), output_file)
            if v:
                print(f'Saved {output_file}')

        run_parameters = collections.OrderedDict()
        for param_group in self.run_params['optimizer'].param_groups:
            run_parameters['learning rate'] = param_group['lr']

        # Record hyper-params into 'results'
        for k, v in self.run_params.items():
            if k in ['batch_size', 'do_early_stop', 'early_stop_patience',
                     'learning_rate_scheduler_patience', 'epochs', 'shuffle']:
                run_parameters[k] = v

        output_file = os.path.join(output_directory, 'run_data.txt')
        run_data_df = pd.DataFrame.from_dict(self.run_data, orient='columns')
        run_data_df.to_csv(output_file)
        if v:
            print(f'Saved {output_file}')

        output_file = os.path.join(output_directory, 'run_params.txt')
        run_parameters_df = pd.DataFrame.from_dict([run_parameters], orient='columns')
        run_parameters_df.to_csv(output_file)
        if v:
            print(f'Saved {output_file}')

        output_file = os.path.join(output_directory, 'results.pkl')
        with open(output_file, 'wb') as f:
            self.model = copy.deepcopy(self.model)
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            if v:
                print(f'Saved {output_file}')

    @classmethod
    def from_file(cls, file):
        with open(file, 'rb') as input_file:
            return pickle.load(input_file)

    # noinspection DuplicatedCode
    @torch.no_grad()
    def track_performance(self):
        classification_results = \
            self._get_loss_and_accuracy(self.train_loader)
        self.train_losses[self.epoch_count] = classification_results.loss
        self.train_accuracies[self.epoch_count] = classification_results.accuracy
        self.train_positive_accuracies[self.epoch_count] = classification_results.positive_accuracy
        self.train_negative_accuracies[self.epoch_count] = classification_results.negative_accuracy
        self.train_classification_results[self.epoch_count] = copy.deepcopy(classification_results)

        if classification_results.loss < self.best_train_loss:
            self.is_best_train_loss_recorded = True
            self.best_train_loss = classification_results.loss
            self._times_since_last_best_train_loss = 0
        else:
            self.is_best_train_loss_recorded = False
            self._times_since_last_best_train_loss += 1

        if classification_results.accuracy > self.best_train_accuracy:
            self.is_best_train_accuracy_recorded = True
            self.best_train_accuracy = classification_results.accuracy
            self.best_train_accuracy_epoch = self.epoch_count
            self.best_train_accuracy_classification_results = copy.deepcopy(classification_results)
            self._times_since_last_best_train_accuracy = 0
        else:
            self.is_best_train_accuracy_recorded = False
            self._times_since_last_best_train_accuracy += 1

        classification_results = self._get_loss_and_accuracy(self.valid_loader)
        self.valid_losses[self.epoch_count] = classification_results.loss
        self.valid_accuracies[self.epoch_count] = classification_results.accuracy
        self.valid_positive_accuracies[self.epoch_count] = classification_results.positive_accuracy
        self.valid_negative_accuracies[self.epoch_count] = classification_results.negative_accuracy
        self.valid_classification_results[self.epoch_count] = copy.deepcopy(classification_results)

        if classification_results.loss < self.best_valid_loss:
            self.is_best_valid_loss_recorded = True
            self.best_valid_loss = classification_results.loss
            self._times_since_last_best_valid_loss = 0
        else:
            self.is_best_valid_loss_recorded = False
            self._times_since_last_best_valid_loss += 1

        if classification_results.accuracy > self.best_valid_accuracy:
            self.is_best_valid_accuracy_recorded = True
            self.best_valid_accuracy = classification_results.accuracy
            self.best_valid_accuracy_epoch = self.epoch_count
            self.best_valid_accuracy_classification_results = copy.deepcopy(classification_results)
            self._times_since_last_best_valid_accuracy = 0
        else:
            self.is_best_valid_accuracy_recorded = False
            self._times_since_last_best_valid_accuracy += 1

    def get_last_valid_accuracy(self):
        return self.valid_accuracies[self.epoch_count]

    def get_last_valid_loss(self):
        return self.valid_losses[self.epoch_count]

    # noinspection DuplicatedCode
    def record_model(self, model_name='recorded_model'):
        model = self.model.eval()

        self.recorded_model_weights = copy.deepcopy(model.state_dict())
        self.recorded_model = copy.deepcopy(model)
        self.recorded_model = self.recorded_model.eval()

        self.recorded_model_epoch = self.epoch_count

        self.recorded_model_valid_accuracy = self.valid_accuracies[self.epoch_count]
        self.recorded_model_valid_loss = self.valid_losses[self.epoch_count]

        self.recorded_model_train_accuracy = self.train_accuracies[self.epoch_count]
        self.recorded_model_train_loss = self.train_losses[self.epoch_count]

        self.recorded_model_train_classification_results = self.train_classification_results[self.epoch_count]
        self.recorded_model_valid_classification_results = self.valid_classification_results[self.epoch_count]

        self.recorded_models[model_name] = collections.OrderedDict(
            weights=self.recorded_model_weights,
            model=copy.deepcopy(self.recorded_model),
            epoch=self.recorded_model_epoch,

            train_classification_results=copy.deepcopy(self.recorded_model_train_classification_results),
            valid_classification_results=copy.deepcopy(self.recorded_model_valid_classification_results),

            valid_accuracy=self.recorded_model_valid_accuracy,
            train_accuracy=self.recorded_model_train_accuracy,

            valid_loss=self.recorded_model_valid_loss,
            train_loss=self.recorded_model_train_loss,
        )

        self.is_model_recorded = True
        self.model = self.model.train()

    def should_early_stop(self):
        return self.do_early_stop and self._times_since_last_best_valid_loss >= self.early_stop_patience

    @torch.no_grad()
    def _get_accuracy(self, loader):
        n_samples = 0
        n_correct = 0

        for batch in loader:
            images = batch[0].to(self.device)
            targets = batch[1].to(self.device).type(torch.long)

            output = self.model(images)
            predictions = torch.zeros_like(output, dtype=torch.long)
            predictions[output >= 0.5] = 1

            n_samples += images.shape[0]
            n_correct += torch.sum(predictions == targets).item()

        return n_samples / n_correct

    @torch.no_grad()
    def _get_loss(self, loader):
        total_loss = 0
        n_samples = 0

        for batch in loader:
            images = batch[0].to(self.device)
            targets = batch[1].to(self.device)

            # print(images.device)
            # print(targets.device)
            predictions = self.model(images)
            loss = self.criterion(predictions, targets)

            n_samples += targets.shape[0]
            total_loss += loss.item()

        total_loss /= n_samples

        return total_loss

    @torch.no_grad()
    def _get_loss_and_accuracy(self, loader):
        self.model = self.model.eval()
        from classificationutils import classify_labeled_dataset

        total_loss = 0

        n_samples = 0
        n_correct = 0
        n_positive_correct = 0
        n_positive_samples = 0
        n_negative_correct = 0
        n_negative_samples = 0
        for batch in loader:
            images = batch[0].to(self.device)
            targets = batch[1].to(self.device).type(torch.long)

            output = self.model(images)
            loss = self.criterion(output, targets)

            total_loss += loss.item()

            predictions = torch.argmax(output, dim=1).type(torch.int)
            n_correct += (predictions == targets).sum().item()

            positive_indices = torch.where(targets == 1)[0]
            n_positive_samples += len(positive_indices)
            n_positive_correct += (predictions[positive_indices] == targets[positive_indices]).sum().item()

            negative_indices = torch.where(targets == 0)[0]
            n_negative_samples += len(negative_indices)
            n_negative_correct += (predictions[negative_indices] == targets[negative_indices]).sum().item()

            n_samples += len(targets)

        total_loss /= n_samples
        accuracy = n_correct / n_samples

        positive_accuracy = n_positive_correct / n_positive_samples
        negative_accuracy = n_negative_correct / n_negative_samples

        self.model = self.model.train()
        classification_results = ClassificationResults(
            loss=total_loss,

            accuracy=accuracy,
            positive_accuracy=positive_accuracy,
            negative_accuracy=negative_accuracy,
            balanced_accuracy=(positive_accuracy + negative_accuracy) / 2,

            n_positive=n_positive_samples,
            n_negative=n_negative_samples,
        )

        return classification_results


class TrainingInterruptSignalHandler(object):
    tracker: TrainingTracker

    def __init__(self, tracker):
        """

        Args:
            tracker (TrainingTracker):
        """
        self.tracker = tracker
        self.signal_raised = False

    def handle(self, sig, frm):
        print(f'Signal {sig} captured.')
        self.tracker.manually_stopped = True
        self.signal_raised = True


def create_weights_for_balanced_sampling(labeled_dataset):
    # thanks to Jordi De La Torre
    # https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/2
    loader = data.DataLoader(labeled_dataset, batch_size=len(labeled_dataset))

    for _, labels in loader:
        n_classes = len(torch.unique(labels))

        class_counts = [0] * n_classes
        for class_label in range(n_classes):
            class_counts[class_label] = len(torch.where(labels == class_label)[0])

        weight_per_class = [.0] * n_classes
        n_samples = sum(class_counts)
        for i in range(n_classes):
            weight_per_class[i] = n_samples / float(class_counts[i])

        weights = [0] * len(labeled_dataset)
        for sample_idx, (_, lbl) in enumerate(labeled_dataset):
            weights[sample_idx] = weight_per_class[lbl]

        return weights


def train(cnn, params, criterion=torch.nn.CrossEntropyLoss(), device='cuda', additional_displays=None):
    # if params changes, following line of code should reflect the changes too
    if additional_displays is None:
        additional_displays = []

    batch_size = int(0.75 * len(params['trainset']))
    if 'batch_size' in params:
        if params['batch_size'] in [None, 'all']:
            batch_size = len(params['trainset'])
        else:
            batch_size = params['batch_size']

    balanced_sampling = True
    if 'balanced_sampling' in params:
        balanced_sampling = params['balanced_sampling']

    trainset = params['trainset']
    if balanced_sampling:
        # thanks to Jordi De La Torre
        # https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/2
        weights = create_weights_for_balanced_sampling(trainset)
        weights = torch.tensor(weights, dtype=torch.double)
        sampler = data.WeightedRandomSampler(weights, len(weights), replacement=True)
    else:
        sampler = data.RandomSampler(trainset, replacement=True)

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        sampler=sampler,
    )

    validset = None
    valid_loader = None
    if 'validset' in  params:
        validset = params['validset']
        valid_loader = torch.utils.data.DataLoader(
            validset,
            batch_size=batch_size,
            shuffle=False
        )

    epochs = params['epochs']
    if 'evaluation_epochs' in params:
        evaluation_epochs = params['evaluation_epochs']
    else:
        evaluation_epochs = 20
    # Set up Optimizer
    if 'optimizer' not in params:
        lr = .001
        weight_decay = 5e-4
        if 'lr' in params:
            lr = params['lr']
        if 'weight_decay' in params:
            weight_decay = params['weight_decay']
        params['optimizer'] = torch.optim.Adam(cnn.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = params['optimizer']

    # Set up learning rate scheduler
    if 'learning_rate_scheduler_patience' not in params:
        if 'early_stop_patience' in params:
            params['learning_rate_scheduler_patience'] = int(0.5 * params['early_stop_patience'])
        else:
            params['learning_rate_scheduler_patience'] = 10
    learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                         'min',
                                                                         patience=params[
                                                                             'learning_rate_scheduler_patience'])

    # Tracker tracks the process and helps with early stopping
    tracker = TrainingTracker(device, additional_displays)
    tracker.start_run(cnn, params, train_loader, valid_loader, criterion, trainset=trainset, validset=validset)

    interrupt_handler = TrainingInterruptSignalHandler(tracker)
    signal.signal(signal.SIGINT, interrupt_handler.handle)

    for epoch in tqdm.tqdm(range(epochs)):
        if tracker.should_early_stop() or interrupt_handler.signal_raised:
            break

        tracker.start_epoch()

        cnn = cnn.train()

        total_loss = 0
        n_samples = 0
        n_correct = 0
        for images, labels in train_loader:
            # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
            optimizer.zero_grad()

            # as per good practice we cast to device to save on gpu memory
            images = images.to(device).type(torch.float32)
            labels = labels.to(device).type(torch.long)

            # print(f'N positive {len(torch.where(labels == 1)[0])}')
            # print(f'N negative {len(torch.where(labels == 0)[0])}')

            output = cnn(images).to(device).type(torch.float32)

            loss = criterion(output, labels)

            predictions = torch.argmax(output, dim=1)

            n_samples += images.shape[0]
            n_correct += torch.sum(predictions == labels).item()

            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            total_loss /= n_samples

        accuracy = n_correct / n_samples
        if epoch % evaluation_epochs == 0 or epoch == epochs - 1:
            # tracker.track_loss_and_accuracy(train_loss=total_loss, train_accuracy=accuracy)
            tracker.track_performance()

            if tracker.is_best_train_accuracy_recorded:
                tracker.record_model('best_train_accuracy')

            if tracker.is_best_valid_accuracy_recorded:
                tracker.record_model('best_valid_accuracy')

            prev_valid_balanced_accuracies = [res.balanced_accuracy for res in list(tracker.valid_classification_results.values())[:-1]]
            cur_valid_balanced_accuracy = list(tracker.valid_classification_results.values())[-1].balanced_accuracy
            if len(prev_valid_balanced_accuracies) == 0 or cur_valid_balanced_accuracy > max(prev_valid_balanced_accuracies):
                tracker.record_model('best_valid_balanced_accuracy')

            prev_train_balanced_accuracies = [res.balanced_accuracy for res in list(tracker.train_classification_results.values())[:-1]]
            cur_train_balanced_accuracy = list(tracker.train_classification_results.values())[-1].balanced_accuracy
            if len(prev_train_balanced_accuracies) == 0 or cur_train_balanced_accuracy > max(prev_train_balanced_accuracies):
                tracker.record_model('best_train_balanced_accuracy')

            prev_valid_positive_accuracies = [res.positive_accuracy for res in list(tracker.valid_classification_results.values())[:-1]]
            cur_valid_positive_accuracy = list(tracker.valid_classification_results.values())[-1].positive_accuracy
            if len(prev_valid_positive_accuracies) == 0 or cur_valid_positive_accuracy > max(prev_valid_positive_accuracies):
                tracker.record_model('best_valid_positive_accuracy')

            prev_train_positive_accuracies = [res.positive_accuracy for res in list(tracker.train_classification_results.values())[:-1]]
            cur_train_positive_accuracy = list(tracker.train_classification_results.values())[-1].positive_accuracy
            if len(prev_train_positive_accuracies) == 0 or cur_train_positive_accuracy > max(prev_train_positive_accuracies):
                tracker.record_model('best_train_positive_accuracy')

            prev_valid_negative_accuracies = [res.negative_accuracy for res in list(tracker.valid_classification_results.values())[:-1]]
            cur_valid_negative_accuracy = list(tracker.valid_classification_results.values())[-1].negative_accuracy
            if len(prev_valid_negative_accuracies) == 0 or cur_valid_negative_accuracy > max(prev_valid_negative_accuracies):
                tracker.record_model('best_valid_negative_accuracy')

            prev_train_negative_accuracies = [res.negative_accuracy for res in list(tracker.train_classification_results.values())[:-1]]
            cur_train_negative_accuracy = list(tracker.train_classification_results.values())[-1].negative_accuracy
            if len(prev_train_negative_accuracies) == 0 or cur_train_negative_accuracy > max(prev_train_negative_accuracies):
                tracker.record_model('best_train_negative_accuracy')

            tracker.display_results()

            # Learning rate scheduler based on accuracy
            learning_rate_scheduler.step(tracker.get_last_valid_accuracy())

        tracker.end_epoch()

    tracker.end_run()

    return tracker


if __name__ == '__main__':
    from generate_datasets import get_cell_and_no_cell_patches

    # Input

    try_load_from_cache = False
    verbose = False
    very_verbose = True
    trainset, validset, _, _, _, _, _ = get_cell_and_no_cell_patches(
        patch_size=21,
        do_hist_match=False,
        n_negatives_per_positive=1,
        standardize_dataset=True,
        temporal_width=1,
        try_load_from_cache=True,
        v=False,
        vv=False
    )
    loader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False)
    CNN(dataset_sample=trainset).to('cuda')

    for ims, lbls in loader:
        output = CNN(ims)
