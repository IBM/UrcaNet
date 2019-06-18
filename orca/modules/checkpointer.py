from typing import Dict, Optional, List, Tuple, Union, Iterable, Any, NamedTuple

import logging
import os
import re
import shutil
import time

import torch

from allennlp.training.trainer import Trainer, TrainerPieces
from allennlp.training.trainer_base import TrainerBase
from allennlp.training.checkpointer import Checkpointer
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import (dump_metrics, gpu_memory_mb, parse_cuda_device, peak_memory_mb,
                                  get_frozen_and_tunable_parameter_names, lazy_groups_of)
from allennlp.common.tqdm import Tqdm
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator, TensorDict
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import util as nn_util
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.momentum_schedulers import MomentumScheduler
from allennlp.training.metric_tracker import MetricTracker
from allennlp.training.optimizers import Optimizer
from allennlp.training.tensorboard_writer import TensorboardWriter
from allennlp.training.trainer_base import TrainerBase
from allennlp.training import util as training_util
from allennlp.training.moving_average import MovingAverage

logger = logging.getLogger(__name__)

class ModifiedCheckpointer(Checkpointer):
    def __init__(self,
                 serialization_dir: str = None,
                 keep_serialized_model_every_num_seconds: int = None,
                 num_serialized_models_to_keep: int = 20,
                 minimal_save: bool = True) -> None:
        super().__init__(serialization_dir=serialization_dir,
                         keep_serialized_model_every_num_seconds=keep_serialized_model_every_num_seconds,
                         num_serialized_models_to_keep=num_serialized_models_to_keep)
        self._minimal_save = minimal_save

    def save_checkpoint(self,
                        epoch: Union[int, str],
                        model_state: Dict[str, Any],
                        training_states: Dict[str, Any],
                        is_best_so_far: bool) -> None:
        if self._serialization_dir is not None:
            model_path = os.path.join(self._serialization_dir, "model_state_epoch_{}.th".format(epoch))
            if not self._minimal_save:
                torch.save(model_state, model_path)
            training_path = os.path.join(self._serialization_dir,
                                         "training_state_epoch_{}.th".format(epoch))
            if not self._minimal_save:
                torch.save({**training_states, "epoch": epoch}, training_path)

            if is_best_so_far:
                logger.info("Best validation performance so far. "
                            "Copying weights to '%s/best.th'.", self._serialization_dir)
                if not self._minimal_save:
                    shutil.copyfile(model_path, os.path.join(self._serialization_dir, "best.th"))
                else:
                    best_model_path = os.path.join(self._serialization_dir, "best.th")
                    torch.save(model_state, best_model_path)

            if self._num_serialized_models_to_keep and self._num_serialized_models_to_keep >= 0:
                self._serialized_paths.append((time.time(), model_path, training_path))
                if len(self._serialized_paths) > self._num_serialized_models_to_keep:
                    paths_to_remove = self._serialized_paths.pop(0)
                    # Check to see if we should keep this checkpoint, if it has been longer
                    # then self._keep_serialized_model_every_num_seconds since the last
                    # kept checkpoint.
                    remove_path = True
                    if self._keep_serialized_model_every_num_seconds is not None:
                        save_time = paths_to_remove[0]
                        time_since_checkpoint_kept = save_time - self._last_permanent_saved_checkpoint_time
                        if time_since_checkpoint_kept > self._keep_serialized_model_every_num_seconds:
                            # We want to keep this checkpoint.
                            remove_path = False
                            self._last_permanent_saved_checkpoint_time = save_time
                    if remove_path:
                        for fname in paths_to_remove[1:]:
                            if os.path.isfile(fname):
                                os.remove(fname)

@TrainerBase.register("modified_trainer")
class ModifiedTrainer(Trainer):
    def __init__(self,
                 model,
                 optimizer: torch.optim.Optimizer,
                 iterator,
                 train_dataset,
                 validation_dataset = None,
                 patience: Optional[int] = None,
                 validation_metric: str = "-loss",
                 validation_iterator = None,
                 shuffle: bool = True,
                 num_epochs: int = 20,
                 serialization_dir: Optional[str] = None,
                 num_serialized_models_to_keep: int = 20,
                 keep_serialized_model_every_num_seconds: int = None,
                 checkpointer: Checkpointer = None,
                 model_save_interval: float = None,
                 cuda_device: Union[int, List] = -1,
                 grad_norm: Optional[float] = None,
                 grad_clipping: Optional[float] = None,
                 learning_rate_scheduler = None,
                 momentum_scheduler = None,
                 summary_interval: int = 100,
                 histogram_interval: int = None,
                 should_log_parameter_statistics: bool = True,
                 should_log_learning_rate: bool = False,
                 log_batch_size_period: Optional[int] = None,
                 moving_average = None,
                 minimal_save = False) -> None:
        super().__init__(model=model,
                         optimizer=optimizer,
                         iterator=iterator,
                         train_dataset=train_dataset,
                         validation_dataset=validation_dataset,
                         patience=patience,
                         validation_metric=validation_metric,
                         validation_iterator=validation_iterator,
                         shuffle=shuffle,
                         num_epochs=num_epochs,
                         serialization_dir=serialization_dir,
                         num_serialized_models_to_keep=num_serialized_models_to_keep,
                         keep_serialized_model_every_num_seconds=keep_serialized_model_every_num_seconds,
                         checkpointer=checkpointer,
                         model_save_interval=model_save_interval,
                         cuda_device=cuda_device,
                         grad_norm=grad_norm,
                         grad_clipping=grad_clipping,
                         learning_rate_scheduler=learning_rate_scheduler,
                         momentum_scheduler=momentum_scheduler,
                         summary_interval=summary_interval,
                         histogram_interval=histogram_interval,
                         should_log_parameter_statistics=should_log_parameter_statistics,
                         should_log_learning_rate=should_log_learning_rate,
                         log_batch_size_period=log_batch_size_period,
                         moving_average=moving_average)
        self._checkpointer = ModifiedCheckpointer(serialization_dir,
                                                  keep_serialized_model_every_num_seconds,
                                                  num_serialized_models_to_keep,
                                                  minimal_save)

    @classmethod
    def from_params(cls, params, serialization_dir, recover):
        pieces = TrainerPieces.from_params(params, serialization_dir, recover)
        return cls.from_params_old(model=pieces.model,
                                    serialization_dir=serialization_dir,
                                    iterator=pieces.iterator,
                                    train_data=pieces.train_dataset,
                                    validation_data=pieces.validation_dataset,
                                    params=pieces.params,
                                    validation_iterator=pieces.validation_iterator)

    @classmethod
    def from_params_old(cls,  # type: ignore
                        model,
                        serialization_dir: str,
                        iterator,
                        train_data,
                        validation_data,
                        params,
                        validation_iterator = None) -> 'Trainer':
        # pylint: disable=arguments-differ
        patience = params.pop_int("patience", None)
        validation_metric = params.pop("validation_metric", "-loss")
        shuffle = params.pop_bool("shuffle", True)
        num_epochs = params.pop_int("num_epochs", 20)
        cuda_device = parse_cuda_device(params.pop("cuda_device", -1))
        grad_norm = params.pop_float("grad_norm", None)
        grad_clipping = params.pop_float("grad_clipping", None)
        lr_scheduler_params = params.pop("learning_rate_scheduler", None)
        momentum_scheduler_params = params.pop("momentum_scheduler", None)

        if isinstance(cuda_device, list):
            model_device = cuda_device[0]
        else:
            model_device = cuda_device
        if model_device >= 0:
            # Moving model to GPU here so that the optimizer state gets constructed on
            # the right device.
            model = model.cuda(model_device)

        parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
        optimizer = Optimizer.from_params(parameters, params.pop("optimizer"))
        if "moving_average" in params:
            moving_average = MovingAverage.from_params(params.pop("moving_average"), parameters=parameters)
        else:
            moving_average = None

        if lr_scheduler_params:
            lr_scheduler = LearningRateScheduler.from_params(optimizer, lr_scheduler_params)
        else:
            lr_scheduler = None
        if momentum_scheduler_params:
            momentum_scheduler = MomentumScheduler.from_params(optimizer, momentum_scheduler_params)
        else:
            momentum_scheduler = None

        if 'checkpointer' in params:
            if 'keep_serialized_model_every_num_seconds' in params or \
                    'num_serialized_models_to_keep' in params:
                raise ConfigurationError(
                        "Checkpointer may be initialized either from the 'checkpointer' key or from the "
                        "keys 'num_serialized_models_to_keep' and 'keep_serialized_model_every_num_seconds'"
                        " but the passed config uses both methods.")
            checkpointer = Checkpointer.from_params(params.pop("checkpointer"))
        else:
            num_serialized_models_to_keep = params.pop_int("num_serialized_models_to_keep", 20)
            keep_serialized_model_every_num_seconds = params.pop_int(
                    "keep_serialized_model_every_num_seconds", None)
            checkpointer = Checkpointer(
                    serialization_dir=serialization_dir,
                    num_serialized_models_to_keep=num_serialized_models_to_keep,
                    keep_serialized_model_every_num_seconds=keep_serialized_model_every_num_seconds)
        model_save_interval = params.pop_float("model_save_interval", None)
        summary_interval = params.pop_int("summary_interval", 100)
        histogram_interval = params.pop_int("histogram_interval", None)
        should_log_parameter_statistics = params.pop_bool("should_log_parameter_statistics", True)
        should_log_learning_rate = params.pop_bool("should_log_learning_rate", False)
        log_batch_size_period = params.pop_int("log_batch_size_period", None)
        minimal_save = params.pop_int("minimal_save", False)

        params.assert_empty(cls.__name__)
        return cls(model, optimizer, iterator,
                   train_data, validation_data,
                   patience=patience,
                   validation_metric=validation_metric,
                   validation_iterator=validation_iterator,
                   shuffle=shuffle,
                   num_epochs=num_epochs,
                   serialization_dir=serialization_dir,
                   cuda_device=cuda_device,
                   grad_norm=grad_norm,
                   grad_clipping=grad_clipping,
                   learning_rate_scheduler=lr_scheduler,
                   momentum_scheduler=momentum_scheduler,
                   checkpointer=checkpointer,
                   model_save_interval=model_save_interval,
                   summary_interval=summary_interval,
                   histogram_interval=histogram_interval,
                   should_log_parameter_statistics=should_log_parameter_statistics,
                   should_log_learning_rate=should_log_learning_rate,
                   log_batch_size_period=log_batch_size_period,
                   moving_average=moving_average,
                   minimal_save=minimal_save)