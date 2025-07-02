import torch

from abc import ABC, abstractmethod


class ModelAdapter(ABC):

    @abstractmethod
    def train_input(self, batch, local_rank):
        """
        Used for training phase.
        Parse the batch and put the batched data into device and prepare it for the model
        If is not overridden, it will return the original batch
        """
        return batch

    @abstractmethod
    def train_output(self, model_output):
        """
        Used for training phase.
        Parse the model output and return the output in a format that is suitable for the loss function
        If is not overridden, it will return the original model output
        """
        return model_output

    @torch.inference_mode()
    @abstractmethod
    def val_input(self, batch, local_rank):
        """
        Used for validation phase.
        Parse the batch and put the batched data into device and prepare it for the model
        If is not overridden, it will return the original batch
        """
        return batch

    @torch.inference_mode()
    @abstractmethod
    def val_output(self, model_output):
        """
        Used for validation phase.
        Parse the model output and return the output in a format that is suitable for the loss function
        If is not overridden, it will return the original model output
        """
        return model_output

    @abstractmethod
    def model_train_forward(self, model, batch, local_rank):
        """
        Forward the batch through the model
        If is not overridden, it will return the original model output
        """
        model_input = self.train_input(batch, local_rank)
        model_output = model(model_input)
        return self.train_output(model_output)

    @torch.inference_mode()
    @abstractmethod
    def model_inference_forward(self, model, batch, local_rank):
        """
        Forward the batch through the model in inference mode
        If is not overridden, it will return the original model output
        """
        model_input = self.val_input(batch, local_rank)
        model_output = model(model_input)
        return self.val_output(model_output)
