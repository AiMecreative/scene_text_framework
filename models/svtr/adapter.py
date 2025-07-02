import torch

from models.model_adapter import ModelAdapter
from utils.tokenizer import CTCTokenizer


class SVTRAdapter(ModelAdapter):

    def __init__(self, train_chaset: str, test_charset: str):
        super().__init__()

        self.train_tokenizer = CTCTokenizer(train_chaset)
        self.test_tokenizer = CTCTokenizer(test_charset)

    def train_input(self, batch, local_rank):
        """
        Parse the batch and put the batched data into device and prepare it for the model.
        """
        images: torch.Tensor
        labels: list[str]
        tokens: torch.Tensor
        lengths: torch.Tensor

        labels, images = batch
        images = images.to(local_rank)
        tokens, lengths = self.train_tokenizer.encode_batch(
            labels,
            device=local_rank,
            requires_lengths=True,
        )
        return images, labels, tokens, lengths

    def val_input(self, batch, local_rank):
        """
        Parse the batch and put the batched data into device and prepare it for the model.
        """
        images: torch.Tensor
        labels: list[str]
        tokens: torch.Tensor
        lengths: torch.Tensor

        labels, images = batch
        images = images.to(local_rank)
        tokens, lengths = self.test_tokenizer.encode_batch(
            labels,
            device=local_rank,
            requires_lengths=True,
        )
        return images, labels, tokens, lengths
    
    def model_train_forward(self, model, batch, local_rank):
        """
        Forward the batch through the model.
        """
        images, labels, tokens, lengths = self.train_input(batch, local_rank)
        model_output = model(images)
        return self.train_output(model_output)

    @torch.inference_mode()
    def model_inference_forward(self, model, batch, local_rank):
        """
        Forward the batch through the model in inference mode.
        """
        images, labels, tokens, lengths = self.val_input(batch, local_rank)
        model_output = model(images)
        return self.val_output(model_output)