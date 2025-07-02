import torch

from typing import Union
from itertools import groupby
from torch.nn.utils.rnn import pad_sequence


class _Tokenizer:

    def __init__(self, charset: str):
        self.charset = charset


class CTCTokenizer(_Tokenizer):

    _BLANK_ = "[B]"

    def __init__(self, charset: str):
        super().__init__(charset)
        self.ext_charset = (self._BLANK_,) + tuple(charset)

        self.i2c = self.ext_charset
        self.c2i = {c: i for i, c in enumerate(self.i2c)}
        self.blank = self.c2i[self._BLANK_]

    def __len__(self):
        return len(self.i2c)

    def encode_label(self, label: str) -> torch.Tensor:
        return torch.tensor([self.c2i[c] for c in label], dtype=torch.long)

    def encode_batch(
        self,
        labels: list[str],
        device,
        requires_lengths: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor]]:
        batch: list[torch.Tensor] = [self.encode_label(l) for l in labels]
        if requires_lengths:
            lengths = torch.tensor([t.shape[0] for t in batch], dtype=torch.long, device=device)
            batch = pad_sequence(batch, batch_first=True, padding_value=self.blank).to(device)
            return batch, lengths
        batch = pad_sequence(batch, batch_first=True, padding_value=self.blank).to(device)
        return batch

    def decode_token(self, token: Union[list[int], torch.Tensor]) -> str:
        if isinstance(token, torch.Tensor):
            token = token.tolist()
        token = list(zip(*groupby(token)))[0]
        return "".join([self.i2c[i] for i in token if i != self.blank])

    def decode_batch(self, tokens: Union[list[list[int]], torch.Tensor]) -> list[str]:
        return [self.decode_token(t) for t in tokens]
