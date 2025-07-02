import torch


class _Criterion:

    def __init__(self):
        self.loss_dict: dict[str, callable] = {}

    def register_loss(self, name: str, method: callable):
        """
        Register a loss function with a name and a method to compute it.
        """
        if name in self.loss_dict:
            raise ValueError(f"Loss '{name}' is already registered.")
        self.loss_dict[name] = method

    def compute_loss(self, name, model_output, *args, **kwargs) -> torch.Tensor:
        """
        Compute the loss based on the model output and additional arguments.
        This method should be overridden by subclasses.
        """
        loss = self.loss_dict[name](model_output, *args, **kwargs)
        return loss.mean() if loss.numel() > 0 else loss


class STRCriterion(_Criterion):
    """
    A criterion for scene text recognition tasks.
    This class is a placeholder for the actual implementation.
    """

    def __init__(self):
        super().__init__()
        self.register_loss("ctc_loss", self._ctc_loss)

    def _ctc_loss(self, model_output, batch):
        """
        Compute the CTC loss based on the model output and ground truth tokens.
        """
        images, labels, targets, target_lengths = batch
        input_lengths = torch.full(
            (images.size(0),),
            model_output.size(0),
            dtype=torch.long,
            device=model_output.device,
        )
        ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)
        log_probs = model_output.log_softmax(dim=-1)
        return ctc_loss(log_probs, targets, input_lengths, target_lengths)
