import torch


class _Validator:

    def __init__(self):
        self.metric_dict: dict[str, callable] = {}

    def register_metric(self, name: str, method: callable):
        """
        Register a metric with a name and a method to compute it.
        """
        if name in self.metric_dict:
            raise ValueError(f"Metric '{name}' is already registered.")
        self.metric_dict[name] = method

    def compute_metric(self, name: str, *args, **kwargs):
        """
        Compute the metric by name using the registered method.
        """
        if name not in self.metric_dict:
            raise ValueError(f"Metric '{name}' is not registered.")
        return self.metric_dict[name](*args, **kwargs)


class STRValidator(_Validator):
    """
    Scene Text Recognition Validator.
    This class is used to validate the model during training and inference.
    It provides methods to compute metrics and handle model outputs.
    """

    def __init__(self):
        super().__init__()
        self.metric_name = "accuracy"  # Default metric name
        self.register_metric(self.metric_name, self._compute_accuracy)

    def _compute_accuracy(self, predictions: torch.Tensor, batch: tuple):
        """
        Compute accuracy based on predictions and targets.
        """
        images, ground_labels, ground_tokens, ground_lengths = batch
        correct = (predictions == ground_tokens).sum().item()
        total = ground_labels.size(0)
        return correct / total if total > 0 else 0.0
