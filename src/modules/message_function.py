from torch import nn


class MessageFunction(nn.Module):
    """
    Module which computes the message for a given interaction.
    """

    def compute_message(self, raw_messages):
        return None


class MLPMessageFunction(MessageFunction):
    def __init__(self, raw_message_dimension, message_dimension):
        super(MLPMessageFunction, self).__init__()

        self.mlp = self.layers = nn.Sequential(
            nn.Linear(raw_message_dimension, raw_message_dimension // 2),
            nn.ReLU(),
            nn.Linear(raw_message_dimension // 2, message_dimension),
        )
        self.batch_normal = nn.BatchNorm1d(message_dimension)

    def compute_message(self, raw_messages):
        messages = self.mlp(raw_messages)
        messages = self.batch_normal(messages)
        return messages


class IdentityMessageFunction(MessageFunction):

    def compute_message(self, raw_messages):
        return raw_messages


# a special message function convert raw_message to message for each type interaction
class myMessageFunction(MessageFunction):
    pass


def get_message_function(module_type, raw_message_dimension, message_dimension):
    if module_type == "mlp":
        return MLPMessageFunction(raw_message_dimension, message_dimension)
    elif module_type == "identity":
        assert raw_message_dimension == message_dimension, \
            "raw_message_dimension must equal message_dimension in identity message_function"
        return IdentityMessageFunction()
