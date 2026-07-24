from abc import ABC, abstractmethod

class BaseSPDA(ABC):
    @abstractmethod
    def forward(self, q, k, v):
        """
        Forward pass for the attention mechanism.

        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, seq_len, num_heads, head_dim).
            k (torch.Tensor): Key tensor of shape (batch_size, seq_len, num_heads, head_dim).
            v (torch.Tensor): Value tensor of shape (batch_size, seq_len, num_heads, head_dim).
        """

