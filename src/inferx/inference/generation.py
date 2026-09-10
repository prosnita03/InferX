"""
Generation result structures and metrics representation.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class GenerationResult:
    """Encapsulates output text, token sequences, and fine-grained timing metrics."""
    prompt: str
    generated_text: str
    prompt_tokens: int
    generated_tokens: int
    total_tokens: int
    total_latency_ms: float
    time_to_first_token_ms: float
    decode_latency_ms: float
    inter_token_latencies_ms: List[float] = field(default_factory=list)
    tokens_per_second: float = 0.0

    def __post_init__(self):
        if self.total_latency_ms > 0 and self.tokens_per_second == 0.0:
            self.tokens_per_second = (self.generated_tokens / (self.total_latency_ms / 1000.0))

    @property
    def mean_inter_token_latency_ms(self) -> float:
        """Calculate mean latency between consecutive generated tokens."""
        if not self.inter_token_latencies_ms:
            return 0.0
        return sum(self.inter_token_latencies_ms) / len(self.inter_token_latencies_ms)

    def to_dict(self) -> Dict[str, Any]:
        """Convert generation result to dictionary."""
        return asdict(self)
