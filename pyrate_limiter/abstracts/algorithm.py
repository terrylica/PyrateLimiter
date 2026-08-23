"""Rate-limiting algorithm abstraction.

Separates the policy (which rates admit an item, how long a rejected one waits,
how far back items may be leaked) from the storage that counts and persists
them. Internal in v4; v5 promotes it to a public extension point.

Retry-after rides on ``Decision`` so one check under one lock yields both the
verdict and the wait. Deriving it afterwards costs a second round trip and
reads state that may have moved - and is impossible for algorithms whose state
is not a log (token bucket, GCRA), which compute the wait in closed form.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Final, List, Optional, Sequence

from .rate import Rate


@dataclass(frozen=True)
class Decision:
    """Outcome of an admit check.

    ``retry_after_ms`` is measured from the checked item's own timestamp.
    ``None`` means "unknown, ask ``AbstractBucket.waiting()``" - either the
    weight can never fit, or the backend does not compute a wait. It does not
    mean "no wait".
    """

    failing_rate: Optional[Rate] = None
    retry_after_ms: Optional[int] = None

    @property
    def allowed(self) -> bool:
        return self.failing_rate is None


#: Reused on every admit; ``Decision`` is immutable, so the hot path allocates nothing.
ADMITTED: Final["Decision"] = Decision()


class Algorithm(ABC):
    """A rate-limiting policy, independent of any storage backend.

    Implementations must be stateless so one instance can be shared across
    buckets and threads.
    """

    @abstractmethod
    def admit(self, rates: List[Rate], counts: Sequence[int], weight: int) -> Decision:
        """Whether ``weight`` more units fit, given ``counts`` aligned to ``rates``."""


class LogAlgorithm(Algorithm):
    """Policy over storage holding one timestamped entry per consumed unit.

    Constant-state policies (token bucket, GCRA) will not implement this.
    """

    @abstractmethod
    def leak_bound(self, rates: List[Rate], now: int) -> int:
        """Timestamp below which an item is outside every rate's window."""

    @abstractmethod
    def blocking_offset(self, rate: Rate, weight: int) -> int:
        """Offset from the newest stored item (0-based) of the one whose expiry
        makes room for ``weight``."""

    @abstractmethod
    def retry_after(self, rate: Rate, blocking_timestamp: int, now: int) -> int:
        """Milliseconds until ``blocking_timestamp`` leaves ``rate``'s window."""

    def decide(
        self,
        rates: List[Rate],
        counts: Sequence[int],
        weight: int,
        now: int,
        peek_timestamp: Callable[[int], Optional[int]],
    ) -> Decision:
        """``admit()``, resolving the retry-after in the same step on denial.

        ``peek_timestamp(offset)`` is only called on the deny path, so backends
        pay for the lookup only when it is needed.
        """
        decision = self.admit(rates, counts, weight)

        if decision.allowed:
            return decision

        rate = decision.failing_rate
        assert rate is not None

        if weight > rate.limit:
            # Can never fit; waiting() reports -1 and the limiter gives up.
            return decision

        blocking_timestamp = peek_timestamp(self.blocking_offset(rate, weight))

        if blocking_timestamp is None:
            return Decision(failing_rate=rate, retry_after_ms=0)

        return Decision(
            failing_rate=rate,
            retry_after_ms=self.retry_after(rate, blocking_timestamp, now),
        )


class SlidingWindowLog(LogAlgorithm):
    """Precise sliding-window-log policy: admit while every rate's rolling
    window stays under its limit.

    Backends may implement it natively for atomicity (Redis Lua, Postgres lock
    + ``COUNT FILTER``, in-memory bisect) but share these definitions.
    """

    def admit(self, rates: List[Rate], counts: Sequence[int], weight: int) -> Decision:
        for rate, count in zip(rates, counts, strict=True):
            if rate.limit - int(count) < weight:
                return Decision(failing_rate=rate)
        return ADMITTED

    def leak_bound(self, rates: List[Rate], now: int) -> int:
        # rates sort ascending by interval, so rates[-1] is the widest window.
        return now - rates[-1].interval

    def blocking_offset(self, rate: Rate, weight: int) -> int:
        # Counting from the newest item keeps this independent of both the
        # in-window count and any expired-but-unleaked entries still stored.
        return rate.limit - weight

    def retry_after(self, rate: Rate, blocking_timestamp: int, now: int) -> int:
        # +1 clears the inclusive lower bound: landing exactly on it leaves the
        # item still counted, so the re-put fails and the limiter spins at 0.
        return blocking_timestamp + rate.interval - now + 1
