"""Rate-limiting algorithm abstraction.

Separates the *policy* (which rates admit an item, how long a rejected item must
wait, and how far back items can be leaked) from the *storage* that counts and
persists those items. In v4 this is an internal seam: the built-in buckets
delegate their per-rate admit decision, retry-after and leak bound here, so the
sliding-window-log logic lives in exactly one place instead of being
copy-pasted across backends.

``Decision`` carries the retry-after alongside the verdict, so a single check
under a single lock yields both. This matters beyond tidiness: deriving the wait
afterwards (the pre-4.5 ``AbstractBucket.waiting()`` path) costs a second
round trip to the backend and reads state that may have changed since the put.
It is also the prerequisite for algorithms whose state is *not* a log - a token
bucket or GCRA has no k-th-newest item to look up, and computes its wait in
closed form.

``LogAlgorithm`` is the sub-interface for policies that do keep a log of
timestamped items (everything shipped in v4). v5 adds a sibling for
constant-state policies plus a ``Store`` interface they can share, at which
point ``Algorithm`` becomes a public extension point.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Final, List, Optional, Sequence

from .rate import Rate


@dataclass(frozen=True)
class Decision:
    """Outcome of an admit check.

    ``failing_rate is None`` means the item was admitted; otherwise it is the
    first rate whose window was full.

    ``retry_after_ms`` is how long after the checked item's own timestamp the
    same weight would fit, or ``None`` when the algorithm could not determine it
    - either because the weight can never fit (it exceeds ``failing_rate.limit``)
    or because the backend does not compute it. ``None`` is not "no wait": it
    means "ask ``AbstractBucket.waiting()``", which falls back to deriving the
    wait from storage.
    """

    failing_rate: Optional[Rate] = None
    retry_after_ms: Optional[int] = None

    @property
    def allowed(self) -> bool:
        return self.failing_rate is None


#: The verdict for an admitted item. ``Decision`` is immutable, so the same
#: instance is reused on every successful put rather than allocating a fresh one
#: on the hot path.
ADMITTED: Final["Decision"] = Decision()


class Algorithm(ABC):
    """A rate-limiting policy, expressed independently of any storage backend.

    A bucket supplies the per-rate windowed counts (however it obtains them -
    in-memory bisect, ``ZCOUNT``, ``COUNT(*) FILTER``...) and the algorithm
    decides admission. Implementations must be stateless so a single instance
    can be shared across buckets and threads.
    """

    @abstractmethod
    def admit(self, rates: List[Rate], counts: Sequence[int], weight: int) -> Decision:
        """Decide whether ``weight`` more items fit, given ``counts[i]`` items
        already inside ``rates[i]``'s window (``counts`` aligned to ``rates``).

        The returned ``Decision`` carries no retry-after; use ``decide()`` to
        get both in one step.
        """


class LogAlgorithm(Algorithm):
    """Policy over storage that keeps one timestamped entry per consumed unit.

    Such a policy can answer "when does this become admissible?" by naming the
    stored entry that has to fall out of the window first - so the retry-after
    is a lookup plus arithmetic rather than a closed form. Constant-state
    policies (token bucket, GCRA) will *not* implement this interface.
    """

    @abstractmethod
    def leak_bound(self, rates: List[Rate], now: int) -> int:
        """Timestamp below which an item is outside *every* rate's window and so
        may be leaked."""

    @abstractmethod
    def blocking_offset(self, rate: Rate, weight: int) -> int:
        """Offset, counting from the *newest* stored item (0-based), of the item
        whose expiry frees enough room for ``weight`` more units under ``rate``.
        """

    @abstractmethod
    def retry_after(self, rate: Rate, blocking_timestamp: int, now: int) -> int:
        """Milliseconds from ``now`` until the item at ``blocking_timestamp``
        has left ``rate``'s window."""

    def decide(
        self,
        rates: List[Rate],
        counts: Sequence[int],
        weight: int,
        now: int,
        peek_timestamp: Callable[[int], Optional[int]],
    ) -> Decision:
        """``admit()``, resolving the retry-after in the same step on denial.

        ``peek_timestamp(offset)`` returns the timestamp of the stored item at
        ``offset`` counting from the newest, or ``None`` if there is no such
        item. It is only ever called when the item was rejected *and* the weight
        can fit in principle, so backends pay for the lookup on the slow path
        only.
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
            # Nothing left to expire - the bucket is already ready.
            return Decision(failing_rate=rate, retry_after_ms=0)

        return Decision(
            failing_rate=rate,
            retry_after_ms=self.retry_after(rate, blocking_timestamp, now),
        )


class SlidingWindowLog(LogAlgorithm):
    """Precise sliding-window-log policy.

    Admit while, for every rate, the count of items within its rolling interval
    stays under the limit. This is PyrateLimiter's default and (until v5) only
    algorithm; backends may implement it natively for atomicity/performance
    (Redis Lua, Postgres lock + ``COUNT FILTER``, in-memory bisect) but share
    this definition of the decision.
    """

    def admit(self, rates: List[Rate], counts: Sequence[int], weight: int) -> Decision:
        for rate, count in zip(rates, counts, strict=True):
            if rate.limit - int(count) < weight:
                return Decision(failing_rate=rate)
        return ADMITTED

    def leak_bound(self, rates: List[Rate], now: int) -> int:
        # rates are sorted ascending by interval (AbstractBucket.rates), so
        # rates[-1] is the widest window.
        return now - rates[-1].interval

    def blocking_offset(self, rate: Rate, weight: int) -> int:
        # Freeing room for `weight` means evicting the oldest
        # `weight - (limit - count)` in-window items. Counting from the newest
        # item instead makes the offset independent of both the count and of
        # how many expired-but-unleaked items storage still holds.
        return rate.limit - weight

    def retry_after(self, rate: Rate, blocking_timestamp: int, now: int) -> int:
        # +1: the window lower bound is inclusive across all backends (an item
        # counts while timestamp >= now - interval). Returning the bare
        # difference lands the retry exactly ON the boundary, where the item is
        # still counted, so the re-put fails and the limiter busy-spins at
        # delay=0. One extra ms pushes strictly past it.
        return blocking_timestamp + rate.interval - now + 1
