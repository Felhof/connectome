import math
import warnings
from collections import defaultdict
from typing import Union, Optional, Iterable

from transformer_lens import HookedTransformer

# Exploration strategies

EndPoint = Union[int, slice]


class Strategy:
    """
    A strategy defines the which interventions to do, based on the results of previous interventions.
    It populates the `to_explore` set with pairs of endpoints to explore, and the `report` method is called
    when the results of an intervention are available.
    """

    def __init__(self):
        self.to_explore: list[tuple[EndPoint, EndPoint]] = []

    def start(self, n_tokens: int):
        """Called at the start of the exploration, with the number of tokens in the input.
        Override this method to initialize the exploration."""
        raise NotImplementedError

    def pop_explorations(self,
                         max_explorations: Optional[int] = None) -> list[tuple[EndPoint, EndPoint]]:
        """Get the next explorations to do.

        If `max_explorations` is not specified, all explorations are returned, otherwise at most `max_explorations` are returned.
        """
        if max_explorations is None:
            max_explorations = len(self.to_explore)

        new = self.to_explore[:max_explorations]
        del self.to_explore[:max_explorations]
        return new

    def explore_next(self, source: EndPoint, target: EndPoint):
        """Add a new exploration to the list of explorations to do.

        If all sources are after all targets, nothing is added."""
        source_start = source.start if isinstance(source, slice) else source
        target_end = target.stop if isinstance(target, slice) else target
        if source_start < target_end:
            self.to_explore.append((source, target))

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.to_explore) == 0:
            raise StopIteration
        return self.to_explore.pop()

    def report(self, source: Union[int, slice], target: Union[int, slice], strength: float):
        """Called when the results of an intervention are available.
        Override to generate new things to explore based on the results of the intervention.
        """
        pass

    def __str__(self):
        return f"{self.__class__.__name__}"


class BasicStrategy(Strategy):

    def start(self, n_tokens: int):
        self.to_explore = [(source, target) for target in range(1, n_tokens)
                           for source in range(1, target + 1)]

    def __len__(self):
        return len(self.to_explore)


class BisectStrategy(Strategy):

    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold

    def start(self, n_tokens: int):
        all_tokens = slice(0, n_tokens)
        self.to_explore = [(all_tokens, all_tokens)]

    def report(self, source: EndPoint, target: EndPoint, strength: float):
        assert type(source) == type(target) == slice

        # If the connexion is too weak, we can skip the whole square
        if abs(strength) < self.threshold:
            return

        if source.stop - source.start == 1:
            source_parts = [source]
        else:
            source_mid = math.ceil((source.start + source.stop) / 2)
            source_parts = [
                slice(source.start, source_mid),
                slice(source_mid, source.stop),
            ]

        if target.stop - target.start == 1:
            target_parts = [target]
        else:
            target_mid = math.ceil((target.start + target.stop) / 2)
            target_parts = [
                slice(target.start, target_mid),
                slice(target_mid, target.stop),
            ]

        # If we can't bisect anymore, because it's a single token
        if len(source_parts) == 1 and len(target_parts) == 1:
            return

        for source_part in source_parts:
            for target_part in target_parts:
                self.explore_next(source_part, target_part)


class BacktrackingStrategy(Strategy):
    """
    A strategy that backtracks from the last token to find all connections that are on a path
    to the last token with edges all above a given threshold.
    """

    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold
        self.seen = set()

    def start(self, n_tokens: int):
        self.seen = set()
        self.to_explore = []
        self.backtrack_from(n_tokens - 1)

    def backtrack_from(self, target: int):
        """Explore all the sources to the given target, if not already explored."""
        if target not in self.seen:
            self.seen.add(target)
            for source in range(1, target + 1):
                self.explore_next(source, target)

    def report(self, source: EndPoint, target: EndPoint, strength: float):
        assert type(source) == type(target) == int
        # If the connexion is too weak, don't explore it
        if abs(strength) >= self.threshold:
            self.backtrack_from(source)


class BacktrackBisectStrategy(Strategy):
    """A strategy that explores targets only when they are directly connected to the last token
    and bisects the sources to find possible earlier nodes."""

    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold
        self.seen = set()

    def start(self, n_tokens: int):
        self.seen.clear()
        self.to_explore = [(slice(1, n_tokens), n_tokens - 1)]

    def report(self, source: EndPoint, target: EndPoint, strength: float):
        assert isinstance(target, int)
        assert isinstance(source, slice)

        if abs(strength) < self.threshold:
            return
        if source.stop == source.start + 1:
            # source.start is individually connected to target
            # -> We explore the possible parents of source.start
            if source.start not in self.seen:
                self.seen.add(source.start)
                self.explore_next(slice(1, source.start + 1), source.start)
        else:
            mid = math.ceil((source.start + source.stop) / 2)
            self.explore_next(slice(source.start, mid), target)
            self.explore_next(slice(mid, source.stop), target)


class SplitStrategy(Strategy):
    """A strategy that groups tokens into clusters and explores the connections between clusters
    before exploring the connections inside clusters."""

    DEFAULT_SPLITS = (
        "\n\n",
        "\n",
        tuple(".!?"),
        tuple(",:;"),
    )

    def __init__(
            self,
            model: HookedTransformer,
            prompt: str,
            threshold: float,
            delimiters: Iterable[Union[str, tuple[str, ...]]] = DEFAULT_SPLITS,
            tokens_as_leaves=True,
            delimiters_as_leaves=False,
    ):
        super().__init__()
        self.model = model
        self.prompt = prompt
        self.threshold = threshold
        self.tokens_as_leaves = tokens_as_leaves
        self.delimiters_as_leaves = delimiters_as_leaves
        # Make all delimiters tuples
        self.delimiters: list[tuple[str, ...]] = [
            (delimiter,) if isinstance(delimiter, str) else delimiter for delimiter in delimiters
        ]
        # every delimiter should be a token
        for delimiters in self.delimiters:
            for delimiter in delimiters:
                tokens = self.model.to_str_tokens(delimiter, prepend_bos=False)
                assert (
                        len(tokens) == 1), f"Delimiter {delimiter} is not a single token but {tokens}."

        self.tree: dict[tuple[int, int], list[EndPoint]] = self.build_tree(model, prompt)

    def build_tree(self, model: HookedTransformer,
                   prompt: str) -> dict[tuple[int, int], list[EndPoint]]:
        tokens = model.to_str_tokens(prompt)

        def new_child(parent: slice, child_start: int, child_end: int):
            """Create a new child and add it to the tree. Both start and end are included."""
            # We include the delimiter in the previous slice
            child = slice(child_start, child_end + 1)
            # Always add the child to the current layer, to continue splitting it later
            current_layer.append(child)
            # Avoid loops
            if child != parent:
                tree[parent.start, parent.stop].append(child)

        # Only for warnings
        delimiter_not_used = {d for ds in self.delimiters for d in ds}

        def needs_splitting(token: str, delimiters: tuple[str, ...]) -> bool:
            # Check if the delimiter is (part of) the token (e.g. \n\n is part of \n\n\n)
            for delim in delimiters:
                if delim in token:
                    delimiter_not_used.discard(delim)
                    return True
            return False

        tree: dict[tuple[int, int], list[EndPoint]] = defaultdict(list)
        last_layer = [slice(1, len(tokens))]
        # Depth by depth in the tree
        for delimiters in self.delimiters:
            current_layer = []
            for parent in last_layer:
                child_start = parent.start
                for child_end, token in enumerate(tokens[parent], start=parent.start):
                    if needs_splitting(token, delimiters):
                        # Put the delimiter on its own group if there is more than one token in the group
                        if self.delimiters_as_leaves and child_start < child_end:
                            new_child(parent, child_start, child_end - 1)  # Before the delimiter
                            new_child(parent, child_end, child_end)  # The delimiter
                        else:
                            new_child(parent, child_start, child_end)
                        child_start = child_end + 1
                # We include the last slice, if it's not empty
                if child_start < parent.stop:
                    new_child(parent, child_start, parent.stop - 1)

            # Keep the parent slices since none have children of the kind
            if current_layer:
                last_layer = current_layer

        if self.tokens_as_leaves:
            for parent in last_layer:
                # Slices of length one are already in the tree
                if parent.stop != parent.start + 1:
                    tree[parent.start,
                    parent.stop] = [slice(t, t + 1) for t in range(parent.start, parent.stop)]

        # Warn if some delimiters were not used
        if delimiter_not_used:
            warnings.warn(f"The following delimiters were not used: {delimiter_not_used}.")

        return tree

    def start(self, n_tokens: int):
        assert n_tokens == len(self.model.to_str_tokens(self.prompt))
        all_tokens = slice(1, n_tokens)
        self.to_explore = [(all_tokens, all_tokens)]

    def report(self, source: Union[int, slice], target: Union[int, slice], strength: float):
        if abs(strength) < self.threshold:
            return

        try:
            starts = self.tree[source.start, source.stop] or [source]
        except AttributeError:  # source is an int
            starts = [source]

        try:
            ends = self.tree[target.start, target.stop] or [target]
        except AttributeError:  # target is an int
            ends = [target]

        if len(ends) == 1 and len(starts) == 1:
            # We can't split the source or target further
            return

        for start in starts:
            for end in ends:
                self.explore_next(start, end)

    def show_tree(self):
        tokens = self.model.to_str_tokens(self.prompt)
        positions = list(range(len(tokens)))
        for source, targets in sorted(self.tree.items()):
            print(f"{source}: {tokens[source[0]:source[1]]}")
            child_positions = []
            for target in targets:
                print(f"  -> {target}: {tokens[target]}")
                if isinstance(target, slice):
                    child_positions.extend(positions[target])
                else:
                    child_positions.append(positions[target])
            assert sorted(child_positions) == positions[source[0]:source[1]], (
                f"Child positions {child_positions} do not match parent positions "
                f"{positions[source[0]:source[1]]}")
