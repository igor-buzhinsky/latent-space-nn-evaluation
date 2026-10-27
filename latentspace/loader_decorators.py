"""Decorators of loader-returning functions: filtering by class, limiting or restarting, and endless cycling."""

from collections.abc import Callable, Generator, Iterator

import torch


def class_specific_loader(target_label: int, parent_loader_fn: Callable) -> Callable:
    """
    Filters the supplied loader `parent_loader_fn`, retaining only elements with `target_label`. Preserves batch size.

    :param target_label: The label to retain.
    :param parent_loader_fn: The loader-returning function to decorate.
    :return: Decorated `parent_loader_fn`.
    """

    def loader() -> Iterator[tuple[torch.Tensor, list[int]]]:
        data_generator = iter(parent_loader_fn())
        result = []
        while True:
            try:
                items, labels = next(data_generator)
                batch_size = len(items)
                # accumulate only items of target_label
                result += [item for item, label in zip(items, labels, strict=True) if label == target_label]
                if len(result) >= batch_size:
                    yield torch.stack(result[:batch_size]), [target_label] * batch_size
                    # save the remainder for the next yield
                    result = result[batch_size:]
            except StopIteration:
                return

    return loader


def fixed_length_loader(num_images: int, parent_loader_fn: Callable, restarts: bool = True) -> Callable:
    """
    Restarts or limits the parent loader so that the desired number of images is produced.

    :param num_images: Desired number of images (the effective number will be a multiple of batch size).
    :param parent_loader_fn: The loader-returning function to decorate.
    :param restarts: If `False`, then just limit the parent loader and do not restart it when the end is reached.
    :return: Decorated `parent_loader_fn`.
    """

    def loader() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        generated = 0
        while generated < num_images:
            data_generator = iter(parent_loader_fn())
            while generated < num_images:
                try:
                    items, labels = next(data_generator)
                except StopIteration:
                    if restarts:
                        data_generator = iter(parent_loader_fn())
                        items, labels = next(data_generator)
                    else:
                        return
                yield items, labels
                generated += len(items)

    return loader


def restarting_cycle(iterable_fn: Callable) -> Generator:
    """
    Endlessly yields items from the iterable returned by `iterable_fn`, calling `iterable_fn` again each time the
    current iterable is exhausted. Unlike `itertools.cycle`, this does not store copies of the yielded items (for a data
    loader, this would keep all batches in memory, see https://github.com/pytorch/pytorch/issues/23900), and each pass
    can produce different items (e.g., reshuffled data).

    :param iterable_fn: Function that returns an iterable.
    :return: Infinite generator of items.
    """
    iterator = iter(iterable_fn())
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable_fn())
