from math import ceil
from typing import Iterator, Iterable, Generator, List


def batchify(generator: Iterable, batch_size: int, lazy: bool = False) -> Generator[List, None, None]:
    """
    Lazyly evaluated batch-wise loading
    """

    if batch_size == 1:
        for item in generator:
            yield item
        return

    if lazy:
        # Lazy returns batches as a generator where objects are only touched upon actually querying them
        iterator = iter(generator)
        try:
            while True:
                first = next(iterator)

                def chunk():
                    try:
                        yield first
                        for _ in range(batch_size - 1):
                            yield next(iterator)
                    except StopIteration:
                        pass

                yield chunk()
        except StopIteration:
            pass
    else:
        # Regular mode materializes all objects within a batch before the batch is returned as a list
        batch = []
        for i, item in enumerate(generator):
            batch.append(item)
            if (i + 1) % batch_size == 0:
                yield batch
                batch = []
        if batch:
            yield batch


def batchify_sliced(tensor, batch_size: int) -> Iterator:
    try:
        n_samples = len(tensor)
    except Exception:
        try:
            n_samples = tensor.shape[0]
        except Exception:
            raise ValueError(f"Cannot infer length of passed tensor with type {type(tensor)}. "
                             f"Ensure to use a common Tensor/Array format")

    n_batches = ceil(n_samples / batch_size)
    for i_batch in range(n_batches):
        if i_batch == n_batches - 1:
            yield tensor[i_batch * batch_size:]  # Return all remaining samples as the last batch
        else:
            yield tensor[i_batch * batch_size: (i_batch + 1) * batch_size]