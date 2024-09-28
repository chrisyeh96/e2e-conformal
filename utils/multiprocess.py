from collections.abc import Callable, Hashable, Mapping, Sequence
from concurrent import futures
from typing import TypeVar

from tqdm.auto import tqdm

T = TypeVar('T')


def run_parallel(
    func: Callable[..., tuple[T, str]],
    kwargs_list: Sequence[Mapping[str, Hashable]],
    workers: int
) -> list[None | T]:
    """Runs a function in parallel with multiple seeds.

    Args:
        func: function to run in parallel with multiple kwargs
        kwargs_list: list of kwargs dicts to pass into func
        workers: maximum number of processes to use

    Returns:
        results: list of results from running func with each seed
    """
    results: list[T | None] = []
    if workers == 1:
        for kwargs in tqdm(kwargs_list):
            name = "(" + ", ".join(f"{k}={v}" for k, v in kwargs.items()) + ")"
            result, msg = func(**kwargs)
            if msg != '':
                tqdm.write(f'Item {name}: {msg}')
            results.append(result)

    else:
        results_by_name: dict[str, None | T] = {}
        with futures.ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_run = {}
            names_list = []
            for kwargs in kwargs_list:
                name = "(" + ", ".join(f"{k}={v}" for k, v in kwargs.items()) + ")"
                future = pool.submit(func, **kwargs)
                future_to_run[future] = name
                names_list.append(name)

            pbar = tqdm(futures.as_completed(future_to_run), total=len(kwargs_list))
            for future in pbar:
                name = future_to_run[future]
                try:
                    result, msg = future.result()
                    if msg != '':
                        tqdm.write(f'Item {name}: {msg}')
                    results_by_name[name] = result
                    pbar.update()
                except Exception as e:
                    print(f'Item {name} generated an exception: {e}')
                    results_by_name[name] = None
        results = [results_by_name[name] for name in names_list]
    return results
