import argparse
from collections.abc import Sequence


DEFAULT_COMMANDS = ('best_hp', 'e2e', 'eto', 'run_saved')


def parse_args(
    commands: Sequence[str] = DEFAULT_COMMANDS,
    lr: bool = True,
    l2reg: bool = True,
    datasets: Sequence[str] = ()
) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        'command', choices=commands,
        help=f'command to run, one of {commands}')
    p.add_argument(
        '--alpha', type=float, nargs='+',
        default=(0.01, 0.05, 0.1, 0.2),
        help='risk level in (0, 1), the uncertainty set is a (1-alpha)-confidence set')

    if lr:
        p.add_argument(
            '--lr', type=float, nargs='+',
            help='learning rate (unused for command "best_hp")')
    if l2reg:
        p.add_argument(
            '--l2reg', type=float, nargs='+',
            help='L2 regularization strength (unused for command "best_hp")')
    if len(datasets) > 1:
        p.add_argument(
            '--dataset', choices=datasets, required=True,
            help='dataset')

    p.add_argument(
        '--obj', choices=('eto', 'e2e', 'e2e_finetune'),
        help='only used (and required) by "run_saved" command')
    p.add_argument(
        '--shuffle', action='store_true',
        help='shuffle the dataset before splitting into train/calib/test')
    p.add_argument(
        '--multiprocess', type=int, default=1,
        help='number of processes to use for multiprocessing')
    p.add_argument(
        '--tag', default='',
        help='tag to append to the model name')
    p.add_argument(
        '--device', default='cpu',
        help='either "cpu", "cuda", or "cuda:<device_id>"')

    args = p.parse_args()

    for alpha in args.alpha:
        if not (0 < alpha < 1):
            raise ValueError(f'alpha must be in (0, 1), got {alpha}')

    if args.tag != '':
        args.tag = f'_{args.tag}'

    if args.command == 'run_saved':
        if args.obj is None:
            raise ValueError('--obj is required for command "run_saved"')

    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
