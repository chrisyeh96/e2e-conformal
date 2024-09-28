from collections.abc import Collection, Sequence
import itertools

from IPython.display import display
import pandas as pd
import seaborn.objects as so


def get_hps_df(
    fmt_str: str, alphas: Collection[float] | None = None
) -> pd.DataFrame:
    """Reads hyperparameter search results from CSV file(s).

    Returns:
        df: DataFrame with columns (lr, l2reg, seed, loss, is_nan) and possibly 'alpha'
    """
    if alphas is None:
        df = pd.read_csv(fmt_str)
    else:
        dfs = []
        for alpha in alphas:
            df = pd.read_csv(fmt_str.format(alpha=alpha))
            df['alpha'] = alpha
            dfs.append(df)
        df = pd.concat(dfs).reset_index(drop=True)
    df['is_nan'] = df['loss'].isna().astype(int)
    return df


def print_best_hp(df: pd.DataFrame, by: str | list[str] | None = None) -> None:
    """Prints best hyperparameters.

    Prints mean/std of loss for each hyperparameter combination.
    Then prints the best hyperparameters, optionally grouped by alpha or seed.

    Args:
        df: DataFrame with columns (lr, l2reg, seed, loss, is_nan) and possibly 'alpha'
        by: whether to find best hyperparameters per `by`, e.g. 'alpha' or 'seed'
    """
    if isinstance(by, str):
        groupby = [by, 'lr', 'l2reg']
        assert by in df.columns
    elif isinstance(by, list):
        groupby = by + ['lr', 'l2reg']
        assert all(col in df.columns for col in by)
    else:
        groupby = ['lr', 'l2reg']
        assert 'alpha' not in df.columns
    agg = df.groupby(groupby).agg({'loss': ['mean', 'std'], 'is_nan': 'sum'})

    print('all hyperparameters:')
    with pd.option_context('display.float_format', '{:.4g}'.format):
        display(agg)

    if by is not None:
        print(f'best hyperparameters by {by}:')
        best_hps = agg[('loss', 'mean')].groupby(by).idxmin()
        with pd.option_context('display.float_format', '{:.4g}'.format):
            display(agg.loc[best_hps, :])
    else:
        print('best hyperparameters:')
        best_hp = agg[('loss', 'mean')].idxmin()
        with pd.option_context('display.float_format', '{:.4g}'.format):
            display(agg.loc[[best_hp]])


def plot_hps(
    df: pd.DataFrame,
    by_alpha: bool,
    ylim: tuple[None | float, None | float] = (None, None)
) -> so.Plot:
    """Plots hyperparameters.

    Plots a vertical (jittered) scatter plot:
        x-axis: lr
        y-axis: loss
        column within each learning rate: l2reg
        faceted by alpha if by_alpha is True

    Args:
        df: DataFrame with columns (lr, l2reg, seed, loss, is_nan) and possibly 'alpha'
        by_alpha: whether to find best hyperparameters per alpha
        y_lim: optional y-axis limits, useful if some losses are extremely large
    """
    df['lr_str'] = df['lr'].map(lambda x: f'{x:.1g}')
    p = (
        so.Plot(df, x='lr_str', y='loss', color='l2reg')
        .add(so.Dots(), so.Dodge())
        .add(so.Dot(), so.Agg(), so.Dodge())
        .add(so.Range(), so.Est(errorbar='sd'), so.Dodge())
        .limit(y=ylim)
    )
    if by_alpha:
        p = p.facet('alpha', wrap=2).layout(size=(10, 10))
    else:
        p = p.layout(size=(5, 5))
    return p


def get_df(
    fmt_str: str, model: str,
    cols: Sequence[str] | None,
    lrs: Sequence[float] | None = None,
    l2regs: Sequence[float] | None = None,
    alphas: Sequence[float] | None = None,
) -> pd.DataFrame:
    index_cols = []
    iter_products = {}
    if lrs is not None:
        index_cols.append('lr')
        iter_products['lr'] = lrs
    if l2regs is not None:
        index_cols.append('l2reg')
        iter_products['l2reg'] = l2regs
    if alphas is not None:
        index_cols.append('alpha')
        iter_products['alpha'] = alphas
    index_cols.append('seed')

    dfs = []
    for hps in itertools.product(*iter_products.values()):
        path = fmt_str.format(**dict(zip(iter_products.keys(), hps)))
        try:
            df = pd.read_csv(path, usecols=cols)
            df['model'] = model
            for key, hp in zip(iter_products.keys(), hps):
                df[key] = hp
            dfs.append(df)
        except Exception as e:
            print(e)

    df = pd.concat(dfs).set_index(index_cols, verify_integrity=True)
    return df


def print_eto_results(df: pd.DataFrame, print_best_hps: bool = False) -> None:
    """
    Args:
        df: DataFrame with index (alpha, lr, l2reg, seed),
            columns model, {train/test}_{task_loss/coverage}
    """
    if print_best_hps:
        df = df[~df['train_task_loss'].isna()]
        grouped = df.groupby(['alpha', 'lr', 'l2reg'])
        agg = grouped[['train_task_loss', 'test_task_loss', 'test_coverage']].agg(['mean', 'std'])
        agg['count'] = grouped.size()

        print('ETO performance by hyperparameters:')
        with pd.option_context('display.float_format', '{:.4g}'.format):
            display(agg[['train_task_loss', 'count']])

        best_hps = agg[('train_task_loss', 'mean')].groupby('alpha').idxmin()
        print('ETO results:')
        with pd.option_context('display.float_format', '{:.4g}'.format):
            display(agg.loc[best_hps])

    else:
        eto_results = (
            df.loc[df['model'] == 'eto']
            .groupby('alpha')[['test_task_loss', 'test_coverage']]
            .agg(['mean', 'std', 'count'])
        )
        print('ETO results:')
        display(eto_results)


def print_best_test_task_loss(df: pd.DataFrame, by: str) -> None:
    """
    Args:
        df: DataFrame with
            index: alpha, lr, l2reg, seed
            columns: model, {by}, test_task_loss, test_coverage
        by: column to use for determining best task loss, e.g., 'val_task_loss'
    """
    grouped = df[~df[by].isna()].groupby(['alpha', 'lr', 'l2reg'])
    agg = grouped[[by, 'test_task_loss', 'test_coverage']].agg(['mean', 'std'])
    agg['count'] = grouped.size()

    print('performance by hyperparameters:')
    with pd.option_context('display.float_format', '{:.4f}'.format):
        display(agg[[by, 'count']])

    best_hps = agg[(by, 'mean')].groupby('alpha').idxmin()
    print('best test task loss:')
    with pd.option_context('display.float_format', '{:.4f}'.format):
        display(agg.loc[best_hps])


def convert_to_long_df(df: pd.DataFrame) -> pd.DataFrame:
    long_df = df.melt(
        id_vars=['model', 'alpha', 'lr', 'l2reg', 'seed'],
        var_name='metric',
        value_name='value')
    long_df['model'] = (
        long_df['model']
        + '_lr' + long_df['lr'].map('{:.3g}'.format)
        + '_reg' + long_df['l2reg'].map('{:.3g}'.format)
    )
    long_df['split'] = long_df['metric'].map(lambda x: x.split('_')[0])
    long_df['metric'] = long_df['metric'].map(lambda x: x.split('_', maxsplit=1)[1])
    return long_df


def plot_eto_vs_e2e(long_df: pd.DataFrame, num_rows: int) -> so.Plot:
    """
    Args:
        long_df: long-form DataFrame with columns
            model, alpha, lr, l2reg, seed, metric, value, split
        num_rows: number of rows in the facet
    """
    return (
        so.Plot(long_df, x='alpha', y='value', color='model')
        .facet(row='metric', col='split')
        .add(so.Dots(), so.Dodge(), so.Jitter(.5))
        .share(y='row')
        .scale(x=so.Nominal())
        .layout(size=(10, num_rows * 3))
    )
