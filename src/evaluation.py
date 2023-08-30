#!/usr/bin/env python3
import click
from downstream.evaluation import evaluate_metrics

@click.command()
@click.argument('pid_go', type=click.Path(exists=True))
@click.argument('pid_go_sc', type=click.Path(exists=True))
@click.argument('ic_file', type=click.Path(exists=True))
@click.option("--without_m_aupr", default=True, is_flag=True)
def main(pid_go, pid_go_sc, ic_file, without_m_aupr):
    (fmax_, t_), aupr_, smin_, maupr_ = evaluate_metrics(pid_go, pid_go_sc, ic_file = ic_file, if_m_aupr = without_m_aupr)
    print(F'Fmax: {fmax_:.3f} {t_:.2f}', F'AUPR: {aupr_:.3f}', F'Smin: {smin_:.3f}', F'M-AUPR: {maupr_:.3f}')

if __name__ == '__main__':
    main()
