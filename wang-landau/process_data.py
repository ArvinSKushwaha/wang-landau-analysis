from glob import glob

import matplotlib as mpl
import polars as pl

from matplotlib import pyplot as plt

mpl.use('module://matplotlib-backend-sixel')

L = [8, 16, 32]

for N in L:
    f = [i for _, i in sorted([(float(f[14 + len(str(N)):-4]), f) for f in glob(f'./ising-{N}/data-*.csv')], reverse=True)][-1]

    df = pl.read_csv(f)
    df = df.filter(pl.col('entropy') > 0)

    plt.plot(df['energies'] / N**2, (df['entropy'] - df['entropy'].min()) / N**2, label=f"${N} \\times {N}$")

plt.legend()
plt.xlabel('$E / N^2$')
plt.ylabel('$S(E) / N^2$')
plt.title(f'Density of States for Periodic Ising Model')
plt.grid(True)
plt.savefig('../report/DOS_Ising.png')
plt.show()
