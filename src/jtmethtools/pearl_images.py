import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from copy import copy

"""Plot strings of pearls representing methylation. Interface is currently a little clunky, here's a worked
example:

```python
import jtmethtools as jtm
import matplotlib.pyplot as plt

# create methylation figure
fig = jtm.pearl_images.MethylationFigure()

# iterate through alignments in a bam file and add them to the figure
for i, aln in enumerate(jtm.iter_bam(bamfn)):
    if i > 20:
        break
    i*=2
    
    # record alignment as True=mCpG, False=unmethylated-CpG, None=Not-a-CpG
    metlist = []
    for m in aln.metstr:
        if m.lower() == 'z':
            metlist.append(m == 'Z')
        else:
            metlist.append(None)
    
    # add the circles and the line
    fig.add_methylation_list(0, i, methylation=metlist)


# render the figure
fig.render(sz_mult=1.5)
# show the figure
plt.show()
```
"""


class MethylationFigure:
    """Plot circles on a line representing methylation
    state of reads"""

    def __init__(self):
        self.pearl_sz = 0.6
        self._patches = []
        self._lines = []
        self.lims = [np.inf, -np.inf, np.inf, -np.inf]

    def _add_unmethylated(self, x, y) -> None:
        circle = plt.Circle(
            (x, y),
            self.pearl_sz,
            edgecolor="#333333",
            facecolor='#fbfbfb',
            linewidth=1
        )

        self._patches.append(circle)

    def _add_methylated(self, x, y) -> None:
        circle = plt.Circle(
            (x, y),
            self.pearl_sz,
            facecolor="#801a00",
            edgecolor='#000000',
            linewidth=1
        )
        self._patches.append(circle)

    def _add_line(self, x, x_max, y) -> None:
        self._lines.append(
            plt.Line2D(
                [x, x_max],
                [y, y],
                color='black',
                linewidth=0.5,
                linestyle='-'
            )
        )

    def _update_limits(self, x, x_max, y):
        if x < self.lims[0]:
            self.lims[0] = x
        if x_max > self.lims[1]:
            self.lims[1] = x_max
        if y < self.lims[2]:
            self.lims[2] = y
        if y > self.lims[3]:
            self.lims[3] = y


    def add_methylation_list(self, x, y, methylation: list[bool | None]):
        """Add string of circles to the image where the vaules True|False|None
        will be rendered as:
            True: Dark red circle
            False: White circle
            None: Nothing"""
        # update x/y limits
        x_max = x + len(methylation)
        self._update_limits(x, x_max, y)

        # add the circle patches
        for i, m in enumerate(methylation):
            if m is None:
                continue
            self._add_methylated(x + i, y) if m else self._add_unmethylated(x + i, y)

        self._add_line(x, x_max, y)


    def render(self, sz_mult=1.):
        """Render the figure with the added patches."""

        # make the default == 1 but also be pretty small
        sz_mult /= 5

        # the interaction of figsize with set_aspect is confusing.
        # I think it's limited by the smallest
        fig, ax = plt.subplots(figsize=(
            sz_mult * (2 + self.lims[1] - self.lims[0]),
            sz_mult * (2 + self.lims[3] - self.lims[2])
        ))
        ax.set_xlim(self.lims[0] - 1, self.lims[1])
        ax.set_ylim(self.lims[2] - 1, self.lims[3] + 1)
        ax.set_aspect('equal', adjustable='box')

        for patch in self._patches:
            ax.add_patch(copy(patch))
        for line in self._lines:
            ax.add_line(copy(line))

        ax.axis('off')

    def plot_random_reads(
            self,
            region_width=45, read_width=30, n_cpg=13,
            n_reads=50, n_hyper_reads=12, max_gap=7,
            render=True
    ):
        """Plot random set of reads"""
        cpg_pos = sorted(
            np.random.choice(list(range(region_width)), size=n_cpg, replace=False)
        )
        cpg_beta_1 = np.random.uniform(0, .3, n_cpg)
        cpg_beta_2 = np.random.uniform(.5, 1, n_cpg)

        read_i_hyper = np.random.choice(list(range(n_reads)), size=n_hyper_reads)

        biggest_gap = max_gap+1
        # Regenerate CpG positions until maximum gap constraint satisfied
        while biggest_gap > max_gap:
            cpg_pos = sorted(
                np.random.choice(list(range(region_width)), size=n_cpg, replace=False)
            )
            biggest_gap = max([cpg_pos[i + 1] - cpg_pos[i] for i in range(n_cpg - 1)])

        # Generate synthetic methylation reads across random starts
        for read_i in range(n_reads):

            is_hyper = (read_i in read_i_hyper)
            cpg_beta = cpg_beta_2 if is_hyper else cpg_beta_1

            read_start = np.random.choice(list(range(0, region_width - read_width)))
            read = []
            # Builds methylation read by sampling per-locus probabilities
            for locus in range(read_start, read_width + read_start):
                if locus in cpg_pos:
                    cpg_i = cpg_pos.index(locus)
                    m = np.random.uniform() > cpg_beta[cpg_i]
                    read.append(m)
                else:
                    read.append(None)
            y = read_i * 1.5

            self.add_methylation_list(read_start, y, read)

        if render:
            self.render()


def ttest_methylation_figure():
    """Test the MethylationFigure class."""
    # create the figure object
    fig = MethylationFigure()
    # add some random reads
    fig.add_methylation_list(0, 0, [True, False, True, True, False])
    fig.add_methylation_list(1, 2, [False, True, False])
    fig.add_methylation_list(2, 1, [True, True, None, None, True])
    for rowi in range(0, 10):
        fig.add_methylation_list(1, 3 + rowi, [False, True, False])
    # render the figure
    fig.render(sz_mult=1.5)
    # show the figure
    plt.show()