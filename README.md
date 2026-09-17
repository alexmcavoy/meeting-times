# meeting-times

Code for computing critical benefit-to-cost ratios $(b/c)^*$ for death-Birth updating with accumulated payoffs on networks. This code is used by the paper "Meeting times on graphs in near-cubic time" by Alex McAvoy ([arXiv](https://arxiv.org/abs/2604.18872)).

## Requirements

Install dependencies with ```pip install numpy scipy networkx matplotlib```.

## Basic use

### Figure 1 (Facebook networks)

```bash
python figure1.py             # compute results and plot
python figure1.py --plot-only # plot from cached results
```

Outputs `Fig1.pdf`.

### Figure 2 (email network)

```bash
python figure2.py             # compute results and plot
python figure2.py --plot-only # plot from cached results
```

Outputs `Fig2.pdf`. The email edge list is downloaded automatically from [SNAP](https://snap.stanford.edu/data/email-Eu-core.html) if not already present.

### Benchmark

```bash
python benchmark.py	# benchmark on random and empirical graphs (time and space)
```

## AI disclosure

Claude Code (Opus 4.6 and Fable 5.1) were used to check and streamline the implementation, as well as develop initial benchmarks. All responsibility remains with the author.