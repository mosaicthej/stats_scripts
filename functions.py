# functions to be used to calculate various statistics distributions and making plots and tests;
# need to take care of different distributions and different tests(binomial, poisson, normal, t, chi2, F, etc.)

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import rich
import scipy.stats as stats
import math
import termplotlib as tpl

from rich.layout import Layout
from rich import print
from rich.panel import Panel

########
# I/O  #
########

# function that parse a string to a list of numbers (space separated)
def parse_string_to_list(stri: str):
    return [float(x) for x in stri.split()]

####################
# basic statistics #
####################

# function that calculates the mean of a list of numbers
def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

# function that calculates the median of a list of numbers
def median(numbers):
    numbers = sorted(numbers)
    if len(numbers) % 2 == 0:
        return (numbers[len(numbers) // 2] \
            + numbers[len(numbers) // 2 - 1]) / 2
    else:
        return numbers[len(numbers) // 2]

# function that calculates the mode of a list of numbers
def mode(numbers):
    counts = dict()
    for number in numbers:
        if number in counts:
            counts[number] += 1
        else:
            counts[number] = 1
    max_count = max(counts.values())
    modes = [number for number, count in counts.items() if count == max_count]
    return modes

# function that calculates the variance of a list of numbers
def variance(numbers):
    mu = mean(numbers)
    return sum([(x - mu) ** 2 for x in numbers]) / len(numbers)

# function that calculates the standard deviation of a list of numbers
def standard_deviation(numbers):
    return math.sqrt(variance(numbers))

# function that return values and frequencies as a dictionary for a list of numbers
def values_and_frequencies(numbers):
    counts = dict()
    for number in numbers:
        if number in counts:
            counts[number] += 1
        else:
            counts[number] = 1
    return counts

# override!
#// function that calculates the range of a list of numbers
#// def range(numbers):
#//   return max(numbers) - min(numbers)

# function that calculates the interquartile range of a list of numbers
def interquartile_range(numbers):
    numbers = sorted(numbers)
    if len(numbers) % 2 == 0:
        return (median(numbers[len(numbers) // 2:]) \
            - median(numbers[:len(numbers) // 2]))
    else:
        return (median(numbers[len(numbers) // 2 + 1:]) \
            - median(numbers[:len(numbers) // 2]))

# function that displays the basic statistics of a list of numbers in form of an aligned table
def display_basic_statistics(numbers):
    print("Basic Statistics")
    print("Mean: ", mean(numbers))
    print("Median: ", median(numbers))
    print("Mode: ", mode(numbers))
    print("Variance: ", variance(numbers))
    print("Standard Deviation: ", standard_deviation(numbers))
    print("Range: ", max(numbers) - min(numbers))
    print("Interquartile Range: ", interquartile_range(numbers))

#####################
# sampling function #
#####################
# function that returns a list of random sample of size n from a list of population
def random_sample(population, n: int, rep=False):
    return np.random.choice(population, n, replace=rep)

# function that returns n random samples of size k from a list of population, also returns the mean of each sample
def random_samples(population, n: int, k: int, rep=False):
    samples = []
    means = []
    for i in range(n):
        sample = random_sample(population, k, rep)
        samples.append(sample)
        means.append(mean(sample))
    return samples, means


####################
# 1D plots         #
####################
# function that plots a histogram of a list of numbers
def plot_histogram(numbers, bins=10, title="Histogram", xlabel="x", ylabel="Frequency"):
    plt.hist(numbers, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# plot histogram, but in console directly (using termplotlib and rich, set detailed to True to see the values and frequencies)
def plot_histogram_console(numbers, bins=10, title="Histogram", xlabel="x", ylabel="Frequency", detailed=False):
    x = np.array(numbers)
    y = np.array([1] * len(x))
    counts, bin_edges = np.histogram(x, bins=bins)
    fig = tpl.figure()
    # default parameters for hist():
    # counts: List[int],
    # bin_edges: List[float],
    # orientation: str = "vertical",
    # max_width: int = 40,
    # grid=None,
    # bar_width: int = 1,
    # strip: bool = False,
    # force_ascii: bool = False,

    # if detailed is True, show the values and frequencies
    if detailed:
        fig.hist(counts, bin_edges, max_width=80, bar_width=1, orientation="horizontal", grid="y")
        rich.print(Panel(fig, title=title, border_style="blue"))
        # put values and frequencies in a table and print it using rich
        table = rich.table.Table(title="Values and Frequencies")
        table.add_column("Values", justify="right")
        table.add_column("Frequencies", justify="right")
        for i in range(len(counts)):
            table.add_row(str(bin_edges[i]), str(counts[i]))
        rich.print(table)
        
    else:
        fig.hist(counts, bin_edges, max_width=80, bar_width=1, orientation="horizontal")
        rich.print(Panel(fig, title=title, border_style="blue"))


    fig.hist(counts, bin_edges, max_width=80, bar_width=1)
    rich.print(Panel(fig, title=title, border_style="blue"))

