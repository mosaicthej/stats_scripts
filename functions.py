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

# function that calculates the variance of a list of numbers as a sample
def variance_sample(numbers):
    mu = mean(numbers)
    return sum([(x - mu) ** 2 for x in numbers]) / (len(numbers) - 1)

# function that calculates the standard deviation of a list of numbers
def standard_deviation(numbers):
    return math.sqrt(variance(numbers))

# function that calculates the standard deviation of a list of numbers as a sample
def standard_deviation_sample(numbers):
    return math.sqrt(variance_sample(numbers))

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
def interquartile_range(numbers, detailed=False):
    numbers = sorted(numbers)
    if len(numbers) % 2 == 0:
        q1 = median(numbers[:len(numbers) // 2])
        q3 = median(numbers[len(numbers) // 2:])
    else:
        q1 = median(numbers[:len(numbers) // 2])
        q3 = median(numbers[len(numbers) // 2 + 1:])
    if detailed:
        print("Q1: ", q1)        
        print("Q3: ", q3)
        print("median: ", median(numbers))
        print("IQR: ", q3 - q1)
    return q3 - q1


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

# function that plots a boxplot of a list of numbers
def plot_boxplot(numbers, title="Boxplot", xlabel="x", ylabel="y"):
    plt.boxplot(numbers)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# function that plots a stem and leaf plot of a list of numbers
def plot_stem_and_leaf(numbers, title="Stem and Leaf Plot", xlabel="x", ylabel="y"):
    plt.stem(numbers)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

#################################
# probability and combinatorics #
#################################
# function that calculates the factorial of a number
def factorial(n: int):
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n == 0:
        return 1
    return n * factorial(n - 1)

# function that calculates the binomial coefficient of n and k
def binomial_coefficient(n: int, k: int):
    if k > n:
        raise ValueError("k cannot be greater than n")
    if k < 0 or n < 0:
        raise ValueError("k and n cannot be negative")
    return factorial(n) / (factorial(k) * factorial(n - k))

# function that calculates the binomial probability of n and k
def binomial_probability(n: int, k: int, p: float, detailed=False):
    if k > n:
        raise ValueError("k cannot be greater than n")
    if k < 0 or n < 0:
        raise ValueError("k and n cannot be negative")
    if p < 0 or p > 1:
        raise ValueError("p must be between 0 and 1")
    q = 1 - p
    if detailed: # use rich to print the formula and local variables (very cool, varible values in different colors)
        print(f"[bold blue]P(X = {k}) = [bold green]{binomial_coefficient(n, k)} * [bold green]{p}^{k} * [bold green]{q}^{n - k}")
        print(f"[green] p = [red]{p}")
        print(f"[green] q = [red]{q}")
        print(f"[green] n = [red]{n}")
        print(f"[green] k = [red]{k}")
        

    return binomial_coefficient(n, k) * (p ** k) * (q ** (n - k))

# function that calculates the main statistics in binomial distribution of n and k
def binomial_distribution(n: int, k: int, p: float):
    if k > n:
        raise ValueError("k cannot be greater than n")
    if k < 0 or n < 0:
        raise ValueError("k and n cannot be negative")
    if p < 0 or p > 1:
        raise ValueError("p must be between 0 and 1")
    q = 1 - p
    mean = n * p
    variance = n * p * q
    standard_deviation = math.sqrt(n * p * q)
    print("Mean: ", mean)
    print("Variance: ", variance)
    print("Standard Deviation: ", standard_deviation)

# function that calculates the probability of a random variable being less than or equal to a certain value
def binomial_cdf(n: int, k: int, p: float):
    if k > n:
        raise ValueError("k cannot be greater than n")
    if k < 0 or n < 0:
        raise ValueError("k and n cannot be negative")
    if p < 0 or p > 1:
        raise ValueError("p must be between 0 and 1")
    q = 1 - p
    probability = 0
    for i in range(k + 1):
        probability += binomial_probability(n, i, p)
    return probability

# function that calculates the probability of a random variable being greater than or equal to a certain value (survival function)
def binomial_sf(n: int, k: int, p: float):
    if k > n:
        raise ValueError("k cannot be greater than n")
    if k < 0 or n < 0:
        raise ValueError("k and n cannot be negative")
    if p < 0 or p > 1:
        raise ValueError("p must be between 0 and 1")
    q = 1 - p
    probability = 0
    for i in range(k, n + 1):
        probability += binomial_probability(n, i, p)
    return probability

# function that calculates the probability of a random variable being between a certain range
def binomial_between(n: int, k: int, p: float, a: int, b: int):
    if k > n:
        raise ValueError("k cannot be greater than n")
    if k < 0 or n < 0:
        raise ValueError("k and n cannot be negative")
    if p < 0 or p > 1:
        raise ValueError("p must be between 0 and 1")
    if a > b:
        raise ValueError("a cannot be greater than b")
    q = 1 - p
    probability = 0
    for i in range(a, b + 1):
        probability += binomial_probability(n, i, p)
    return probability

# function that calculates the probability of a random variable being outside a certain range
def binomial_outside(n: int, k: int, p: float, a: int, b: int):
    if k > n:
        raise ValueError("k cannot be greater than n")
    if k < 0 or n < 0:
        raise ValueError("k and n cannot be negative")
    if p < 0 or p > 1:
        raise ValueError("p must be between 0 and 1")
    if a > b:
        raise ValueError("a cannot be greater than b")
    q = 1 - p
    probability = 0
    for i in range(a):
        probability += binomial_probability(n, i, p)
    for i in range(b + 1, n + 1):
        probability += binomial_probability(n, i, p)
    return probability

# probability in a hypergeometric distribution (when each trial is not independent)
def hypergeometric_probability(n: int, x: int, N: int, X: int):
    if x > n:
        raise ValueError("x cannot be greater than n")
    if x < 0 or n < 0:
        raise ValueError("x and n cannot be negative")
    if X > N:
        raise ValueError("X cannot be greater than N")
    if X < 0 or N < 0:
        raise ValueError("X and N cannot be negative")
    return binomial_coefficient(x, X) * binomial_coefficient(n - x, N - X) / binomial_coefficient(n, N)

#####################
# probability plots #
#####################
# function that plots the binomial probability distribution
def binomial_plot(n: int, k: int, p: float):
    if k > n:
        raise ValueError("k cannot be greater than n")
    if k < 0 or n < 0:
        raise ValueError("k and n cannot be negative")
    if p < 0 or p > 1:
        raise ValueError("p must be between 0 and 1")
    q = 1 - p
    x = []
    y = []
    for i in range(n + 1):
        x.append(i)
        y.append(binomial_probability(n, i, p))
    plt.bar(x, y)
    plt.title("Binomial Probability Distribution")
    plt.xlabel("k")
    plt.ylabel("P(X = k)")
    plt.show()

# function that plots the binomial cumulative distribution
def binomial_cdf_plot(n: int, k: int, p: float):
    if k > n:
        raise ValueError("k cannot be greater than n")
    if k < 0 or n < 0:
        raise ValueError("k and n cannot be negative")
    if p < 0 or p > 1:
        raise ValueError("p must be between 0 and 1")
    q = 1 - p
    x = []
    y = []
    for i in range(n + 1):
        x.append(i)
        y.append(binomial_cdf(n, i, p))
    plt.bar(x, y)
    plt.title("Binomial Cumulative Distribution")
    plt.xlabel("k")
    plt.ylabel("P(X <= k)")
    plt.show()

# function that plots the binomial survival function 
def binomial_sf_plot(n: int, k: int, p: float):
    if k > n:
        raise ValueError("k cannot be greater than n")
    if k < 0 or n < 0:
        raise ValueError("k and n cannot be negative")
    if p < 0 or p > 1:
        raise ValueError("p must be between 0 and 1")
    q = 1 - p
    x = []
    y = []
    for i in range(n + 1):
        x.append(i)
        y.append(binomial_sf(n, i, p))
    plt.bar(x, y)
    plt.title("Binomial Survival Function")
    plt.xlabel("k")
    plt.ylabel("P(X >= k)")
    plt.show()

# function that plots the binomial probability distribution between a certain range
def binomial_between_plot(n: int, k: int, p: float, a: int, b: int):
    if k > n:
        raise ValueError("k cannot be greater than n")
    if k < 0 or n < 0:
        raise ValueError("k and n cannot be negative")
    if p < 0 or p > 1:
        raise ValueError("p must be between 0 and 1")
    if a > b:
        raise ValueError("a cannot be greater than b")
    q = 1 - p
    x = []
    y = []
    for i in range(a, b + 1):
        x.append(i)
        y.append(binomial_probability(n, i, p))
    plt.bar(x, y)
    plt.title("Binomial Probability Distribution")
    plt.xlabel("k")
    plt.ylabel("P(X = k)")
    plt.show(block=False)

#########################
# Poisson Distributions #
#########################
# function that calculates the probability of a random variable being a certain value
def poisson_probability(l: float, k: int):
    if k < 0:
        raise ValueError("k cannot be negative")
    return (l ** k) * (math.e ** (-1*l)) / math.factorial(k)

# function that calculates the probability of a random variable being less than or equal to a certain value
def poisson_cdf(l: float, k: int):
    if k < 0:
        raise ValueError("k cannot be negative")
    probability = 0
    for i in range(k + 1):
        probability += poisson_probability(l, i)
    return probability

# function that calculates the probability of a random variable being greater than or equal to a certain value
def poisson_sf(l: float, k: int):
    if k < 0:
        raise ValueError("k cannot be negative")
    return 1 - poisson_cdf(l, k - 1)

# function that calculates the probability of a random variable being between a certain range
def poisson_between(l: float, a: int, b: int):
    if a > b:
        raise ValueError("a cannot be greater than b")
    probability = 0
    for i in range(a, b + 1):
        probability += poisson_probability(l, i)
    return probability


# function that plots the poisson probability distribution
def poisson_plot(l: float, k: int):
    if k < 0:
        raise ValueError("k cannot be negative")
    x = []
    y = []
    for i in range(k + 1):
        x.append(i)
        y.append(poisson_probability(l, i))
    plt.bar(x, y)
    plt.title("Poisson Probability Distribution")
    plt.xlabel("k")
    plt.ylabel("P(X = k)")
    plt.show()

# function that plots the poisson cumulative distribution
def poisson_cdf_plot(l: float, k: int):
    if k < 0:
        raise ValueError("k cannot be negative")
    x = []
    y = []
    for i in range(k + 1):
        x.append(i)
        y.append(poisson_cdf(l, i))
    plt.bar(x, y)
    plt.title("Poisson Cumulative Distribution")
    plt.xlabel("k")
    plt.ylabel("P(X <= k)")
    plt.show()

# function that plots the poisson survival function
def poisson_sf_plot(l: float, k: int):
    if k < 0:
        raise ValueError("k cannot be negative")
    x = []
    y = []
    for i in range(k + 1):
        x.append(i)
        y.append(poisson_sf(l, i))
    plt.bar(x, y)
    plt.title("Poisson Survival Function")
    plt.xlabel("k")
    plt.ylabel("P(X >= k)")
    plt.show()

# function that plots the poisson probability distribution between a certain range
def poisson_between_plot(l: float, k: int, a: int, b: int):
    if k < 0:
        raise ValueError("k cannot be negative")
    if a > b:
        raise ValueError("a cannot be greater than b")
    x = []
    y = []
    for i in range(a, b + 1):
        x.append(i)
        y.append(poisson_probability(l, i))
    plt.bar(x, y)
    plt.title("Poisson Probability Distribution")
    plt.xlabel("k")
    plt.ylabel("P(X = k)")
    plt.show(block=False)

