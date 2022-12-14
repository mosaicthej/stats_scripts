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

###########################
# Geometric Distributions #
###########################
# function that calculates the probability of a random variable being a certain value
def geometric_probability(p: float, k: int):
    if k < 1:
        raise ValueError("k cannot be less than 1")
    if p < 0 or p > 1:
        raise ValueError("p must be between 0 and 1")
    q = 1 - p
    return q ** (k - 1) * p

# function that calculates the probability of a random variable being less than or equal to a certain value
def geometric_cdf(p: float, k: int):
    if k < 1:
        raise ValueError("k cannot be less than 1")
    if p < 0 or p > 1:
        raise ValueError("p must be between 0 and 1")
    q = 1 - p
    probability = 0
    for i in range(k):
        probability += geometric_probability(p, i + 1)
    return probability

# function that calculates the probability of a random variable being greater than or equal to a certain value
def geometric_sf(p: float, k: int):
    if k < 1:
        raise ValueError("k cannot be less than 1")
    if p < 0 or p > 1:
        raise ValueError("p must be between 0 and 1")
    q = 1 - p
    return 1 - geometric_cdf(p, k)

# function that calculates the probability of a random variable being between a certain range
def geometric_between(p: float, a: int, b: int):
    if a > b:
        raise ValueError("a cannot be greater than b")
    if p < 0 or p > 1:
        raise ValueError("p must be between 0 and 1")
    q = 1 - p
    probability = 0
    for i in range(a, b + 1):
        probability += geometric_probability(p, i)
    return probability



###########################
# Continuous Distributions#
###########################

###########################
# Normal Distributions    #
###########################
# function that finds the z-score of a value
def z_score(mean: float, std: float, value: float):
    return (value - mean) / std

# function that finds the value of a z-score
def value(mean: float, std: float, z: float):
    return mean + (std * z)

# function that finds the z-score from a probability
def z_score_from_probability(probability: float):
    return stats.norm.ppf(probability)

# function that finds the probability from a z-score
def probability_from_z_score(z: float):
    return stats.norm.cdf(z)

# function that use normal probability density function to find the probability of a random variable being a certain value (use stats.norm)
def normal_probability(mean: float, std: float, value: float):
    return stats.norm.pdf(value, mean, std)

# function that use normal cumulative distribution function to find the probability of a random variable being less than or equal to a certain value
def normal_cdf(mean: float, std: float, value: float):
    return stats.norm.cdf(value, mean, std)

# function that use normal survival function to find the probability of a random variable being greater than or equal to a certain value
def normal_sf(mean: float, std: float, value: float):
    return stats.norm.sf(value, mean, std)

# function that use normal probability density function to find the probability of a random variable being between a certain range
def normal_between(mean: float, std: float, a: float, b: float):
    return stats.norm.cdf(b, mean, std) - stats.norm.cdf(a, mean, std)

# function that plots the normal probability density function
def normal_plot(mean: float, std: float, a: float, b: float):
    x = []
    y = []
    for i in np.arange(a, b, 0.01):
        x.append(i)
        y.append(normal_probability(mean, std, i))
    plt.plot(x, y)
    plt.title("Normal Probability Density Function")
    plt.xlabel("x")
    plt.ylabel("P(X = x)")
    plt.show(block=False)

# function that plots the normal cumulative distribution function
def normal_cdf_plot(mean: float, std: float, a: float, b: float):
    x = []
    y = []
    for i in np.arange(a, b, 0.01):
        x.append(i)
        y.append(normal_cdf(mean, std, i))
    plt.plot(x, y)
    plt.title("Normal Cumulative Distribution Function")
    plt.xlabel("x")
    plt.ylabel("P(X <= x)")
    plt.show(block=False)

# function that plots the normal survival function
def normal_sf_plot(mean: float, std: float, a: float, b: float):
    x = []
    y = []
    for i in np.arange(a, b, 0.01):
        x.append(i)
        y.append(normal_probability(mean, std, i))
    plt.plot(x, y)
    plt.title("Normal Survival Function")
    plt.xlabel("x")
    plt.ylabel("P(X >= x)")
    plt.show(block=False)

# function that plots the normal probability density function between a certain range (plot from mu-5*sigma to mu+5*sigma)
# for parts < a and > b, use a different color to show that they are not in the range
def normal_between_plot(mean: float, std: float, a: float, b: float):
    x = []
    y = []
    for i in np.arange(mean - 5 * std, mean + 5 * std, 0.01):
        x.append(i)
        y.append(normal_probability(mean, std, i))
    plt.plot(x, y)
    plt.fill_between(x, y, where=(np.array(x) >= a) & (np.array(x) <= b), color="red")
    plt.fill_between(x, y, where=(np.array(x) < a) | (np.array(x) > b), color="blue")
    plt.title("Normal Probability Density Function")
    plt.xlabel("x")
    plt.ylabel("P(X = x)")
    plt.show(block=False)


###########################
# confidence interval     #
###########################
# function that calculates the confidence interval of a normal distribution
def normal_confidence_interval(mean: float, std: float, n: int, confidence: float, detailed=False):
    if confidence < 0 or confidence > 1:
        raise ValueError("confidence must be between 0 and 1")
    if n <= 0:
        raise ValueError("n must be greater than 0")
    z = z_score_from_probability(confidence)
    margin_of_error = z * (std / math.sqrt(n))
    if detailed:
        print("z-score: " + str(z))
        print("margin of error: " + str(margin_of_error))
    return mean - margin_of_error, mean + margin_of_error

# function that calculates the confidence interval of a normal distribution
def normal_confidence_interval_from_std_error(mean: float, std_error: float, confidence: float, detailed=False):
    if confidence < 0 or confidence > 1:
        raise ValueError("confidence must be between 0 and 1")
    z = z_score_from_probability(confidence)
    margin_of_error = z * std_error
    if detailed:
        print("z-score: " + str(z))
        print("margin of error: " + str(margin_of_error))
    return mean - margin_of_error, mean + margin_of_error

# function that calculates the confidence interval for a proportion from sample data
def proportion_confidence_interval_from_data(data: list, confidence: float, detailed=False):
    if confidence < 0 or confidence > 1:
        raise ValueError("confidence must be between 0 and 1")
    p = sum(data) / len(data)
    n = len(data)
    z = z_score_from_probability(confidence)
    margin_of_error = z * math.sqrt((p * (1 - p)) / n)
    if detailed:
        print("z-score: " + str(z))
        print("margin of error: " + str(margin_of_error))
    return p - margin_of_error, p + margin_of_error

# function that calculates the confidence interval for a proportion from statistics
def proportion_confidence_interval_from_statistics(p: float, n: int, confidence: float, detailed=False):
    if confidence < 0 or confidence > 1:
        raise ValueError("confidence must be between 0 and 1")
    if n <= 0:
        raise ValueError("n must be greater than 0")
    z = z_score_from_probability(confidence)
    margin_of_error = z * math.sqrt((p * (1 - p)) / n)
    if detailed:
        print("z-score: " + str(z))
        print("margin of error: " + str(margin_of_error))
    return p - margin_of_error, p + margin_of_error

###########################
# proportion             #
###########################
# function that calculates the confidence interval of a proportion
def proportion_confidence_interval(p: float, n: int, confidence: float, detailed=False):
    if confidence < 0 or confidence > 1:
        raise ValueError("confidence must be between 0 and 1")
    if n <= 0:
        raise ValueError("n must be greater than 0")
    z = z_score_from_probability(confidence)
    margin_of_error = z * math.sqrt((p * (1 - p)) / n)
    if detailed:
        print("z-score: " + str(z))
        print("margin of error: " + str(margin_of_error))
    return p - margin_of_error, p + margin_of_error


###########################
# t-distribution          #
###########################
# when the population standard deviation is unknown, use t-distribution
# function that calculates the t-score from a probability with n-1 degrees of freedom
def t_score_from_probability(probability: float, n: int):
    return stats.t.ppf(probability, n - 1)

# function that calculates the probability from a t-score with n-1 degrees of freedom
def t_probability_from_score(score: float, n: int):
    return stats.t.cdf(score, n - 1)

# from a list of data, find the statiscs of the t-distribution
def t_distribution_from_data(data: list, detailed=False):
    n = len(data)
    mu = mean(data)
    std_err = standard_deviation_sample(data) / math.sqrt(n)
    if detailed:
        print("mean: " + str(mu))
        print("standard error: " + str(std_err))
    return mu, std_err

# function that plots the t-distribution
def t_plot(n: int, a: float, b: float):
    x = []
    y = []
    for i in np.arange(a, b, 0.01):
        x.append(i)
        y.append(stats.t.pdf(i, n - 1))
    plt.plot(x, y)
    plt.title("t-distribution")
    plt.xlabel("x")
    plt.ylabel("P(X = x)")
    plt.show(block=False)

# function that estimate a (1-alpha)% confidence interval for the population mean
def t_confidence_interval(data: list, confidence: float, detailed=False):
    if confidence < 0 or confidence > 1:
        raise ValueError("confidence must be between 0 and 1")
    n = len(data)
    mu, std_err = t_distribution_from_data(data, detailed)
    t = t_score_from_probability(confidence, n)
    margin_of_error = t * std_err
    if detailed:
        print("t-score: " + str(t))
        print("margin of error: " + str(margin_of_error))
    return mu - margin_of_error, mu + margin_of_error

###########################
# 2-sample t-test         #
###########################

# if the population standard deviation is known, use z-test
def two_sample_z_test_from_statistics(mu1: float, mu2: float, std1: float, std2: float, n1: int, n2: int, detailed=False):
    std_err = math.sqrt((std1 ** 2 / n1) + (std2 ** 2 / n2))
    z = (mu1 - mu2) / std_err
    p = probability_from_z_score(z)
    if detailed:
        print("z-score: " + str(z))
        print("p-value: " + str(p))
        print("p-value (2-tailed): " + str(p * 2))
        print("std_err: " + str(std_err))
    return z, p


# same as above, but with data as input
def two_sample_z_test_from_data(data1: list, data2: list, detailed=False):
    mu1 = mean(data1)
    mu2 = mean(data2)
    std1 = standard_deviation(data1)
    std2 = standard_deviation(data2)
    n1 = len(data1)
    n2 = len(data2)
    return two_sample_z_test_from_statistics(mu1, mu2, std1, std2, n1, n2, detailed)

# if the population standard deviation is unknown, use t-test
# when both samples standard deviations are unknown, but equal
def two_sample_t_test_same_sd_from_statistics(mu1: float, mu2: float, std: float, n1: int, n2: int, detailed=False):
    sd_pooled = math.sqrt(((n1 - 1) * (std ** 2) + (n2 - 1) * (std ** 2)) / (n1 + n2 - 2))
    std_err = sd_pooled * math.sqrt((1 / n1) + (1 / n2))
    t = (mu1 - mu2) / std_err
    p = t_probability_from_score(t, n1 + n2 - 2)
    if detailed:
        print("t-score: " + str(t))
        print("p-value: " + str(p))
        print("p-value (2-tailed): " + str(p * 2))
        print("pool standard deviation: " + str(sd_pooled))
        print("std_err: " + str(std_err))
    return t, p