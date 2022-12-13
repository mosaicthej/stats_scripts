# functions to be used to calculate various statistics distributions and making plots and tests;
# need to take care of different distributions and different tests(binomial, poisson, normal, t, chi2, F, etc.)

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import rich
import scipy.stats as stats
import math


########
# I/O  #
########

# function that parse a string to a list of numbers (space separated)
def parse_string_to_list(string):
    return [float(x) for x in string.split()]

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

# function that plots a boxplot of a list of numbers