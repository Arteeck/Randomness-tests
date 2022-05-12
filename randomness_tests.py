from itertools import islice, groupby, permutations
from collections import Counter, defaultdict
from numpy import sqrt
from scipy.stats import chi2, norm
from math import factorial

A = [[4529.4, 9044.9, 13568, 18091, 22615, 27892],
     [9044.9, 18097, 27139, 36187, 45234, 55789],
     [13568, 27139, 40721, 54281, 67852, 83685],
     [10891, 36187, 54281, 72414, 90470, 111580],
     [22615, 45234, 67852, 90470, 113262, 139476],
     [27892, 55789, 83685, 111580, 139476, 172860]]

B = [1 / 6, 5 / 24, 11 / 120, 19 / 720, 29 / 5040, 1 / 890]


def main():
    count = 10000
    prime_number = 5
    mod = 10 + prime_number
    multiplier = 30 - prime_number
    incrementer = 71
    random_sequence = list(islice(create_lcg(mod, multiplier, incrementer, 1), 0, count))
    kolmogorov_smirnov_test(random_sequence, mod)
    chi_squared_test(random_sequence, mod, 10)
    gap_test(random_sequence, mod)
    poker_test(random_sequence, mod)
    serial_test(random_sequence, mod)
    permutation_test(random_sequence)
    runs_up_down_test(random_sequence)
    runs_up_length_test(random_sequence)
    runs_down_length_test(random_sequence)


def kolmogorov_smirnov_test(random_list, mod):
    length = len(random_list)
    sorted_random_list = sorted(map(lambda x: x / mod, random_list))
    d_plus = max(map(lambda i: (i + 1) / length - sorted_random_list[i], range(length)))
    d_minus = max(map(lambda i: sorted_random_list[i] - i / length, range(length)))
    d_stat = max(d_plus, d_minus)
    critical_value = 1.36 / sqrt(length)
    if d_stat > critical_value:
        print("По тесту Колмогорова-Смирнова данная последовательность не является случайной")
    else:
        print("По тесту Колмогорова-Смирнова нет оснований считать, что данная последовательность не является случайной")


def chi_squared_test(random_list, mod, intervals_count):
    frequencies = defaultdict(int)
    frequencies.update(
        map(lambda z: (z[0], len(list(z[1]))), groupby(sorted(map(lambda x: x / mod, random_list)), lambda y: int(y * intervals_count))))
    expected = len(random_list) / intervals_count
    chi_square = sum(map(lambda x: (frequencies[x] - expected) ** 2 / expected, range(intervals_count)))
    critical_value = chi2.ppf(0.95, intervals_count - 1)
    if chi_square > critical_value:
        print("По тесту Хи-квадрат данная последовательность не является случайной")
    else:
        print("По тесту Хи-квадрат нет оснований считать, что данная последовательность не является случайной")


def calculate_gaps_length(random_list):
    current_gaps = defaultdict(int)
    result_gaps = list()
    for number in random_list:
        for i in current_gaps.keys():
            if i != number:
                current_gaps[i] += 1
        if current_gaps[number] != 0:
            result_gaps.append(current_gaps[number])
        current_gaps[number] = 0
    return Counter(result_gaps)


def gap_test(random_list, mod):
    gaps = defaultdict(int)
    gaps.update(calculate_gaps_length(random_list))
    gaps_count = sum(gaps.values())
    gap_length = 4
    current_gap_min_length = 0
    cumulative_gaps_count = 0
    max_difference = -1
    while cumulative_gaps_count < gaps_count:
        next_gap_min_length = current_gap_min_length + gap_length
        cumulative_gaps_count += sum(gaps[i] for i in range(current_gap_min_length, next_gap_min_length))
        current_difference = abs(cumulative_gaps_count / gaps_count - (1 - (1 - 1 / mod) ** next_gap_min_length))
        if current_difference > max_difference:
            max_difference = current_difference
        current_gap_min_length = next_gap_min_length
    critical_value = 1.36 / sqrt(len(random_list))
    if max_difference > critical_value:
        print("По GAP тесту данная последовательность не является случайной")
    else:
        print("По GAP тесту нет оснований считать, что данная последовательность не является случайной")


def create_poker_values(mod):
    if mod < 10 or mod > 100:
        raise ValueError("В данный момент покер-тест поддерживает модуль не больше 100 и не меньше 10")
    pair_numbers = int((mod - 1) / 11)
    return 10 / mod, (mod - 10 - pair_numbers) / mod, pair_numbers / mod


def poker_test(random_list, mod):
    expected_poker_counts = [i * len(random_list) for i in create_poker_values(mod)]
    poker_values_count = len(expected_poker_counts)
    pair_numbers = len(list(filter(lambda x: x != 0 and x % 11 == 0, random_list)))
    one_digit_numbers = len(list(filter(lambda x: x < 10, random_list)))
    not_pair_numbers = len(random_list) - pair_numbers - one_digit_numbers
    actual_poker_counts = one_digit_numbers, not_pair_numbers, pair_numbers
    chi_square = sum(map(lambda x: (actual_poker_counts[x] - expected_poker_counts[x]) ** 2 / expected_poker_counts[x], range(poker_values_count)))
    critical_value = chi2.ppf(0.95, poker_values_count - 1)
    if chi_square > critical_value:
        print("По покер-тесту данная последовательность не является случайной")
    else:
        print("По покер-тесту нет оснований считать, что данная последовательность не является случайной")


def serial_test(random_list, mod):
    counter_cells = Counter([(random_list[2 * i], random_list[2 * i + 1]) for i in range(int(len(random_list) / 2))])
    counter_cells_as_dict = defaultdict(int)
    counter_cells_as_dict.update(counter_cells)
    expected = len(random_list) / (2 * (mod ** 2))
    chi_square = sum(map(lambda x: (counter_cells_as_dict[x] - expected) ** 2 / expected, [(i, j) for i in range(mod) for j in range(mod)]))
    critical_value = chi2.ppf(0.95, mod ** 2 - 1)
    if chi_square > critical_value:
        print("По Serial тесту данная последовательность не является случайной")
    else:
        print("По Serial тесту нет оснований считать, что данная последовательность не является случайной")


def permutation_test(random_list):
    chunk_size = 3
    ordered_chunks = [order_chunk(random_list[i:i + chunk_size]) for i in range(0, len(random_list), chunk_size)]
    ordered_chunks_as_dict = defaultdict(int)
    ordered_chunks_as_dict.update(Counter(ordered_chunks))

    expected = len(random_list) / chunk_size / factorial(chunk_size)
    chi_square = sum(map(lambda x: (ordered_chunks_as_dict[x] - expected) ** 2 / expected, permutations(range(3))))
    critical_value = chi2.ppf(0.95, factorial(chunk_size) - 1)
    if chi_square > critical_value:
        print("По тесту перестановок данная последовательность не является случайной")
    else:
        print("По тесту перестановок нет оснований считать, что данная последовательность не является случайной")


def order_chunk(chunk):
    ordered_indices = sorted(range(len(chunk)), key=lambda k: chunk[k])
    return tuple(map(lambda x: ordered_indices.index(x), range(len(chunk))))


def runs_up_down_test(random_list):
    length = len(random_list)
    up_down_count = len(list(groupby([random_list[i + 1] > random_list[i] for i in range(length - 1)])))
    mean = (2 * length - 1) / 3
    variance = (16 * length - 29) / 90
    z_stat = (up_down_count - mean) / sqrt(variance)
    critical_value = norm.ppf(1 - (1 - 0.95) / 2)
    if abs(z_stat) > critical_value:
        print("По тесту runs up & down последовательность не является случайной")
    else:
        print("По тесту runs up & down нет оснований считать, что данная последовательность не является случайной")


def runs_up_length_test(random_list):
    classes = 6
    length = len(random_list)
    descending_indices = [0] + [i for i in range(1, length) if random_list[i] < random_list[i - 1]] + [length]
    ascending_lists_length = map(lambda x: x if x < classes else classes,
                                 [descending_indices[i + 1] - descending_indices[i] for i in range(len(descending_indices) - 1)])

    asc_lists_length_as_dict = defaultdict(int)
    asc_lists_length_as_dict.update(Counter(ascending_lists_length))

    chi_square = sum(
        [(asc_lists_length_as_dict[i] - length * B[i - 1]) * (asc_lists_length_as_dict[j] - length * B[j - 1]) * A[i - 1][j - 1]
         for i in range(1, classes + 1) for j in range(1, classes + 1)]) / length
    critical_value = chi2.ppf(0.95, classes)
    if chi_square > critical_value:
        print("По тесту runs up length последовательность не является случайной")
    else:
        print("По тесту runs up length нет оснований считать, что данная последовательность не является случайной")


def runs_down_length_test(random_list):
    classes = 6
    length = len(random_list)
    ascending_indices = [0] + [i for i in range(1, length) if random_list[i] > random_list[i - 1]] + [length]
    descending_lists_length = map(lambda x: x if x < classes else classes,
                                  [ascending_indices[i + 1] - ascending_indices[i] for i in range(len(ascending_indices) - 1)])

    desc_lists_length_as_dict = defaultdict(int)
    desc_lists_length_as_dict.update(Counter(descending_lists_length))

    chi_square = sum(
        [(desc_lists_length_as_dict[i] - length * B[i]) * (desc_lists_length_as_dict[j] - length * B[j]) * A[i][j]
         for i in range(classes) for j in range(classes)]) / length
    critical_value = chi2.ppf(0.95, classes)
    if chi_square > critical_value:
        print("По тесту runs down length последовательность не является случайной")
    else:
        print("По тесту runs down length нет оснований считать, что данная последовательность не является случайной")


def create_lcg(mod, multiplier, incrementer, seed):
    value = seed
    while True:
        value = (multiplier * value + incrementer) % mod
        yield value


if __name__ == '__main__':
    main()
