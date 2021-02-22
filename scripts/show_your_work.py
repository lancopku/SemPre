import argparse
import os

import pandas as pd
import seaborn as sns

sns.set()
from matplotlib import pyplot as plt
from scipy.special import comb


def safe_equal(a, b, eps=1e-6):
    return -eps <= (a - b) <= eps


def _show_your_work_2_unbiased(values):
    values = sorted(values)
    size = len(values)

    def vn(n):
        de = comb(size, n)
        return sum(v * comb(j, n - 1) / de for j, v in enumerate(values))

    expected_maximums = [vn(i) for i in range(1, size + 1)]

    assert safe_equal(
        expected_maximums[0], sum(values) / size
    ), f"{expected_maximums[0]} != {sum(values) / len(values)}"
    assert all(v <= values[-1] for v in expected_maximums)

    return expected_maximums


def _show_your_work_2(values):
    values = sorted(values)
    size = len(values)

    def vn(n):
        sizen = size ** n
        return sum((((j + 1) ** n - j ** n) / sizen) * v for j, v in enumerate(values))

    expected_maximums = [vn(i) for i in range(1, size + 1)]

    assert safe_equal(
        expected_maximums[0], sum(values) / size
    ), f"{expected_maximums[0]} != {sum(values) / len(values)}"
    assert all(v <= values[-1] for v in expected_maximums)

    return expected_maximums


def _show_your_work(values):
    # v can be x1, x2, ..., xt
    steps = sorted(list(set(values)))
    # pi = p(v<=xi)
    probs = [sum(1 if v <= step else 0 for v in values) / len(values) for step in steps]
    # evj = xt - (xi+1 - xi)pi^j (i=0,...,t-1)
    expected_maximums = [
        steps[-1]
        - sum(
            (steps[j + 1] - steps[j]) * (probs[j] ** i)
            for j in range(0, len(steps) - 1)
        )
        for i in range(1, len(values) + 1)
    ]
    # print(steps)
    # print(probs)
    # print(expected_maximums)
    # print(sum(values) / len(values))
    assert safe_equal(
        expected_maximums[0], sum(values) / len(values)
    ), f"{expected_maximums[0]} != {sum(values) / len(values)}"
    assert all(v <= steps[-1] for v in expected_maximums)

    return expected_maximums


def draw(y1, y2, y1_name, y2_name):
    xs = []
    ys = []
    series = []
    types = []

    length = len(y1)
    xs.extend(list(range(1, length + 1)))
    series.extend([y1_name] * length)
    types.extend(["Biased"] * length)
    ys.extend(_show_your_work_2(y1))

    length = len(y1)
    xs.extend(list(range(1, length + 1)))
    series.extend([y1_name] * length)
    types.extend(["Unbiased"] * length)
    ys.extend(_show_your_work_2_unbiased(y1))

    length = len(y2)
    xs.extend(list(range(1, length + 1)))
    series.extend([y2_name] * length)
    types.extend(["Biased"] * length)
    ys.extend(_show_your_work_2(y2))

    length = len(y2)
    xs.extend(list(range(1, length + 1)))
    series.extend([y2_name] * length)
    types.extend(["Unbiased"] * length)
    ys.extend(_show_your_work_2_unbiased(y2))
    data = pd.DataFrame({"x": xs, "y": ys, "Series": series, "Types": types})

    # data = pd.DataFrame(
    #     {
    #         "x": range(1, max(len(y1), len(y2)) + 1),
    #         y1_name: pd.Series(_show_your_work(y1)),
    #         y2_name: pd.Series(_show_your_work(y2)),
    #         'y1_name'
    #     }
    # )
    # data_ = pd.melt(data, ["x"])

    with sns.axes_style("white"):

        plt.figure()
        ax = sns.lineplot(
            x="x",
            y="y",
            hue="Series",
            style="Types",
            #           markers=True,
            estimator=None,
            # dashes=False,
            data=data,
        )
        # plt.axhline(max(data[y1_name]))
        # plt.axhline(max(data[y2_name]))
        ax.set(
            xlabel="Number of Trials",
            ylabel="Expected Maximum Validation Performance",
            # xlim=(1, max(len(y1), len(y2)) + 1)
            # xticks=list(range(1, max(len(y1), len(y2)) + 1)),
            # xticklabels=list(range(1, max(len(y1), len(y2)) + 1)),
        )
        # ax.set(xticklabels=[],xlabel=None, yticklabels=[], ylabel=None)
        # handles, labels = ax.get_legend_handles_labels()
        # ax.legend(handles=handles[1:], labels=labels[1:])
        plt.show()


def read_file(file):
    numbers = []
    with open(file, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if line:
                numbers.append(float(line))
    return numbers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data1", type=str, help="file containing numbers one per line")
    parser.add_argument("data2", type=str, help="file containing numbers one per line")
    args = parser.parse_args()
    draw(
        read_file(args.data1),
        read_file(args.data2),
        os.path.basename(args.data1),
        os.path.basename(args.data2),
    )


if __name__ == "__main__":
    main()
