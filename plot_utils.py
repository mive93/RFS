#! /usr/bin/python3
import matplotlib.pyplot as plt


def plot_lines(prec_data, rec_data, f1_data, markers, labels):
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharey=True)
    images = list(range(1, len(prec_data[0]) + 1))

    for i in range(len(prec_data)):
        ax1.plot(images, rec_data[i], label=labels[i], marker=markers[i])
        ax2.plot(images, prec_data[i], label=labels[i], marker=markers[i])
        ax3.plot(images, f1_data[i], label=labels[i], marker=markers[i])

    ax1.set_xlabel("Recall")
    ax2.set_xlabel("Precision")
    ax3.set_xlabel("F1Score")

    ax1.set_ylim([0, 1.02])
    ax2.set_ylim([0, 1.02])
    ax3.set_ylim([0, 1.02])

    handles, labels = ax1.get_legend_handles_labels()
    plt.figlegend(handles, labels, bbox_to_anchor=(1, 0.5), loc="center left")
    plt.tight_layout()
    plt.savefig('lines.svg', format="svg", bbox_inches="tight")
    plt.show()


def plot_boxplots(prec_data, rec_data, f1_data, labels):

    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharey=True)

    ax1.boxplot(rec_data)
    ax2.boxplot(prec_data)
    ax3.boxplot(f1_data)

    ax1.set_xlabel("Recall")
    ax2.set_xlabel("Precision")
    ax3.set_xlabel("F1Score")

    ax1.set_ylim([0, 1.02])
    ax2.set_ylim([0, 1.02])
    ax3.set_ylim([0, 1.02])

    plt.setp([ax1, ax2, ax3], xticklabels=labels)

    plt.tight_layout()

    plt.savefig('boxplots.svg', format="svg", bbox_inches="tight")
    plt.show()
