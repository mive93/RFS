#! /usr/bin/python3
import matplotlib.pyplot as plt


def plot_lines(filename, prec_data, rec_data, f1_data, markers, labels, to_show=True):
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharey=True)
    images = list(range(1, len(prec_data[0]) + 1))

    loop_around = 30
    div = len(images) / loop_around
    img = list(range(loop_around))
    assert(len(images) % loop_around == 0)

    re = []
    pr = []
    f1 = []

    for j in range(len(prec_data)):
        re.append([0] * loop_around)
        pr.append([0] * loop_around)
        f1.append([0] * loop_around)

        for i in range(len(prec_data[j])):
            re[j][i % loop_around] = re[j][i % loop_around] + rec_data[j][i]
            pr[j][i % loop_around] = pr[j][i % loop_around] + prec_data[j][i]
            f1[j][i % loop_around] = f1[j][i % loop_around] + f1_data[j][i]

        re[j] = [x/div for x in re[j]]
        pr[j] = [x/div for x in pr[j]]
        f1[j] = [x/div for x in f1[j]]


    for i in range(len(prec_data)):
        ax1.plot(img, re[i], label=labels[i], marker=markers[i])
        ax2.plot(img, pr[i], label=labels[i], marker=markers[i])
        ax3.plot(img, f1[i], label=labels[i], marker=markers[i])

    ax1.set_xlabel("Recall")
    ax2.set_xlabel("Precision")
    ax3.set_xlabel("F1Score")

    ax1.set_ylim([0, 1.02])
    ax2.set_ylim([0, 1.02])
    ax3.set_ylim([0, 1.02])

    handles, labels = ax1.get_legend_handles_labels()
    plt.figlegend(handles, labels, bbox_to_anchor=(1, 0.5), loc="center left")
    plt.tight_layout()
    plt.savefig(filename, format="svg", bbox_inches="tight")
    plt.legend()
    if to_show:
        plt.show()
    plt.close('all')


def plot_boxplots(filename, prec_data, rec_data, f1_data, labels, to_show=True):

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

    plt.savefig(filename, format="svg", bbox_inches="tight")
    if to_show:
        plt.show()
    plt.close('all')


def plot_tracked_object(tracker):
    plt.figure()
    x = [det.x for det in tracker.dets.dets]
    y = [det.y for det in tracker.dets.dets]
    plt.plot(x, y)
    plt.show()
    plt.close('all')


def plot_all_tracked_objects(trackers, title, to_show=False):
    fig = plt.figure()
    fig.suptitle(title)
    axes = plt.gca()
    axes.set_xlim([0, 1080])
    axes.set_ylim([0, 720])
    plt.gca().set_aspect('equal', adjustable='box')
    for tracker in trackers.trackers:
        if tracker.confidence > 0.45:
            x = [det.x*1080 for det in tracker.dets.dets]
            y = [720 - det.y*720 for det in tracker.dets.dets]
            plt.plot(x, y, '.-', color=tracker.color)
    plt.savefig('../img/track/'+title+'.png', format="png", bbox_inches="tight")
    if to_show:
        plt.show()

    plt.close('all')
