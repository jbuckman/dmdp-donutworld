import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import csv

def transparent_cmap(cmap, N=255):
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0, 0.8, N+4)
    return mycmap
mycmap = transparent_cmap(plt.cm.Blues)

if __name__ == "__main__":
    ## plot charts
    with open('out.csv', 'rb') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        data = [line for line in datareader]
    try:
        idx, loss, reward_loss, transition_loss, _, empirical_value_diff, theoretical_value_diff = zip(*data[:-1])

        idx = [float(x) for x in idx]
        loss = [float(x) for x in loss]
        reward_loss = [float(x) for x in reward_loss]
        transition_loss = [float(x) for x in transition_loss]
        empirical_value_diff = [float(x) for x in empirical_value_diff]
        theoretical_value_diff = [float(x) for x in theoretical_value_diff]

        plt.plot(idx, loss, label='loss')
        plt.plot(idx, empirical_value_diff, label='Empirical Value Diff')
        plt.plot(idx, theoretical_value_diff, label='Theoretical Value Diff')
        plt.legend()
        plt.savefig("match_theory.png")
        plt.close()

        plt.plot(idx, reward_loss, label='Reward Loss')
        plt.plot(idx, transition_loss, label='Transition Loss')
        plt.legend()
        plt.savefig("optimization.png")
        plt.close()
    except ValueError:
        print "plotting heatmaps only..."

    ## plot heatmaps
    pairs = []
    for source_i in [i+j for i in [70, 300, 325, 525] for j in [0,602,602*2,602*3]]:
        try:
            # plot env
            with open("env%d.csv" % source_i) as f:
                datareader = csv.reader(f, delimiter=',')
                data = [line for line in datareader]
            size = 64 if max([int(d[0]) for d in data]) > 32 else 32
            env = np.zeros([size,size])
            for i, j, h in data:
                env[int(i),int(j)] = float(h)

            plt.xlim(1, size-1)
            plt.ylim(1, size-1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(env, cmap='gray')
            plt.imshow((env == 0).astype(np.float), cmap=mycmap)
            plt.savefig("env%d.png" % source_i)
            plt.close()

            # plot heatmap
            with open("heatmap%d.csv" % source_i) as f:
                datareader = csv.reader(f, delimiter=',')
                data = [line for line in datareader]
            size = 64 if max([int(d[0]) for d in data]) > 32 else 32
            heatmap = np.zeros([size,size])
            for i, j, h in data:
                heatmap[int(i), int(j)] = float(h)

            plt.xlim(1,size-1)
            plt.ylim(1,size-1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(heatmap, cmap='hot', interpolation='nearest')
            plt.imshow((env == 0).astype(np.float), cmap=mycmap)
            plt.savefig("heatmap%d.png" % source_i)
            plt.close()

        except IOError:
            pass