from __future__ import with_statement, print_function
import numpy as np
import random

from tsp_solver.greedy_numpy import solve_tsp

try:
    import psyco

    psyco.full()
except:
    pass

def random_generate(x_size,y_size):
    tmp = [[x, y] for x in range(x_size) for y in range(y_size)]
    random.shuffle(tmp)
    block = np.array([[2*x+1,2*y] for (x,y) in tmp])
    random.shuffle(tmp)
    rfid = np.array([[2*x,2*y+1] for (x,y) in tmp])

    return block, rfid

def generate_dist_matrix(block, rfid):
    assert len(block)==len(rfid)
    dist_matrix = np.eye(2*len(block))
    self_dist_matrix = np.sum(np.abs(block-rfid),axis=1)
    for i in range(2*len(block)) :
        for j in range(2 * len(block)):
            if i<=j:
                continue
            if i<len(block) or j>=len(block):
                dist_matrix[i][j] = np.inf
            elif i-len(block) == j:
                dist_matrix[i][j] = 0
            else:
                dist_matrix[i][j] = self_dist_matrix[i-len(block)] + np.sum(np.abs(rfid[i-len(block)]-block[j]))
    dist_matrix += dist_matrix.T - np.diag(dist_matrix.diagonal())
    return dist_matrix

def figure_plot(block, rfid, path):
    coordinate = np.concatenate([block, rfid], axis=0).T
    try:
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
    except ImportError as err:
        print("Can't show plot, matplotlib module not available:", err)
        print("Either install matplotlib or set an -o option")
        exit(2)
    # plt.plot(coordinate[0, path], coordinate[1, path], ':', block.T[0,:], block.T[1,:], 'bs', rfid.T[0, :], rfid.T[1, :], 'r*')
    # plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line, line2, line3, = ax.plot([], [], '-.k', block.T[0, :], block.T[1, :], 'bs', rfid.T[0, :], rfid.T[1, :], 'r*')
    ax.set_xlabel('Distance in X axis (unit cell)')
    ax.set_ylabel('Distance in Y axis (unit cell)')
    ax.set_title('Winter Camp Solution', fontsize=15)

    def update(data):
        line.set_xdata(data[0])
        line.set_ydata(data[1])
        return line,

    def data_gen():
        end = 0
        start = 0
        while end<len(path):

            # if end>4:
            #     start = end - 4
            yield coordinate[:, path[start:end]]
            end += 1

    ani = animation.FuncAnimation(fig, update, data_gen, interval=400)
    plt.show()

def main():
    from optparse import OptionParser

    parser = OptionParser( description = "Winter Camp TSP solver"  )
    parser.add_option( "-r", "--random", dest="random",  default=True,
                       help="Generate the coordination pairs of the certain rfid and block" )
    parser.add_option( "-x", "--xsize", dest="x_size", type="int", default=6,
                       help="Set the size in x" )
    parser.add_option( "-y", "--ysize", dest="y_size", type="int", default=6,
                       help="Set the size in y" )
    parser.add_option( "-p", "--plot",
                       dest="show_plot", default=True,
                       help="Whether to show the figure plot" )


    (options, args) = parser.parse_args()
    if options.random:
        x_size = options.x_size
        y_size = options.y_size
        block, rfid = random_generate(x_size, y_size)

    # print("block",block)
    # print("rfid",rfid)

    dist_matrix = generate_dist_matrix(block, rfid)
    path = solve_tsp(dist_matrix,optim_steps=1000000)

    show_path = [i + 1 if i < len(block) else -i + len(block) - 1 for i in path]
    print(show_path)

    if options.show_plot:
        figure_plot(block, rfid, path)

if __name__ == '__main__':
    main()


