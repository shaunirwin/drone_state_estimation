from matplotlib import pyplot as plt 
import struct


if __name__ == "__main__":
    filename = "build/sim.dat"

    struct_fmt = '=3f3fi'           # float[3], float[3], int
    struct_len = struct.calcsize(struct_fmt)
    struct_unpack = struct.Struct(struct_fmt).unpack_from

    results = []
    with open(filename, "rb") as f:
        while True:
            data = f.read(struct_len)
            if not data: break
            s = struct_unpack(data)
            results.append(s)

    print(results[0])
    print(results[1])
    print(results[2])

    # plot

    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
    ax[0].plot([p[0] for p in results], label="pos x")
    ax[0].plot([p[1] for p in results], label="pos y")
    ax[0].plot([p[2] for p in results], label="pos z")
    ax[0].legend()
    ax[0].grid()
    ax[1].plot([p[3] for p in results], label="vel x")
    ax[1].plot([p[4] for p in results], label="vel y")
    ax[1].plot([p[5] for p in results], label="vel z")
    ax[1].legend()
    ax[1].grid()
    plt.show()
