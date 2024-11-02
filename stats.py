import numpy as np
import pickle
import lzma
import matplotlib.pyplot as plt

STATS_FOLDER = "./stats/"

fd_data = pickle.load(lzma.open(f"{STATS_FOLDER}federated_model.npz", "rb"))
lcl_data = pickle.load(lzma.open(f"{STATS_FOLDER}local_model.npz", "rb"))


def getSpeedStats(fd, lcl):
    speed_fd_data = []
    speed_lcl_data = []
    mean_speed_fd = 0
    mean_speed_lcl = 0
    for i in range(0, len(fd)):
        speed_fd_data.append(fd[i].car_speed)
        speed_lcl_data.append(lcl[i].car_speed)
        mean_speed_fd += fd[i].car_speed
        mean_speed_lcl += lcl[i].car_speed
    mean_speed_fd = mean_speed_fd / len(fd)
    mean_speed_lcl = mean_speed_lcl / len(lcl)
    print(f"Mean speed with fd : {mean_speed_fd}")
    print(f"Mean speed with lcl : {mean_speed_lcl}")
    plt.figure()
    plt.plot(range(0, len(speed_fd_data)), speed_fd_data)
    plt.plot(range(0, len(speed_lcl_data)), speed_lcl_data)
    plt.legend(["Federated speed", "Base speed"])
    plt.savefig(f"{STATS_FOLDER}speed.png")


def getMeanDistances(fd, lcl):
    totalDistances = {"fd": [], "lcl": []}
    for i in range(len(fd)):
        totalDistances["fd"].append(mean(fd[i].raycast_distances))
        totalDistances["lcl"].append(mean(lcl[i].raycast_distances))
    return totalDistances


def getSomeDistances(fd, lcl):

    def getRays(arr):
        return [arr[6]]

    totalDistances = {"fd": [], "lcl": []}
    for i in range(len(fd)):
        totalDistances["fd"].append(sum(getRays(fd[i].raycast_distances)))
        totalDistances["lcl"].append(sum(getRays(lcl[i].raycast_distances)))
    return totalDistances


def mean(arr):
    return sum(arr) / len(arr)


def getRayGraphs():
    res = getMeanDistances(fd_data, lcl_data)

    print(f"Mean distance fd : {mean(res['fd'])} Mean distance lcl: {mean(res['lcl'])}")

    plt.figure()
    plt.plot(range(len(res["fd"])), res["fd"])
    plt.plot(range(len(res["lcl"])), res["lcl"])
    plt.legend(["Federated distance", "Base distance"])
    plt.savefig(f"{STATS_FOLDER}distances.png")


def countTooCloseForConfort(fd, lcl):
    TRESH = 5
    count = {"fd": 0, "lcl": 0}
    counts = {"fd": [0], "lcl": [0]}
    for i in range(len(fd)):
        addFd = False
        addLcl = False
        for j in range(len(fd[i].raycast_distances)):
            if fd[i].raycast_distances[j] <= TRESH:
                addFd = True
            if lcl[i].raycast_distances[j] <= TRESH:
                addLcl = True
        currLcl = counts["lcl"][-1]
        currFd = counts["fd"][-1]
        if addFd:
            counts["fd"].append(currFd + 1)
        else:
            counts["fd"].append(currFd)
        if addLcl:
            counts["lcl"].append(currLcl + 1)
        else:
            counts["lcl"].append(currLcl)
    return counts


def graphsTooClose():
    res = countTooCloseForConfort(fd_data, lcl_data)
    plt.figure()
    plt.plot(range(len(res["fd"])), res["fd"])
    plt.plot(range(len(res["lcl"])), res["lcl"])
    plt.legend(["Federated", "Basic"])
    plt.title("Too close for confort counts")
    plt.xlabel("Frames")
    plt.ylabel("Counts")
    plt.savefig(f"{STATS_FOLDER}tooClose.png")
    pass


getSpeedStats(fd_data, lcl_data)
getRayGraphs()
graphsTooClose()
