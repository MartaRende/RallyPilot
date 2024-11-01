import numpy
import pickle
import lzma
import matplotlib.pyplot as plt

fd_data = pickle.load(lzma.open("./stats/federated_model.npz", "rb"))
lcl_data = pickle.load(lzma.open("./stats/local_model.npz", "rb"))

speed_fd_data= []
speed_lcl_data = []
mean_speed_fd = 0 
mean_speed_lcl = 0 
for i in range(0,len(fd_data)):
    speed_fd_data.append(fd_data[i].car_speed)
    speed_lcl_data.append(lcl_data[i].car_speed)
    mean_speed_fd += fd_data[i].car_speed
    mean_speed_lcl+=lcl_data[i].car_speed
mean_speed_fd = mean_speed_fd/len(fd_data)
mean_speed_lcl = mean_speed_lcl/len(lcl_data)
print(f"Mean speed with fd : {mean_speed_fd}")
print(f"Mean speed with lcl : {mean_speed_lcl}")
plt.figure()
plt.plot(range(0,len(speed_fd_data)),speed_fd_data)
plt.savefig(f"speed_fd.png")
plt.figure()
plt.plot(range(0,len(speed_lcl_data)),speed_lcl_data)
plt.savefig(f"speed_lcl.png")
