import matplotlib.pyplot as plt
from utilities import FileReader
import numpy as np

type = "spiral"


def plot_errors(filename, Q, R):
    
    headers, values=FileReader(filename).read_file()
    
    time_list=[]
    
    first_stamp=values[0][-1]
    
    for val in values:
        time_list.append(val[-1] - first_stamp)

    fig, axes = plt.subplots(2,1, figsize=(14,6))

    print(headers[len(headers) - 3])
    print(headers[len(headers) - 2])

    axes[0].plot([lin[len(headers) - 3] for lin in values], [lin[len(headers) - 2] for lin in values], label="EKF Prediction")
    axes[0].set_title(f"State Space - Point Trajectory - Q = {Q}, R = {R}")
    axes[0].grid()

    if type == "spiral":

        spiralx = [0]
        spiraly = [0]
        v_sim = 0
        th_sim = 0
        for i in range(2, len(values)):
            dt = values[i][-1] - values[i-1][-1]
            v_sim += 0.1 * dt
            th_sim += 1 * dt
            spiralx.append(v_sim*np.cos(th_sim))
            spiraly.append(v_sim*np.sin(th_sim))

        axes[0].plot(spiralx, spiraly, label="Desired Path")
        axes[0].legend()

    elif type == "point":
        headers2, values2=FileReader("CSVs/robot-pose-point-LAB2.csv").read_file()
        time_list2 = []
        first_stamp2=values2[0][-1]
    
        for val in values2:
            time_list2.append(val[-1] - first_stamp2)

        axes[0].plot([lin[0] for lin in values2], [lin[1] for lin in values2], label="Path From Lab 2")
        axes[0].legend()
    
    axes[1].set_title(f"Each Individual State - Point Trajectory - Q = {Q}, R = {R}")
    for i in range(0, len(headers) - 1):
        axes[1].plot(time_list, [lin[i] for lin in values], label= headers[i])

    axes[1].legend()
    axes[1].grid()

    plt.show()

Q = 0.5
R = 0.5
    
plot_errors(f"CSVs/robot_pose-V2-{type}-Q0{int(Q*10)}-R0{int(R*10)}.csv", Q, R)


