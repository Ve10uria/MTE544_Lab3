import matplotlib.pyplot as plt
from utilities import FileReader
import numpy as np




def plot_errors(filename):
    
    headers, values=FileReader(filename).read_file()
    
    time_list=[]
    
    first_stamp=values[0][-1]
    
    for val in values:
        time_list.append(val[-1] - first_stamp)

    
    
    fig, axes = plt.subplots(2,1, figsize=(14,6))

    print(headers[len(headers) - 3])
    print(headers[len(headers) - 2])


    axes[0].plot([lin[len(headers) - 3] for lin in values], [lin[len(headers) - 2] for lin in values], label="EKF Prediction")
    axes[0].set_title("state space")
    axes[0].grid()

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

    
    axes[1].set_title("each individual state")
    for i in range(0, len(headers) - 1):
        axes[1].plot(time_list, [lin[i] for lin in values], label= headers[i])

    axes[1].legend()
    axes[1].grid()

    plt.show()
    
    





import argparse

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Process some files.')
    parser.add_argument('--files', nargs='+', required=True, help='List of files to process')
    
    args = parser.parse_args()
    
    print("plotting the files", args.files)

    filenames=args.files
    for filename in filenames:
        plot_errors(filename)


