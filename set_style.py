import matplotlib
import matplotlib.pyplot as plt

def set_style():
    plt.style.use(['seaborn-white', 'seaborn-paper'])
    matplotlib.rc("font", family="Times New Roman", size= 16)
    matplotlib.rc('ytick', labelsize=14)
    matplotlib.rc('xtick', labelsize=14)
    matplotlib.rc('lines', lw = 2)
    params = {'legend.fontsize': 'small',
         'axes.labelsize': 'small',
         'axes.titlesize':'small',}
    matplotlib.rcParams.update(params)