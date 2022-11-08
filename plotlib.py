from matplotlib import pyplot as plt

def plot_my_data(my_data):
    """Plot one dimensional data.
    
    Parameters
    ----------
    my_data : array like
        n x 0 array
    """

    fig, ax = plt.plot(my_data)
    return ax