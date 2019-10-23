import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def plot_field(coord,val,dim):

    if dim == 1:
        fig = plt.figure()
        plt.plot(coord[0], val)


    if dim == 2:
        fig = plt.figure()
        ax = fig.gca(projection = '3d')
        # Plot the surface.
        surf = ax.plot_surface(coord[0], coord[1], val, cmap = cm.coolwarm,
                               linewidth = 0.2, antialiased = False, edgecolors='black')
        ax.set_xlabel('x[0] direction')
        ax.set_ylabel('x[1] direction')

    if dim == 3:
        multi_slice_viewer(val[:,:,:]) # x direction is stable,
        print('Plot 3D: switch slice in x-direction by j,k key')
    return

def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)

def multi_slice_viewer(volume):
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[0]//2
    ax.imshow(volume[ax.index])
    fig.canvas.mpl_connect('key_press_event', process_key)

def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    fig.canvas.draw()

def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1)%volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])

def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1)%volume.shape[0]
    ax.images[0].set_array(volume[ax.index])


    # TODO : add evaluation of function in coords
    # TODO : add gradient in n-D