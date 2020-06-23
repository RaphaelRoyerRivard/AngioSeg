from matplotlib import pyplot as plt
import os
import cv2


def show_image_and_capture_click(image):
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    ax = plt.gca()
    fig = plt.gcf()
    implot = ax.imshow(image)
    click_location = None

    def onclick(event):
        if event.xdata != None and event.ydata != None:
            nonlocal click_location
            click_location = (int(round(event.xdata)), int(round(event.ydata)))
            plt.close(fig)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()

    return click_location


if __name__ == '__main__':
    skeletons_folder = r"C:\Users\Raphael\Pictures\skeleton"
    for path, subfolders, files in os.walk(skeletons_folder):
        folder = path.split("/")[-1].split("\\")[-1]
        file = f"{folder}.tif"
        if file in files:
            image_path = path + "\\" + file
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            ostium = show_image_and_capture_click(image)
            f = open(f"{path}\\{folder}_ostium.txt", 'w')
            f.write(str(ostium[0]) + " " + str(ostium[1]))
            f.close()
