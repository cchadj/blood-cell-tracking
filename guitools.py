import cv2
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.colors


def rgba(r, g, b, a=1):
    return r / 255, g / 255, b / 255, a


def add_point(scat, new_point, c='k'):
    old_off = scat.get_offsets()
    new_off = np.concatenate([old_off, np.array(new_point, ndmin=2)])

    old_c = scat.get_facecolors()
    new_c = np.concatenate([old_c, np.array(matplotlib.colors.to_rgba(c), ndmin=2)])

    scat.set_offsets(new_off)
    scat.set_facecolors(new_c)


class MplScatterPlotPointSelector(object):
    def __init__(self, points, fig_ax=None, image=None,
                 selected_color=rgba(0, 130, 0, 0.6),
                 added_point_selected_color=rgba(0, 240, 0, 0.6),
                 unselected_color=rgba(20, 20, 230, 0.45),
                 added_point_unselected_color=rgba(0, 255, 255, 0.45)):
        self.fig_ax = fig_ax

        if self.fig_ax is None:
            plt.ioff()
            self.fig_ax = plt.subplots()

        self.fig, self.ax = self.fig_ax
        self.image = image
        self.selected_point_indices = []

        # ids for the point picker and point click on scatter plot.
        self.pick_point_cid = None
        self.add_point_cid = None
        self.scat = None

        self.is_activated = False
        self.initial_points = points
        self.selected_color = selected_color
        self.unselected_color = unselected_color
        self.added_point_unselected_color = added_point_unselected_color
        self.added_point_selected_color = added_point_selected_color

        self.point_picked = False
        self.added_points = np.zeros_like(points, shape=(0, 2))
        self.all_points = self.initial_points.copy()
        # self.fig.canvas.set_window_title(
        #     'Left click to pick, right remove.\n'
        #     'Close window to finish.')

    def activate(self):
        if self.is_activated:
            raise Exception('Make sure that PointSelector is deactivated before activating')
        self.pick_point_cid = self.fig.canvas.mpl_connect('pick_event', self.on_point_pick)
        self.add_point_cid = self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        self.scat = self.ax.scatter(self.initial_points[:, 0],
                                    self.initial_points[:, 1],
                                    s=100,
                                    c=[self.unselected_color] * len(self.initial_points),
                                    picker=True)
        self._reinit_selected_points()
        self.is_activated = True

    def deactivate(self):
        if not self.is_activated:
            raise Exception('Make sure that PointSelector is activated before deactivating')

        # Disconnect all listeners or they will continue to listen when we change frame.
        # disconnect point pick listener
        self.fig.canvas.mpl_disconnect(self.pick_point_cid)
        self.pick_point_cid = None

        # disconnect add point listener
        self.fig.canvas.mpl_disconnect(self.add_point_cid)
        self.add_point_cid = None

        # remove the scatter plot from the axis
        self.scat.remove()
        self.scat = None
        self.is_activated = False

    def _reinit_selected_points(self):
        """ Give selected_color to selected points when going from deactivated state to activated state """
        try:
            for ind in self.selected_point_indices:
                # noinspection PyProtectedMember
                self.scat._facecolors[ind, :] = self.selected_color
                print(ind)
        except ValueError:
            print('Make sure that self.col scatter plot is created before changing the face colors.')
        self.fig.canvas.draw_idle()

    def on_mouse_click(self, event):
        LEFT_CLICK = 1
        if self.point_picked:
            # Don't add point if a point is picked
            self.point_picked = False
            return

        if event.button == LEFT_CLICK:
            if event.xdata is None or event.ydata is None:
                return

            coordinate = np.array([event.xdata, event.ydata])[np.newaxis, ...].round().astype(np.int32)

            self.added_points = np.concatenate((self.added_points, coordinate), axis=0)
            self.all_points = np.concatenate((self.all_points, coordinate), axis=0)
            added_point_ind = len(self.all_points) - 1
            self.selected_point_indices.append(added_point_ind)

            add_point(self.scat, coordinate, c=self.added_point_selected_color)

        self.fig.canvas.draw_idle()

    def on_point_pick(self, event):
        self.point_picked = True
        LEFT_CLICK = 1
        RIGHT_CLICK = 3
        ind = event.ind
        # event.ind returns a list.
        # even when multiple indices select the first
        ind = ind[0]

        if event.mouseevent.button == RIGHT_CLICK:
            try:
                self.selected_point_indices.remove(ind)

                is_manually_added_point = ind >= len(self.initial_points)
                if is_manually_added_point:
                    # noinspection PyProtectedMember
                    self.scat._facecolors[ind, :] = self.added_point_unselected_color
                else:
                    # noinspection PyProtectedMember
                    self.scat._facecolors[ind, :] = self.unselected_color

            except ValueError:
                pass
        elif event.mouseevent.button == LEFT_CLICK:
            self.selected_point_indices.append(ind)

            is_manually_added_point = ind >= len(self.initial_points)
            if is_manually_added_point:
                # noinspection PyProtectedMember
                self.scat._facecolors[ind, :] = self.added_point_selected_color
            else:
                # noinspection PyProtectedMember
                self.scat._facecolors[ind, :] = self.selected_color

        self.fig.canvas.draw_idle()


def scatter_plot_point_selector(points, ax=None, image=None):
    selected_point_indices = []

    if ax is None:
        _, ax = plt.subplots()

    fig = plt.gcf()
    if image is not None:
        ax.imshow(image, cmap='gray')
    ax.set_title('Select points to keep.')

    point_selector = MplScatterPlotPointSelector(points, fig_ax=(fig, ax))
    point_selector.activate()
    plt.show()
    return point_selector.selected_point_indices


class CvGuiSelector(object):
    def __init__(self, window_name, image):
        self.window_name = window_name
        self.image = np.uint8(image).copy()
        self.modified_image = np.uint8(image).copy()
        if len(self.modified_image.shape) == 2:
            self.modified_image = self.modified_image[..., np.newaxis]
            self.modified_image = np.concatenate((self.modified_image, self.modified_image, self.modified_image), axis=-1)
        self.empty_image = self.modified_image.copy()
        self.points_selected = []

    def point_select(self, event, x, y, flags, param):
        pass

    def activate(self):
        ESC = 25
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.point_select)

        # keep looping until the 'q' key is pressed
        while True:
            # display the image and wait for a keypress
            cv2.imshow(self.window_name, self.modified_image)
            key = cv2.waitKey(1) & 0xFF
            # if the 'r' key is pressed, reset the cropping region
            if key == ord("r"):
                self.modified_image = self.empty_image.copy()
                self.points_selected = []
            # if the 'c' key is pressed, break from the loop
            elif key in [ord('c'), ord('q'), ord('\n'), ord('\r'), ESC]:
                cv2.destroyAllWindows()
                break


class CvPointSelector(CvGuiSelector):
    def __init__(self, window_name, image, point_thickness):
        super(CvPointSelector, self).__init__(window_name, image)
        self.points_thickness = point_thickness

    def point_select(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt = [(x, y)]
            self.points_selected.append((x, y))
            # modified_image = self.empty_image.copy()
            # # for (x, y) in self.points_selected:
            # #     modified_image = cv2.circle(modified_image,  (y, x), radius=0, color=(255, 0, 0), thickness=self.points_thickness)
            # self.modified_image = modified_image


class CvRoipolySelector(CvPointSelector):
    def __init__(self, window_name, image, point_thickness=5000):
        super().__init__(window_name, image, point_thickness)
        self.points_thickness = point_thickness
        self._mask = np.zeros(self.image.shape[:2], dtype=np.uint8)

    def point_select(self, event, x, y, flags, param):
        super().point_select(event, x, y, flags, param)
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt = [(x, y)]
            if len(self.points_selected) < 3:
                self.modified_image = cv2.polylines(self.empty_image.copy(), np.int32([self.points_selected]), color=(0, 0, 255), isClosed=False)
            else:
                self.modified_image = cv2.polylines(self.empty_image.copy(), np.int32([self.points_selected]), color=(0, 0, 255), isClosed=True)

    @property
    def mask(self):
        if len(self.points_selected) > 2:
            self._mask = cv2.fillPoly(self._mask, np.int32([self.points_selected]), color=(1, 1, 1))
        return self._mask.astype(np.bool8)

    @mask.setter
    def mask(self, value):
        self._mask = value


if __name__ == '__main__':
    points = np.random.randint(0, 10, size=(20, 2))
    points_selected_indices = scatter_plot_point_selector(points)

    print(points_selected_indices)
