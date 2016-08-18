import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plotter import ThumbnailPlotter, get_factory

class VideoPlotter(ThumbnailPlotter):

    def __init__(self, filename=None, name=None, cmap='Greys', max_num_frame=9,
                 max_num_col=9):
        super(VideoPlotter, self).__init__(
            filename=filename, name=name, cmap=cmap, max_num_col=max_num_col)
        self.max_num_frame = max_num_frame
        pass

    def plot(self, img, axis=1):
        num_ex = img.shape[0]
        num_items = min(img.shape[axis], self.max_num_frame)
        if img.shape[axis] > self.max_num_frame:
            if axis == 3:
                img = img[:, :, :, :self.max_num_frame]
            elif axis == 1:
                img = img[:, :self.max_num_frame]
            else:
                raise Exception('Axis not supported')
            pass

        num_row, num_col, calc = self.calc_row_col(num_ex, num_items)

        f1, axarr = plt.subplots(num_row, num_col, figsize=(10, num_row))
        self.set_axis_off(axarr, num_row, num_col)

        for ii in xrange(num_ex):
            for jj in xrange(num_items):
                row, col = calc(ii, jj)
                if axis == 3:
                    x = img[ii, :, :, jj]
                elif axis == 1:
                    x = img[ii, jj]
                else:
                    raise Exception('Axis not supported')
                if num_col > 1 and num_row > 1:
                    ax = axarr[row, col]
                elif num_row > 1:
                    ax = axarr[row]
                elif num_col > 1:
                    ax = axarr[col]
                if x.shape[-1] == 3 and len(x.shape) == 3:
                    x = x[:, :, [2, 1, 0]]
                if x.shape[-1] == 1 and len(x.shape) == 3:
                    x = x.reshape(x.shape[0], x.shape[1])
                ax.imshow(x, cmap=self.cmap)
                ax.text(0, -0.5, '[{:.2g}, {:.2g}]'.format(x.min(), x.max()),
                        color=(0, 0, 0), size=8)

        plt.tight_layout(pad=1.0, w_pad=0.0, h_pad=0.0)
        plt.savefig(self.filename, dpi=150)
        plt.close('all')
        self.register()
        pass

    def listen(self, results):
        """Plot results.

        Args:
            images: [B, T, H, W] or [B, T, H, W, 3] or [B, H, W, T]
        """
        axis = 1
        img = results['images']
        # print self.name, self.filename, img.shape
        self.plot(img, axis=axis)
        pass
    pass

get_factory().register('video', VideoPlotter)
