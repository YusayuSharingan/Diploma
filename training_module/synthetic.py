import os
import abc
import random
import numpy as np
import matplotlib.pyplot as plt
import warnings
from PIL import Image
from io import BytesIO

warnings.filterwarnings("error", category=RuntimeWarning)

class Graph(abc.ABC):

    types = {"Curve": 0, "Scatter": 1, "MultipleCurve": 2}
    graph_type: str

    max_density = 30
    noise_level = 50
    padding_x = 0.0
    padding_y = 0.0

    dot_size = 9

    page_size = (8.3, 11.7)

    class Bounds:
        def __init__(self, pw, ph, l, b):
            self.plot_width = pw
            self.plot_height = ph
            self.left = l
            self.bottom = b

    def __init__(self, multiple = False):
        self.noice = random.choice([0, self.noise_level/2, self.noise_level])
        self.multiple = multiple
        self.dependence = self.gen_dependence()
        self.gen_bboxes()

    def gen_dependence(self): pass


    def convert_data(self, fig, ax):
        dpi = fig.get_dpi()
        w = fig.get_size_inches()[0] * dpi
        h = fig.get_size_inches()[1] * dpi

        x_coords = self.dependence[:,0]
        y_coords = self.dependence[:,1]
        density = len(x_coords[0])

        flat_max_x, flat_min_x = np.argmax(x_coords), np.argmin(x_coords)
        flat_max_y, flat_min_y = np.argmax(y_coords), np.argmin(y_coords)

        max_x_point = (
            x_coords[flat_max_x // density][flat_max_x % density],
            y_coords[flat_max_x // density][flat_max_x % density]
        )
        min_x_point = (
            x_coords[flat_min_x // density][flat_min_x % density],
            y_coords[flat_min_x // density][flat_min_x % density]
        )
        max_y_point = (
            x_coords[flat_max_y // density][flat_max_y % density],
            y_coords[flat_max_y // density][flat_max_y % density]
        )
        min_y_point = (
            x_coords[flat_min_y // density][flat_min_y % density],
            y_coords[flat_min_y // density][flat_min_y % density]
        )

        fig.canvas.draw()
        pmax_x_point, pmin_x_point = ax.transData.transform(max_x_point), \
            ax.transData.transform(min_x_point)
        pmax_y_point, pmin_y_point = ax.transData.transform(max_y_point), \
            ax.transData.transform(min_y_point)

        scale_x, scale_y = \
            (pmax_x_point[0] - pmin_x_point[0]) / (max_x_point[0] - min_x_point[0]), \
            (pmax_y_point[1] - pmin_y_point[1]) / (max_y_point[1] - min_y_point[1])
        
        points = [
            (
                np.abs(((d[0] - min_x_point[0])*scale_x + pmin_x_point[0]) / w).tolist(),
                np.abs(((d[1] - min_y_point[1])*scale_y + pmin_y_point[1]) / (-h) + 1).tolist()
            ) for d in self.dependence]

        return (scale_x, scale_y), (min_x_point[0], min_y_point[1]), \
            points, (self.dot_size / w, self.dot_size/ h) 


    def gen_bboxes(self):
        plot_width, plot_height = \
            np.random.uniform(0.1, 0.6), \
            np.random.uniform(0.1, 0.6)
        left, bottom = \
            np.random.uniform(0.05, 1 - plot_width), \
            np.random.uniform(0.05, 0.95 - plot_height)
        self.bounds = self.Bounds(
            plot_width, 
            plot_height, 
            left, bottom
        )
        self.bboxes = map(str, [
            self.bounds.left + self.bounds.plot_width / 2,
            1 - (self.bounds.bottom + self.bounds.plot_height / 2),
            self.bounds.plot_width + self.padding_x,
            self.bounds.plot_height + self.padding_y
        ])
    

    def add_image_noise(self, fig):
        buf = BytesIO()
        fig.savefig(buf, format='jpg')
        buf.seek(0)

        img = Image.open(buf).convert('RGB')
        img_np = np.array(img).astype(np.float32)

        mask = np.random.normal(0, self.noice, img_np.shape)
        noisy_img = np.clip(img_np + mask, 0, 255).astype(np.uint8)

        return Image.fromarray(noisy_img)


    def plot_graph(self, title):
        bbounds_fig = plt.figure(figsize=self.page_size)
        bbounds_ax = bbounds_fig.add_axes([
            self.bounds.left,
            self.bounds.bottom,
            self.bounds.plot_width,
            self.bounds.plot_height
        ])

        points_fig, points_ax = plt.subplots(figsize=(6.4, 6.4))

        if isinstance(self, Scatter):
            scatter = self.dependence[0]
            bbounds_ax.scatter(*scatter)
            points_ax.scatter(*scatter)
        else:
            for curve in self.dependence:
                color = (random.random(), random.random(), random.random())
                marker = random.choice([None, None, None, "o", "^", "s", "x", "D", "*"])
                style  = random.choice([None, None, None, "-", "--", "-.", ':'])
                bbounds_ax.plot(*curve, linestyle=style, marker=marker, color=color)
                points_ax.plot(*curve, linestyle=style, marker=marker, color=color)

        bbounds_ax.set_title(title)
        points_ax.grid(random.getrandbits(1))
        bbounds_ax.set_xlabel("X")
        bbounds_ax.set_ylabel("Y")

        bbounds_ax.grid(random.getrandbits(1))

        return bbounds_fig, bbounds_ax, points_fig, points_ax


    def save_data(self, path, idx):
        make_repo(path)
        title = f"{self.graph_type}_{idx}"

        bbounds_fig, bbounds_ax, points_fig, points_ax = self.plot_graph(title)

        try:
            # scale, pads, _, _ = self.convert_data(bbounds_fig, bbounds_ax)  # TODO: union these operations -
            _, _, points, dsize = self.convert_data(points_fig, points_ax)  # just crop bbounds_fig to points_fig
        except RuntimeWarning:
            plt.close(bbounds_fig)
            plt.close(points_fig)
            plt.close('all')
            return

        noisy_bbounds_img = self.add_image_noise(bbounds_fig)
        noisy_points_img = self.add_image_noise(points_fig)

        noisy_bbounds_img.save(f"{path[0]}/detect_bounds/images/{path[1]}/{title}.jpg", format="JPEG")
        noisy_points_img.save(f"{path[0]}/detect_points/images/{path[1]}/{title}.jpg", format="JPEG")
        plt.close(bbounds_fig)
        plt.close(points_fig)
        plt.close('all')

        with open(f"{path[0]}/detect_bounds/labels/{path[1]}/{title}.txt", 'w') as f:
            f.write(f"{self.types[self.graph_type]} {' '.join(self.bboxes)}")

        with open(f"{path[0]}/detect_points/labels/{path[1]}/{title}.txt", 'w') as f:
            for idx in range(len(points)):
                x, y = points[idx]
                for point in zip(x, y):
                    f.write(f"{idx} {point[0]} {point[1]} {dsize[0]} {dsize[1]}\n")
        
        # with open(f"{path[0]}/detect_points/scales/{path[1]}/{title}.txt", 'w') as f:
        #     f.write(f"{pads[0]} {pads[1]} {scale[0]} {scale[1]}\n")




class Curve(Graph):
    def __init__(self, multiple = False):
        super().__init__(multiple)
        self.graph_type = "MultipleCurve" if self.multiple else "Curve"

    def __gen_poly(self, degree=5): 
        return np.polynomial.Polynomial(
            # a0*x^0 .. an*x^n, n == degree
            np.random.rand(degree)   
        )
    

    def get_points(self, bounds, density = None):
        # ось абсцисс
        if density == None:
            density = random.randint(10, self.max_density)

        x = np.linspace(*bounds, density)

        # места склейки сплайнов
        n_gaps = random.randint(0, 10)
        gaps = np.zeros(n_gaps+2, dtype=int)
        
        for i in range(1, n_gaps + 1):
            gaps[i] = random.randint(gaps[i-1], density + i - n_gaps)
        gaps[-1] = density

        # ось ординат
        y = np.zeros(density)
        prev_gap = 0
        for gap in gaps:
            p = self.__gen_poly()
            y[prev_gap:gap] = p(x[prev_gap:gap])
            prev_gap = gap

        return x, y
    

    def gen_dependence(self):
        num_of_curves = random.randint(2, 5) if self.multiple else 1
        lbound, rbound = np.sort(
                np.random.randint(-100, 100, size = (1, 2)),
                axis=None
            )
        density = random.randint(10, self.max_density)

        return np.array([self.get_points((lbound, rbound), density) for _ in range(num_of_curves)])



class Scatter(Graph):
    graph_type = "Scatter"

    def get_points(self, bounds, density):
        x = np.random.uniform(bounds[0], bounds[1], density)
        y = np.random.uniform(bounds[2], bounds[3], density)
        return x, y
    
    def gen_dependence(self):
        bounds = np.sort(
            np.random.randint(-100, 100, size = (1, 4)),
            axis=None
        )
        density = random.randint(10, self.max_density)
        return np.array([self.get_points(bounds, density)])


def make_repo(path):
    os.makedirs(f"{path[0]}/detect_bounds/images/{path[1]}", exist_ok=True)
    os.makedirs(f"{path[0]}/detect_bounds/labels/{path[1]}", exist_ok=True)

    os.makedirs(f"{path[0]}/detect_points/images/{path[1]}", exist_ok=True)
    os.makedirs(f"{path[0]}/detect_points/labels/{path[1]}", exist_ok=True)
    os.makedirs(f"{path[0]}/detect_points/scales/{path[1]}", exist_ok=True)



if __name__ == "__main__":
    graph = Curve()
    graph.save_data(("test_ds", "test"),0)

    graph2 = Scatter()
    graph2.save_data(("test_ds", "test"), 0)

    graph3 = Curve(multiple=True)
    graph3.save_data(("test_ds", "test"), 0)