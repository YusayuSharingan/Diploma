from ultralytics import YOLO
from torch.cuda import is_available
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import json
import os

class BoundSearcher:
    def __init__(self, weights):
        device = "cuda" if is_available() else "cpu"
        self.__model = YOLO(weights).to(device)
    

    def search(self, path, idx):
        results = self.__model(path)
        results[0].save(f"output/{path[:-4]}/intermediate/graph_bounds_{idx}.jpg")
        bboxes = results[0].boxes
        images = []

        for box in bboxes:
            cls = int(box.cls[0].item())
            img = Image.open(path)
            img = img.crop(box.xyxy[0].tolist())
            images.append((cls, np.array(img)))
        return images
    

class PointSearcher:
    def __init__(self, weights):
        device = "cuda" if is_available() else "cpu"
        self.__model = YOLO(weights).to(device)


    def __get_center(self, x1, y1, x2, y2):
        return (x1 + x2) / 2, (y1 + y2) / 2


    def __scaling(self, point, pads, scale, sizes):
       return point[0]/scale[0] + pads[0], (sizes[0] - point[1])/scale[1] + pads[1]


    def search(self, img, path, idx):
        results = self.__model(img)
        results[0].save(f"output/{path[:-4]}/intermediate/points_{idx}.jpg")
        pboxes = results[0].boxes

        data = {}
        for box in pboxes:
            cls = int(box.cls[0].item())
            if cls not in data: data[cls] = []
            point = self.__scaling(
                self.__get_center(*box.xyxy[0].tolist()),
                (-48, 0), (9.8, 0.00041),
                img.shape[:-1]
            )
            data[cls].append(point)

        for value in data.values():
            value.sort()
    
        return data



def approx_func(points):
    if len(points) == 1: return points[0]

    x = list(map(lambda point: point[0], points))
    y = list(map(lambda point: point[1], points))

    cs = CubicSpline(x, y, bc_type='natural')

    xnew = np.linspace(x[0], x[-1], 100)
    ynew = cs(xnew)

    return xnew, ynew



def extract_points(path):
    bsearcher = BoundSearcher("weights/yolo_for_bounds/best.pt")
    psearcher = PointSearcher("weights/yolo_for_points/best.pt")

    os.makedirs(f"output/{path[:-4]}/intermediate", exist_ok=True)

    graphs = bsearcher.search(path, 0)
    graph_types = {0: "Curve", 1: "Scatter", 2: "MultipleCurve"}

    for idx, (gtype, box) in enumerate(graphs):
        data = psearcher.search(box, path, idx)
        with open(f"output/{path[:-4]}/points_{idx}.json", 'w') as f:
            json.dump(data, f)
        
        if gtype != 1:
            fig, axes = plt.subplots(len(data) // 3 + 2, 3, figsize=(12, 8))
            for ncurve, curve in enumerate(data.values()):
                x, y = approx_func(curve)

                axes[ncurve // 3][ncurve % 3].plot(x, y)
                axes[ncurve // 3][ncurve % 3].set_title(f"{graph_types[gtype]} Curve #{ncurve+1}")
            
            fig.savefig(f"output/{path[:-4]}/points_{idx}.jpg", format="jpg")
            


if __name__ == "__main__":
    extract_points(input())














