import matplotlib.pyplot as plt
import json
import cv2
import numpy as np

COLORS = [
    "red", 
    "green", 
    "blue",
    "yellow",
    "orange"
]

def visualize(file_path: str):
    """Visualize the results by plotting poses on the test images"""
    results = None
    
    with open(file_path, "r") as f:
        results = json.load(f)
    
    # results = results[0]
    for k in results:
        img_path = k["image_paths"][0]
        all_people = np.array(k['preds'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        colors = iter(COLORS)
        print(img_path, all_people.shape[0])
        for idx in range(all_people.shape[0]):
            kps = np.array(all_people[idx, :, :])       
            kps_x = kps[:, 0]
            kps_y = kps[:, 1]
            assert kps_x.shape==(18,)
            assert kps_y.shape==(18,)
            color = next(colors)
            plt.scatter(kps_x, kps_y, color=color)
        plt.savefig(f"vis_res/{img_path.split('/')[-1]}", dpi=180)
        plt.close()

if __name__ == "__main__":
    
    RESULTS_FILE = "higherhrnet_penalty_results.json"
    visualize(RESULTS_FILE)