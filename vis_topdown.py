from os import makedirs
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json
import cv2
import numpy as np
from os.path import join
COLORS = [
    "red", 
    "green", 
    "blue",
    "orange"
]

def visualize(file_path: str):
    """Visualize the results by plotting poses on the test images"""
    results = None
    
    with open(file_path, "r") as f:
        results = json.load(f)
    
    res_dir = file_path.split("_results")[0]
    makedirs(res_dir, exist_ok=True)
    vis_store = {}
    for k in results:
        
        curr_img = k["image_paths"]
        kpt_dict = {}
        kps = np.array(k["preds"]).squeeze()
        
        kps_refined = np.zeros((18, 2))
        kps_refined = kps[:, :2]
        
        kps_x = kps_refined[:, 0]
        kps_y = kps_refined[:, 1]
        
        kpt_dict["x"] = kps_x
        kpt_dict["y"] = kps_y
        assert len(k['image_paths'])==1
        if k["image_paths"][0] not in vis_store.keys():
            vis_store[k["image_paths"][0]] = []
            vis_store[k["image_paths"][0]].append(kpt_dict)
        else:
            vis_store[k["image_paths"][0]].append(kpt_dict)
    
    input_dir = f"data/coco/{res_dir}"
    annnot_json = f"/home/devin/mcgill/pose/mmpose/data/coco/annotations/{res_dir.split('_')[0]+'_'+res_dir.split('_')[1]}.json"

    with open(annnot_json, "r") as f:
        annot = json.load(f)

    imgid2name = {}
    for i in annot['images']:
        name = int(i['file_name'].split("_")[1].split(".")[0])-1
        imgid2name[i['id']] = f"{name}.png"

    collector = {}
    for i in annot['annotations']:
        path = join(input_dir, imgid2name[i['image_id']])

        if path in collector.keys():
            collector[path].append(i['bbox'])
        else:
            collector[path] = [i['bbox']]
        
    fig = plt.figure()
    for idx, k in enumerate(vis_store.keys()):
        ax = fig.add_subplot(111)
        img = cv2.imread(k)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        
        # draw keypoints
        colors = iter(COLORS)
        for kpt_dict in (vis_store[k]):
            kps_x = kpt_dict["x"]
            kps_y = kpt_dict["y"]
            color = next(colors)
            # if color=="orange":
            #     continue
            ax.scatter(kps_x, kps_y, color=color)
        
        # draw bboxes
        colors = iter(COLORS)
        for i in collector[list(collector.keys())[idx]]:
            x, y, w, h = i
            ax.add_patch(Rectangle([x, y], height=h, width=w, fc = 'none', ec = next(colors), lw=3))

        plt.savefig(f"{res_dir}/{idx}.png", dpi=600)
        plt.clf()
        plt.cla()

if __name__ == "__main__":
    
    RESULTS_FILE = "2017-11-08-bos-nyr-national_020132_vipnas_results.json"
    visualize(RESULTS_FILE)