import root_path
from tools.data.label_studio import rename_images_in_coco_json_file
from tools.data.cut_images_from_bbox import get_cut_images_from_gt_bboxes



if __name__ == "__main__":
    # rename_images_in_coco_json_file("dataset/spine_fracture/xray/annotations/result.json")
    get_cut_images_from_gt_bboxes("dataset/spine_fracture/xray/images",
                                  "dataset/spine_fracture/xray/annotations/fracture_normal.json",
                                  "dataset/spine_fracture/cut_xray")