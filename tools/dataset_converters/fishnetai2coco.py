import os
import numpy as np
import fiftyone as fo
from glob import glob
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd

csv_data = 'D:/PROJECT/ICU/PYTHON/FishMoonProject/Image Recognition/Data/Mini FishNet Datasets/foid_labels_v100.csv'
image_data = 'D:/PROJECT/ICU\PYTHON/FishMoonProject/Image Recognition/Data/Mini FishNet Datasets/images'
type_img = '.jpg'

df = pd.read_csv(csv_data)
df_train = df[df['train'] == True]
df_val = df[df['val'] == True]
df_test = df[df['test'] == True]


def bbox_convert_to_fo_format(filepath: str, bbox):
    image = cv2.imread(filepath)
    shape = image.shape
    if len(shape) == 3:
        h, w, c = shape
    elif len(shape) == 2:
        h, w = shape
    else:
        raise ValueError("Wrong Dimension Data")

    x_max = bbox[1]
    y_max = bbox[3]
    x_min = bbox[0]
    y_min = bbox[2]

    # pt1 = (x_min,y_min)
    # pt2 = (x_max,y_max)

    width = abs(x_max - x_min)
    height = abs(y_max - y_min)

    # Bounding box coordinates should be relative values
    # in [0, 1] in the following format:
    # [top-left-x, top-left-y, width, height]
    bbox = [x_min / w, y_min / h, width / w, height / h]

    return bbox


def convert_fishnet_to_coco(df, src_img=image_data, img_type=type_img):
    name_datasets = 'fishnet-img'

    # Create dataset
    try:
        dataset = fo.Dataset(name=name_datasets)
    except:
        dataset = fo.load_dataset(name=name_datasets)

    # Persist the dataset on disk in order to
    # be able to load it in one line in the future
    dataset.persistent = True
    labels = []

    img_id_uniques = df['img_id'].unique()
    for idx, img_id in enumerate(img_id_uniques):
        df_bbox = df[df['img_id'] == img_id][['x_min', 'x_max', 'y_min', 'y_max', 'label_l2']]
        # print(df_bbox)

        filename_img = f"{img_id}{img_type}"
        filepath_img = os.path.join(src_img, filename_img)

        sample = fo.Sample(filepath=filepath_img)
        detections = []

        print(f'processing {idx + 1}/{len(img_id_uniques)} {filename_img}')

        for row in df_bbox.iterrows():
            label = row[1]['label_l2']
            labels.append(label)

            x_min = row[1]['x_min']
            x_max = row[1]['x_max']
            y_min = row[1]['y_min']
            y_max = row[1]['y_max']

            bbox = (x_min, x_max, y_min, y_max)

            bbox_fo = bbox_convert_to_fo_format(filepath_img, bbox)

            # Convert detections to FiftyOne format
            detections.append(fo.Detection(label=label, bounding_box=bbox_fo))

        # Store detections in a field name of your choice
        sample["ground_truth"] = fo.Detections(detections=detections)

        dataset.add_sample(sample)
    return dataset, labels


def export_datasets(datasets, export_dir, label_field='ground_truth'):
    # Export the dataset
    datasets.export(
        export_dir=export_dir,
        dataset_type=fo.types.COCODetectionDataset,
        label_field=label_field,
    )
    print(f"Succes create data to {export_dir}")


def main(out_data):

    train_datasets, train_labels = convert_fishnet_to_coco(df_train)
    print(f"train labels {train_labels}")
    val_datasets, val_labels = convert_fishnet_to_coco(df_val)
    print(f"val labels {val_labels}")
    test_datasets, test_labels = convert_fishnet_to_coco(df_test)
    print(f"test labels {test_labels}")

    # Exporting Data
    export_datasets(train_datasets, os.path.join(out_data, 'train'))
    export_datasets(val_datasets, os.path.join(out_data, 'val'))
    export_datasets(test_datasets, os.path.join(out_data, 'test'))

    return train_labels, val_labels, test_labels


if __name__ == '__main__':
    OUT_DATA = "./data/coco-fishnet"
    main(OUT_DATA)