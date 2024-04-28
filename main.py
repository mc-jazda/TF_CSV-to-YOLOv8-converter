import pandas as pd

def csv_to_yolo_format(src_path, dest_path):
    data = pd.read_csv(src_path, header=0, names=('width', 'height', 'roi.x1', 'roi.y1', 'roi.x2', 'roi.y2', 'class_id', 'path'))
    
    for i in range(data.shape[0]):
        # fetching data
        width = data.loc[i, 'width']
        height = data.loc[i, 'height']
        roi_x1 = data.loc[i, 'roi.x1']
        roi_y1 = data.loc[i, 'roi.y1']
        roi_x2 = data.loc[i, 'roi.x2']
        roi_y2 = data.loc[i, 'roi.y2']
        path = data.loc[i, 'path'].split('/')[1].split('.')[0]
        class_id = data.loc[i, 'class_id'] - 1

        # normalized xywh
        x_center = ((roi_x2 + roi_x1) / 2) / width
        y_center = ((roi_y2 + roi_y1) / 2) / height
        box_width = (roi_x2 - roi_x1) / width
        box_height = (roi_y2 - roi_y1) / height
        
        with open(f'{dest_path}/{path}.txt', 'a') as file:
            file.write(f'{class_id} {x_center} {y_center} {box_width} {box_height}\n')


csv_to_yolo_format('train_frames.csv', 'labels/train_labels')