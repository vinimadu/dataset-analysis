
import fiftyone as fo
from fiftyone import ViewField as F
import matplotlib.pyplot as plt
import math
from fiftyone.core.expressions import ViewExpression
import pandas as pd 
import fiftyone as fo
import tkinter as tk
from tkinter import messagebox
import requests
import os
from tqdm import tqdm # Para a barra de progresso
from zipfile import ZipFile
import gdown  # Para downloads do Google Drive
import tarfile
import json
import xml.etree.ElementTree as ET
import cv2
import shutil
import numpy as np
from PIL import Image

#%% CONSTANTS

datasets_dir = '../../FOdatasets/'

with open('datasets_info.json','r') as f:
    datasets_info = json.load(f)

# Bboxes are in [top-left-x, top-left-y, width, height] format
#%% Analysis expressions
analysis_expressions = {
    'count': {
        '# total samples': None,
        '# samples with signs': 'detections',
        '# total objects': 'detections.detections',
    },
    'statistical': {
        'images-width': F("$metadata.width"),
        'images-height': F("$metadata.height"),
        # 'bbox-top-left-x': F("$metadata.width") * F("bounding_box")[0],
        # 'bbox-top-left-y': F("$metadata.height") * F("bounding_box")[1],
        'bbox-width': F("$metadata.width") * F("bounding_box")[2],
        'bbox-height': F("$metadata.height") * F("bounding_box")[3],
        'objects per sample': F("detections.detections").length(),
        'objects per class': F("label"),
    },
}

#%% Derived expressions
analysis_expressions['statistical']['bbox-area'] = (
    analysis_expressions['statistical']['bbox-width'] * analysis_expressions['statistical']['bbox-height']
)
analysis_expressions['statistical']['bbox-aspect-ratio'] = (
    analysis_expressions['statistical']['bbox-width'] / analysis_expressions['statistical']['bbox-height']
)

# analysis_expressions['statistical']['occupancy ration'] = (
#     analysis_expressions['statistical']['bbox-area'] /
#     (analysis_expressions['statistical']['images-width'] * analysis_expressions['statistical']['images-height'])
# )

analysis_expressions['count']['# small-objects'] = (
    analysis_expressions['statistical']['bbox-area'] < 32 ** 2
)
analysis_expressions['count']['# medium-objects'] = (
    (32 ** 2 < analysis_expressions['statistical']['bbox-area']) &
    (analysis_expressions['statistical']['bbox-area'] < 96 ** 2)
)
analysis_expressions['count']['# large-objects'] = (
    analysis_expressions['statistical']['bbox-area'] > 96 ** 2
)

#%% Inconsistencies expressions
bbox_area = (
    F("$metadata.width") * F("bounding_box")[2] *
    F("$metadata.height") * F("bounding_box")[3]
)

aspect_ratio = (F("$metadata.width") * (F("bounding_box")[2]) / 
               (F("$metadata.height") * F("bounding_box")[3]))

inconsistencies_expr = {
        'zero_bbox': (F("bounding_box")[2] == 0) | 
                     (F("bounding_box")[3] == 0),

        'negative_bbox': (F("bounding_box")[0] < 0) | 
                         (F("bounding_box")[1] < 0) | 
                         (F("bounding_box")[2] < 0) | 
                         (F("bounding_box")[3] < 0),

        'occupancy_ratio': ((F("bounding_box")[2] * 
                             F("bounding_box")[3]) > 0.125),

        'aspect_ratio': ( (aspect_ratio > 10)),
    }

#%% FUNCTIONS
def get_datasets_info(datasets, plot=False):
    dataset_info = {}
    inconsistencies = {}

    for dataset_name in datasets:
        print(f"Dataset: {dataset_name}")
        dataset = fo.load_dataset(dataset_name)
        dataset.delete_saved_views()

        print('Checking for inconsistencies...')
        # Check for inconsistencies

        dataset_ = dataset
        with fo.ProgressBar() as pb:
            for inconsistency_name, expr in pb(inconsistencies_expr.items()):
                view = dataset_.filter_labels('detections',expr)

                if inconsistency_name in ['zero_bbox', 'negative_bbox'] :
                    dataset_ = dataset_.filter_labels("detections", ~expr,only_matches=False)
                    
                if len(view) > 0:
                    dataset.save_view(inconsistency_name, view)
                inconsistencies.setdefault(dataset_name, {})[f'{inconsistency_name}'] = len(view.distinct('detections.detections'))

        dataset = dataset_

        print('Performing analysis...')
        with fo.ProgressBar() as pb:
            for type_, exprs in pb(analysis_expressions.items()):

                type_info = dataset_info.setdefault(dataset_name, {}).setdefault(type_, {})
                for name, expr in exprs.items():

                    name_info = type_info.setdefault(name, {})
                    if type_ == 'statistical': 

                        mask = expr if name == 'objects per sample' else F("detections.detections[]").apply(expr)
                        if name == 'objects per class':

                            if dataset_name == 'Mapillary':
                                view = dataset.filter_labels('detections',F('label') != 'other-sign')
                            else:
                                view = dataset

                            values = list(view.count_values(mask).values())
                            values.sort()
                            name_info['bounds'] = [
                                round(x, 2) for x in [min(values),max(values)] if x is not None
                            ]
                            name_info['mean'] = float(round(np.mean(values),2))
                            name_info['std'] = float(round(np.std(values), 2))
                            name_info['quantiles'] = [
                                float(round(x, 2)) for x in np.percentile(values,[25,50,75,90]) if x 
                            ]
                        else:

                            values = dataset.values(mask)
                            name_info['bounds'] = [
                                round(x, 2) for x in dataset.bounds(mask) if x is not None
                            ]
                            name_info['mean'] = round(dataset.mean(mask), 2)
                            name_info['std'] = round(dataset.std(mask), 2)
                            name_info['quantiles'] = [
                                round(x, 2) for x in dataset.quantiles(mask, [0.25, 0.5, 0.75, 0.9]) if x
                            ]

                        if plot:
                            # Create a new figure for each name with histogram and boxplot side by side
                            fig, (ax_hist, ax_box) = plt.subplots(1, 2, figsize=(12, 4))
                            fig.suptitle(f"{dataset_name} - {name}", fontsize=14)

                            # Histogram
                            ax_hist.hist(values, bins=20, alpha=0.7, color="dodgerblue", edgecolor="black")
                            ax_hist.set_xlabel(name)
                            ax_hist.set_ylabel("Frequency")
                            ax_hist.set_title(f"Histogram: {name}")
                            ax_hist.grid(True)

                            # Boxplot
                            ax_box.boxplot(values, vert=False, patch_artist=True, showfliers=True)
                            ax_box.set_xlabel(name)
                            ax_box.set_title(f"Boxplot: {name}")
                            ax_box.grid(True)

                            plt.tight_layout(rect=[0, 0, 1, 0.95])
                            plt.show()

                    elif type_ == 'count':
                        if isinstance(expr, ViewExpression):
                            mask = F("detections.detections[]").apply(expr)
                            type_info[name] = dataset.count_values(mask).get(True, 0)
                        else:
                            type_info[name] = dataset.count(expr)

            if plot:
                # Plot heatmap das posições dos bounding boxes
                print("Plotting heatmap of bounding box positions...")
                plot_bbox_heatmap(dataset)

    #prepare data

    # 1. Tabela de contagens (cada linha = dataset)
    counts ={
        dataset: content["count"]
        for dataset, content in dataset_info.items()
    }



    # Novo formato para evitar usar .T
    geometry_dict = {}
    for dataset, content in dataset_info.items():
        for metric, stats in content['statistical'].items():
                          
            if 'bounds' in stats:
                key = f'bounds {metric}'
                if 'objects' in metric:
                    counts[dataset][key] = stats['bounds']
                else:
                    geometry_dict.setdefault(key, {})[dataset] = stats['bounds']
            if 'mean' in stats:
                key = f'mean {metric}'
                if 'objects' in metric:
                    counts[dataset][key] = stats['mean']
                else:
                    geometry_dict.setdefault(key, {})[dataset] = stats['mean']
            if 'std' in stats:
                key = f'std {metric}'
                if 'objects' in metric:
                    counts[dataset][key] = stats['std']
                else:
                    geometry_dict.setdefault(key, {})[dataset] = stats['std']

    # Criando o DataFrame diretamente no formato desejado
    geometry_stats = pd.DataFrame.from_dict(geometry_dict, orient='index')

    variables = [ x.replace('bounds ','').replace('mean ','').replace('std ','').replace('# ', '') 
                 for x in geometry_stats.index]

    metrics = [x.split(' ')[0].replace('#', 'amount') for x in geometry_stats.index]

    columns = list(geometry_stats.columns)
    columns.insert(0,'Metric')

    geometry_stats['Metric'] = metrics

    geometry_stats.index = variables

    geometry_stats = geometry_stats[columns]

    #geometry_stats.index.names = ['Variable']

    df_counts = pd.DataFrame(counts)

    variables = [ x.replace('bounds ','').replace('mean ','').replace('std ','').replace('# ', '')
              for x in df_counts.index]

    metrics = [x.split(' ')[0].replace('#', 'amount') for x in df_counts.index]

    columns = list(df_counts.columns)
    columns.insert(0,'Metric')

    df_counts['Metric'] = metrics

    df_counts.index = variables

    df_counts = df_counts[columns]  

    #df_counts.index.names = ['Variable']

    return dataset_info, df_counts, geometry_stats, inconsistencies


def plot_combined_bbox_heatmap(dataset_names, bins=(100, 100), show=True):
    """
    Plots a heatmap of bounding box centers for the union of all given datasets.
    Args:
        dataset_names: List of FiftyOne dataset names.
        bins: Number of divisions for the heatmap (resolution).
        show: If True, displays the heatmap.
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter
    all_x = []
    all_y = []
    for dataset_name in dataset_names:
        dataset = fo.load_dataset(dataset_name)
        for sample in dataset:
            if hasattr(sample, "detections") and sample.detections is not None:
                for det in sample.detections.detections:
                    x, y, w, h = det.bounding_box
                    cx = x + w / 2
                    cy = y + h / 2
                    all_x.append(cx)
                    all_y.append(cy)
    heatmap, xedges, yedges = np.histogram2d(all_x, all_y, bins=bins, range=[[0, 1], [0, 1]])
    heatmap_smooth = gaussian_filter(heatmap, sigma=2)
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap_smooth.T, origin='lower', cmap='hot', interpolation='bilinear', extent=[0, 1, 0, 1])
    plt.xlabel('Relative X Position')
    plt.ylabel('Relative Y Position')
    plt.title('Combined Heatmap of Bounding Box Centers')
    plt.colorbar(label='Frequency')
    if show:
        plt.show()
    return heatmap_smooth

def delete_datasets():
    """
    Opens a Tkinter window to select and delete FiftyOne datasets.
    """
    datasets = fo.list_datasets()
    root = tk.Tk()
    root.title("Delete Datasets")
    tk.Label(root, text="Select datasets to delete:").pack(pady=10)
    checkboxes = {}
    for dataset in datasets:
        var = tk.BooleanVar()
        checkboxes[dataset] = var
        tk.Checkbutton(root, text=dataset, variable=var).pack(anchor='w')

    def delete_selected_datasets(checkboxes, root):
        selected_datasets = [dataset for dataset, var in checkboxes.items() if var.get()]
        if selected_datasets:
            confirm = messagebox.askyesno(
                "Confirm Deletion",
                f"Are you sure you want to delete the selected datasets: {', '.join(selected_datasets)}?"
            )
            if confirm:
                for dataset in selected_datasets:
                    try:
                        fo.delete_dataset(dataset)
                    except Exception as e:
                        print(f"Error deleting dataset {dataset}: {e}")
                messagebox.showinfo("Success", f"Datasets deleted successfully: {', '.join(selected_datasets)}")
                root.destroy()
        else:
            messagebox.showwarning("No Selection", "Please select at least one dataset to delete.")

    delete_button = tk.Button(root, text="Delete Selected Datasets", command=lambda: delete_selected_datasets(checkboxes, root))
    delete_button.pack(pady=10)
    root.mainloop()


def select_dataset():
    """
    Opens a Tkinter window to select a FiftyOne dataset and returns the loaded dataset.
    """
    datasets = fo.list_datasets()
    root = tk.Tk()
    root.title("Select Dataset")
    tk.Label(root, text="Select dataset to load:").pack()
    root.geometry("300x80")
    clicked = tk.StringVar()
    clicked.set(datasets[0])
    drop = tk.OptionMenu(root, clicked, *datasets)
    drop.pack()
    button = tk.Button(root, text='Apply', command=root.destroy)
    button.pack()
    root.mainloop()
    return fo.load_dataset(clicked.get())


def delete_selected_fields(dataset):
    """
    Opens a Tkinter window to select and delete fields from a FiftyOne dataset.
    """
    exceptions = list(dataset._get_default_sample_fields())
    exceptions.append('detections')
    print('Exceptions: ', exceptions)
    root = tk.Tk()
    root.title("Delete Fields")
    tk.Label(root, text="Select fields to delete:").pack(pady=10)
    checkboxes = {}
    for field in dataset.get_field_schema():
        if field in exceptions:
            continue
        var = tk.BooleanVar()
        checkboxes[field] = var
        tk.Checkbutton(root, text=field, variable=var).pack(anchor='w')

    def delete_selected_datasets(checkboxes, root):
        selected_fields = [field for field, var in checkboxes.items() if var.get()]
        if selected_fields:
            confirm = messagebox.askyesno(
                "Confirm Deletion",
                f"Are you sure you want to delete the selected fields for {dataset.name}: {', '.join(selected_fields)}?"
            )
            if confirm:
                for field in selected_fields:
                    try:
                        dataset.delete_sample_field(field)
                    except Exception as e:
                        print(f"Error deleting field {field}: {e}")
                messagebox.showinfo("Success", f"Fields deleted successfully: {', '.join(selected_fields)}")
                root.destroy()
        else:
            messagebox.showwarning("No Selection", "Please select at least one field to delete.")

    delete_button = tk.Button(root, text="Delete Selected Fields", command=lambda: delete_selected_datasets(checkboxes, root))
    delete_button.pack(pady=10)
    root.mainloop()

def plot_bbox_heatmap(dataset, bins=(100, 100), show=True):
    """
    Plota um mapa de calor das posições centrais dos bounding boxes do dataset.
    Args:
        dataset: FiftyOne dataset carregado.
        bins: Número de divisões do heatmap (resolução).
        show: Se True, exibe o heatmap.
    """
    import numpy as np
    all_x = []
    all_y = []
    for sample in dataset:
        if hasattr(sample, "detections") and sample.detections is not None:
            for det in sample.detections.detections:
                x, y, w, h = det.bounding_box
                cx = x + w / 2
                cy = y + h / 2
                all_x.append(cx)
                all_y.append(cy)
    heatmap, xedges, yedges = np.histogram2d(all_x, all_y, bins=bins, range=[[0, 1], [0, 1]])
    # Apply Gaussian smoothing for a smoother heatmap
    from scipy.ndimage import gaussian_filter
    heatmap_smooth = gaussian_filter(heatmap, sigma=2)
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap_smooth.T, origin='lower', cmap='hot', interpolation='bilinear', extent=[0, 1, 0, 1])
    plt.xlabel('Relative X Position')
    plt.ylabel('Relative Y Position')
    plt.title('Heatmap of Bounding Box Centers')
    plt.colorbar(label='Frequency')
    if show:
        plt.show()
    return heatmap_smooth

def baixar_arquivo(dataset, url=None, downloaded_filename=None, drive_file_id=None,
                   folder_to_extract=None, folder_in_zip=False):
    """
    Downloads and extracts dataset files, with progress bar and error handling.
    """
    try:
        if drive_file_id:
            if downloaded_filename is None:
                downloaded_filename = f"{drive_file_id}.download"
            if not os.path.exists(datasets_dir):
                os.makedirs(datasets_dir)
            dataset_dir = os.path.join(datasets_dir, dataset)
            if not os.path.exists(dataset_dir):
                os.makedirs(dataset_dir)
            download_path = os.path.join(dataset_dir, downloaded_filename)
            if not os.path.exists(download_path):
                print(f"Downloading Google Drive file {drive_file_id} to {download_path}...")
                try:
                    gdown.download(id=drive_file_id, output=download_path, quiet=False)
                except Exception as e:
                    print(f"Error downloading from Google Drive: {e}")
                    return None
                print(f"Download of '{downloaded_filename}' complete!")
            else:
                print(f"File already exists: {downloaded_filename}.")
        else:
            if downloaded_filename is None:
                downloaded_filename = os.path.basename(url)
                if not downloaded_filename:
                    downloaded_filename = "downloaded_file"
                    print(f"Could not determine filename from URL, using '{downloaded_filename}'.")
            if not os.path.exists(datasets_dir):
                os.makedirs(datasets_dir)
            dataset_dir = os.path.join(datasets_dir, dataset)
            if not os.path.exists(dataset_dir):
                os.makedirs(dataset_dir)
            download_path = os.path.join(dataset_dir, downloaded_filename)
            if not os.path.exists(download_path):
                print(f"Downloading {url} to {download_path}...")
                try:
                    response = requests.get(url, stream=True, timeout=30)
                    response.raise_for_status()
                except Exception as e:
                    print(f"Error downloading file: {e}")
                    return None
                total_size_in_bytes = int(response.headers.get('content-length', 0))
                block_size = 1024
                progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc=downloaded_filename)
                try:
                    with open(download_path, 'wb') as file:
                        for data in response.iter_content(block_size):
                            progress_bar.update(len(data))
                            file.write(data)
                except Exception as e:
                    print(f"I/O error saving file: {e}")
                    progress_bar.close()
                    return None
                progress_bar.close()
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    print("ERROR, something went wrong during download.")
                    return None
                else:
                    print(f"Download of '{downloaded_filename}' completed!")
            else:
                print(f"File already exists: {downloaded_filename}.")
        # Extraction
        if folder_to_extract and not os.path.exists(os.path.join(dataset_dir, folder_to_extract)):
            if not folder_in_zip:
                folder_to_extract_path = os.path.join(dataset_dir, folder_to_extract)
            else:
                folder_to_extract_path = dataset_dir
            print(f"Extracting {download_path} to {folder_to_extract_path}...")
            try:
                if downloaded_filename.endswith('.zip'):
                    with ZipFile(download_path, 'r') as zip_obj:
                        files = zip_obj.namelist()
                        with tqdm(total=len(files), desc="Extracting", unit="file") as pbar:
                            for file in files:
                                zip_obj.extract(file, folder_to_extract_path)
                                pbar.update(1)
                    print(f"All files extracted from '{download_path}' to '{folder_to_extract_path}'.")
                elif downloaded_filename.endswith('.tar'):
                    with tarfile.open(download_path, 'r') as tar_obj:
                        members = tar_obj.getmembers()
                        with tqdm(total=len(members), desc="Extracting", unit="file") as pbar:
                            for member in members:
                                tar_obj.extract(member, folder_to_extract_path)
                                pbar.update(1)
                    print(f"All files extracted from '{download_path}' to '{folder_to_extract_path}'.")
            except FileNotFoundError:
                print(f"Error: Archive file not found at '{download_path}'.")
                return None
            except Exception as e:
                print(f"An error occurred during extraction: {e}")
                return None
        elif folder_to_extract:
            print(f"Folder {folder_to_extract} already exists in {dataset_dir}")
        print()
        return downloaded_filename
    except requests.exceptions.HTTPError as errh:
        print(f"HTTP Error: {errh}")
        if 'progress_bar' in locals() and progress_bar:
            progress_bar.close()
    except requests.exceptions.RequestException as err:
        print(f"A request error occurred: {err}")
        if 'progress_bar' in locals() and progress_bar:
            progress_bar.close()
    except IOError as e:
        print(f"I/O error saving file: {e}")
        if 'progress_bar' in locals() and progress_bar:
            progress_bar.close()
    except Exception as e:
        print(f"Unexpected error: {e}")
    return None


def resize_samples(dataset, W, H):

    with fo.ProgressBar() as pb:
        for sample in pb(dataset):
            # Resize images
            img = Image.open(sample.filepath)
            img = img.resize((W, H))
            img.save(sample.filepath)

            # Convert annotations to new size
            # Bboxes are in [top-left-x, top-left-y, width, height] format
            # if sample.detections is None:
            #     continue
            # for detection in sample.detections.detections:
            #     detection.bounding_box = [
            #         detection.bounding_box[0] * sample.metadata.width / W,
            #         detection.bounding_box[1] * sample.metadata.height / H,
            #         detection.bounding_box[2] * sample.metadata.width / W,
            #         detection.bounding_box[3] * sample.metadata.height / H,
            #     ]
            # sample.save()
        dataset.clear_sample_field('metadata')  # Remove old metadata
        dataset.compute_metadata()  # Recompute metadata with new dimensions

def load_datasets(datasets):

    if datasets == 'all':
        datasets = list(datasets_info.keys())

    for name in datasets:
        if name not in list(datasets_info.keys()):
            print(f"Dataset {name} not found. Please select one of the available datasets: {list(datasets_info.keys())}")
            continue
        print(f"Downloading dataset: {name}")
        if name in fo.list_datasets():
            print(f'Dataset {name} already exists.')
            continue
        for file_info in datasets_info[name]:
            
            baixar_arquivo(name, **file_info)

        if name == 'TT100k' and not fo.dataset_exists(name):

            dataset = fo.Dataset.from_dir(datasets_dir + f"{name}/tt100k_2021/train",
                                            tags='train',
                                            dataset_type=fo.types.ImageDirectory,
                                            name=name)
            
            dataset.add_dir(datasets_dir + f"{name}/tt100k_2021/test", 
                            tags='test',
                            dataset_type=fo.types.ImageDirectory)
            
            dataset.persistent = True
       
            dataset.compute_metadata()

            # Load the annotations
            annotations = datasets_dir + f"{name}/tt100k_2021/annotations_all.json"

            with open(annotations, encoding='utf-8', errors='replace') as f:
                data = json.load(f)

            dataset.tag_samples(name)

            # Add the labels to the dataset
            with fo.ProgressBar() as pb:
                for sample in pb(dataset):
                        if sample.filename[:-4] in data['imgs']:
                            detections = []
                            for object in data['imgs'][sample.filename[:-4]]['objects']:
                                # Convert to [top-left-x, top-left-y, width, height]
                                # in relative coordinates in [0, 1] x [0, 1]
                                W, H = sample.metadata.width, sample.metadata.height
                                x1, x2, y1, y2 = object['bbox']['xmin']/W, object['bbox']['xmax']/W, object['bbox']['ymin']/H, object['bbox']['ymax']/H
                                rel_box = [x1, y1, (x2 - x1), (y2 - y1)]
                                detections.append(
                                                fo.Detection(
                                                    label=object['category'],
                                                    bounding_box=rel_box,
                                                )
                                            )
                            if len(detections) != 0:                   
                                sample['detections'] = fo.Detections(detections=detections)
                            sample.save()
        
        
        if name == 'BDD100k' and not fo.dataset_exists(name):

            dataset = fo.Dataset.from_dir(datasets_dir + f"{name}/100k/train",
                                    tags='train',
                                    dataset_type=fo.types.ImageDirectory,
                                    name=name)
    
            dataset.add_dir(datasets_dir + f"{name}/100k/test", 
                            tags='test',
                            dataset_type=fo.types.ImageDirectory)
            
            dataset.add_dir(datasets_dir + f"{name}/100k/val", 
                            tags='val',
                            dataset_type=fo.types.ImageDirectory)
            
            dataset.persistent = True

            dataset.compute_metadata()

            dataset.tag_samples(name)

            with fo.ProgressBar() as pb:
                for sample in pb(dataset):
                    with open(sample.filepath.replace('.jpg','.json'), encoding='utf-8', errors='replace') as f:
                        data = json.load(f)
                    detections = []

                    W, H = sample.metadata.width, sample.metadata.height

                    for obj in data['frames'][0]['objects']:
                        if obj['category'] != 'traffic sign':
                            continue
                        
                        if 'box2d' in obj:
                            x1, y1, x2, y2 = obj['box2d']['x1']/W, obj['box2d']['y1']/H, obj['box2d']['x2']/W, obj['box2d']['y2']/H
                            rel_box = [x1, y1, x2 - x1, y2 - y1]
                            detections.append(
                                fo.Detection(
                                    label=obj['category'],
                                    bounding_box=rel_box,
                                )
                            )
                    if len(detections) != 0:  
                        sample['detections'] = fo.Detections(detections=detections)
                        sample['tags'] = [data['attributes']['weather'], data['attributes']['timeofday']]

                    sample.save()

        if name == 'CCTSDB' and not fo.dataset_exists(name):

            # Function to parse XML and extract data

            def parse_xml(xml_file):
                tree = ET.parse(xml_file)
                root = tree.getroot()

                tree = ET.parse(xml_file)
                root = tree.getroot()

                W = int(root.find('size/width').text)
                H = int(root.find('size/height').text)

                detections = []

                detections = []
                for obj in root.findall('object'):
                    label = obj.find('name').text
                    bbox = obj.find('bndbox')
                    xmin = int(float(bbox.find('xmin').text)) / W
                    ymin = int(float(bbox.find('ymin').text)) / H
                    xmax = int(float(bbox.find('xmax').text)) / W
                    ymax = int(float(bbox.find('ymax').text)) / H
                    rel_box = [xmin, ymin, xmax - xmin, ymax - ymin]
                    detections.append(
                        fo.Detection(
                            label=label,
                            bounding_box=rel_box,
                        )
                    )
                return detections
            
            dataset = fo.Dataset.from_dir(datasets_dir + f"{name}/train_img",
                                  tags='train',
                                  dataset_type=fo.types.ImageDirectory,
                                  name=name)
    
            dataset.add_dir(datasets_dir + f"{name}/test_img",
                            tags='test',
                            dataset_type=fo.types.ImageDirectory)
            
            dataset.persistent = True

            dataset.tag_samples(name)
            
            dataset.compute_metadata()

            # Directory containing XML files
            xml_dir = datasets_dir + f"{name}/xml"

            for xml_file in tqdm(os.listdir(xml_dir)):
                if xml_file.endswith('.xml'):
                    xml_file_path = os.path.join(xml_dir, xml_file)
                    filename = xml_file.replace('xml','jpg')
                    detections = parse_xml(xml_file_path)
                    view = dataset.match(F("filepath").contains_str(filename))
                    if len(view) > 0:
                        sample = view.first()
                        if len(detections) != 0:  
                            sample['detections'] = fo.Detections(detections=detections)
                        sample.save()
            dataset.compute_metadata()

        if name == 'GTSDB' and not fo.dataset_exists(name):

            CLASSES = ( 'speed limit 20', 'speed limit 30', 'speed limit 50', 'speed limit 60',
                    'speed limit 70', 'speed limit 80', 'restriction ends 80', 'speed limit 100', 'speed limit 120',
                    'no overtaking', 'no overtaking (trucks)', 'priority at next intersection', 'priority road',
                    'give way', 'stop', 'no traffic both ways',
                    'no trucks', 'no entry', 'danger', 'bend left','bend right', 'bend', 'uneven road',
                    'slippery road ', 'slippery road', 'road narrows', 'construction','traffic signal', 'pedestrian crossing', 'school crossing',
                    'cycles crossing', 'snow', 'animals', 'restriction ends',
                    'go right', 'go left', 'go straight', 'go right or straight','keep right',
                    'keep left ', 'roundabout', 'restriction ends', 'restriction ends')
            
            dataset_dir = datasets_dir + name + '/FullIJCNN2013/'

            images_dir = dataset_dir + 'images'

            if not os.path.exists(images_dir):
                os.makedirs(images_dir)

            for file in os.listdir(dataset_dir):
                #Convert ppm to jpg

                if not os.path.isfile(os.path.join(dataset_dir, file)) and os.path.join(dataset_dir, file) != images_dir:
                    shutil.rmtree(os.path.join(dataset_dir, file))

                if file.endswith('.ppm'):
                    img = cv2.imread(os.path.join(dataset_dir, file))
                    cv2.imwrite(os.path.join(images_dir, file.replace('.ppm','.jpg')), img)
                    os.remove(os.path.join(dataset_dir, file))

            #load GTSDB dataset


            dataset = fo.Dataset.from_dir(images_dir,
                                        dataset_type=fo.types.ImageDirectory,
                                        name=name)
            
            dataset.persistent = True

            with open(dataset_dir + 'gt.txt', encoding='utf-8', errors='replace') as f:
                data = pd.read_csv(f,sep=';',names=['file','left','top','right','bottom','id'],index_col=False)

            dataset.compute_metadata()

            dataset.tag_samples(name)

            with fo.ProgressBar() as pb:
                
                for sample in pb(dataset):
                    filename = sample.filename.replace('.jpg','.ppm')
                    view = data[data['file'] == filename]
                    detections = []
                    W, H = sample.metadata.width, sample.metadata.height
                    
                    for index, row in view.iterrows():
                        x1, y1, x2, y2 = row['left']/W, row['top']/H, row['right']/W, row['bottom']/H
                        rel_box = [x1, y1, x2 - x1, y2 - y1]
                        detections.append(
                            fo.Detection(
                                label=CLASSES[row['id']], #, 'traffic sign'
                                bounding_box=rel_box,
                            )
                        )

                    if len(detections) != 0:  
                        sample['detections'] = fo.Detections(detections=detections)
                    sample.save()

        if name == 'BelgiumTS':
            
            #[camera]/[image];[x1];[y1];[x2];[y2];[class id];[superclass id];

            SUPERCLASSES = {
                "undefined_traffic_sign": -1,
                "other_defined_TS": 0,
                "triangles": [2, 3, 4, 7, 8, 9, 10, 12, 13, 15, 17, 18, 22, 26, 27, 28, 29, 34, 35],
                "redcircles": [36, 43, 48, 50, 55, 56, 57, 58, 59, 61, 65],
                "bluecircles": [72, 75, 76, 78, 79, 80, 81],
                "redbluecircles": [82, 84, 85, 86],
                "diamonds": [32, 41],
                "revtriangle": [31],
                "stop": [39],
                "forbidden": [42],
                "squares": [118, 151, 155, 181],
                "rectanglesup": [37, 87, 90, 94, 95, 96, 97, 149, 150, 163],
                "rectanglesdown": [111, 112]
            }

            SUPERCLASSES = list(SUPERCLASSES.keys())

            classnames = [
                'Warning for a bad road surface',  # 0
                'Warning for a speed bump',  # 1
                'Warning for a slippery road surface',  # 2
                'Warning for a curve to the left',  # 3
                'Warning for a curve to the right',  # 4
                'Warning for a double curve, first left then right',  # 5
                'Warning for a double curve, first left then right',  # 6
                'Watch out for children ahead',  # 7
                'Watch out for cyclists',  # 8
                'Watch out for cattle on the road',  # 9
                'Watch out for roadwork ahead',  # 10
                'Traffic light ahead',  # 11
                'Watch out for railroad crossing with barriers ahead',  # 12
                'Watch out ahead for unknown danger',  # 13
                'Warning for a road narrowing',  # 14
                'Warning for a road narrowing on the left',  # 15
                'Warning for a road narrowing on the right',  # 16
                'Warning for side road on the right',  # 17
                'Warning for an uncontrolled crossroad',  # 18
                'Give way to all drivers',  # 19
                'Road narrowing, give way to oncoming drivers',  # 20
                'Stop and give way to all drivers',  # 21
                'Entry prohibited (road with one-way traffic)',  # 22
                'Cyclists prohibited',  # 23
                'Vehicles heavier than indicated prohibited',  # 24
                'Trucks prohibited',  # 25
                'Vehicles wider than indicated prohibited',  # 26
                'Vehicles higher than indicated prohibited',  # 27
                'Entry prohibited',  # 28
                'Turning left prohibited',  # 29
                'Turning right prohibited',  # 30
                'Overtaking prohibited',  # 31
                'Driving faster than indicated prohibited (speed limit)',  # 32
                'Mandatory shared path for pedestrians and cyclists',  # 33
                'Driving straight ahead mandatory',  # 34
                'Mandatory left',  # 35
                'Driving straight ahead or turning right mandatory',  # 36
                'Mandatory direction of the roundabout',  # 37
                'Mandatory path for cyclists',  # 38
                'Mandatory divided path for pedestrians and cyclists',  # 39
                'Parking prohibited',  # 40
                'Parking and stopping prohibited',  # 41
                '',  # 42
                '',  # 43
                'Road narrowing, oncoming drivers have to give way',  # 44
                'Parking is allowed',  # 45
                'parking for handicapped',  # 46
                'Parking for motor cars',  # 47
                'Parking for goods vehicles',  # 48
                'Parking for buses',  # 49
                'Parking only allowed on the sidewalk',  # 50
                'Begin of a residential area',  # 51
                'End of the residential area',  # 52
                'Road with one-way traffic',  # 53
                'Dead end street',  # 54
                '',  # 55
                'Crossing for pedestrians',  # 56
                'Crossing for cyclists',  # 57
                'Parking exit',  # 58
                'Information Sign: Speed bump',  # 59
                'End of the priority road',  # 60
                'Begin of a priority road'  # 61
            ]

            dataset_dir = datasets_dir + name

            images_dir = dataset_dir + '/images'
            

            if not os.path.exists(images_dir):
                os.makedirs(images_dir)

            for folder in os.listdir(dataset_dir):
                folder_path = os.path.join(dataset_dir, folder)

                if not os.path.isdir(folder_path) or \
                    'BelgiumTSD_annotations' in folder_path or \
                    'images' in folder_path:
                    continue
                
                print(f'Processing folder: {folder_path}')
                for file in tqdm(os.listdir(folder_path)):
                    new = os.path.join(images_dir, os.path.basename(folder_path) + '_' + file.replace('.jp2','.jpg'))
                    if os.path.exists(new):
                        continue
                    if file.endswith('.jp2'):
                        img = cv2.imread(os.path.join(folder_path, file))
                        cv2.imwrite( new, img)
                
                shutil.rmtree(folder_path)

            dataset = fo.Dataset.from_dir(images_dir,
                                        dataset_type=fo.types.ImageDirectory,
                                        name=name)
            
            dataset.persistent = True
            dataset.compute_metadata()

            dataset.tag_samples(name)

            with open(dataset_dir + '/BelgiumTSD_annotations/BTSD_training_GT.txt', encoding='utf-8', errors='replace') as f:
                training_data = pd.read_csv(f, sep=';', names=['file', 'left', 'top', 'right', 'bottom', 'id', 'super_id'], index_col=False)
            
            with open(dataset_dir + '/BelgiumTSD_annotations/BTSD_testing_GT.txt', encoding='utf-8', errors='replace') as f:
                test_data = pd.read_csv(f, sep=';', names=['file', 'left', 'top', 'right', 'bottom', 'id', 'super_id'], index_col=False)

            data = pd.concat([training_data, test_data])

            with fo.ProgressBar() as pb:

                for sample in pb(dataset):

                    camera = sample.filepath.split('_')[0][-2:]
                    filename = camera + '/' + sample.filename.replace('.jpg','.jp2').split('_')[1]
                    view = data[data['file'] == filename]

                    detections = []
                    W, H = sample.metadata.width, sample.metadata.height

                    for index, row in view.iterrows():
                        x1, y1, x2, y2 = row['left']/W, row['top']/H, row['right']/W, row['bottom']/H
                        rel_box = [x1, y1, x2 - x1, y2 - y1]
                        detections.append(
                            fo.Detection(
                                label=classnames[row['id']] if row['id'] < len(classnames) else 'undefined_traffic_sign',
                                super_label=SUPERCLASSES[row['super_id']],
                                bounding_box=rel_box,
                            )
                        )
                    if len(detections) != 0:  
                        sample['detections'] = fo.Detections(detections=detections)
                    sample.save()
