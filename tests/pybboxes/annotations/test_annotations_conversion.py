import pytest
from pybboxes.annotations import Annotations
import os
import shutil
import glob
from functools import partial
from collections import Counter
from pycocotools.coco import COCO
from huggingface_hub import HfApi, hf_hub_download
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# hugging face repo from where we will be downloading our fixture for unit testing
repo_id = "gauravparajuli/coco_test_set_pybboxes"

sample_yolo_dataset_path = str(os.path.join('tests', 'pybboxes', 'annotations', 'testing_data_yolo'))
sample_voc_dataset_path = str(os.path.join('tests', 'pybboxes', 'annotations', 'testing_data_voc'))
sample_coco_dataset_path = str(os.path.join('tests', 'pybboxes', 'annotations', 'testing_data_coco', 'annotations_coco.json')) # source
persist_coco_test_path = str(os.path.join('tests', 'pybboxes', 'annotations', 'persist_as_coco_test.json')) # file generated during test_persist_as_coco

sample_images = str(os.path.join('tests', 'pybboxes', 'annotations', 'testing_data_images'))

def downloadfile(filename, local_dir):
    hf_hub_download(
        repo_id=repo_id,
        repo_type='dataset',
        filename=filename,
        local_dir=local_dir,
    )

def count_files(directory, extensions):
    all_files = []
    for ext in extensions:
        all_files.extend(glob.glob(f"{directory}/*{ext}"))
    return Counter(file.split('.')[-1] for file in all_files)

sample_coco_dataset = Annotations(annotation_type='coco')

def test_import_from_fiftyone():
    anns = Annotations(annotation_type='fiftyone')
    with pytest.raises(NotImplementedError):
        anns.load_from_fiftyone()

def test_import_from_albumentations():
    anns = Annotations(annotation_type='albumentations')
    with pytest.raises(NotImplementedError):
        anns.load_from_albumentations()

def test_save_as_fiftyone():
    anns = Annotations(annotation_type='albumentations')
    with pytest.raises(NotImplementedError):
        anns.save_as_fiftyone()

def test_save_as_albumentations():
    anns = Annotations(annotation_type='fiftyone')
    with pytest.raises(NotImplementedError):
        anns.save_as_albumentations()

def test_annotations_initialization():
    # annotation_type should be either: yolo, coco, voc, albumentations or fiftyone
    with pytest.raises(ValueError):
        anns = Annotations(annotation_type='not_this_type')

def test_annotations_only_appropriate_loading_method_allowed():
    # tests if unappropriate method is used to load annotations
    anns = Annotations('yolo')
    with pytest.raises(TypeError):
        anns.load_from_voc(labels_dir='./labels')
    with pytest.raises(TypeError):
        anns.load_from_coco(json_path='./sample.json')

    anns = Annotations('coco')
    with pytest.raises(TypeError):
        anns.load_from_yolo(labels_dir='./labels', images_dir='./images', classes_file='./classes.txt')

def test_import_from_coco():
    anns = sample_coco_dataset
    anns.load_from_coco(sample_coco_dataset_path)

    assert (type(anns.names_mapping)) == dict
    assert anns.names_mapping == dict(raccoons=0, raccoon=1)

    # randomly test the accuracy of annotations here
    
@pytest.mark.depends(on=['test_save_as_yolo'])
def test_import_from_yolo():
    anns = Annotations(annotation_type='yolo')
    anns.load_from_yolo(labels_dir=sample_yolo_dataset_path, images_dir=sample_images, classes_file=str(os.path.join(sample_yolo_dataset_path, 'classes.txt')))

    assert (type(anns.names_mapping)) == dict
    assert anns.names_mapping == dict(raccoons=0, raccoon=1)

@pytest.mark.depends(on=['test_save_as_voc'])
def test_import_from_voc():
    anns = Annotations(annotation_type='voc')
    anns.load_from_voc(labels_dir=sample_voc_dataset_path)

    assert (type(anns.names_mapping)) == dict
    assert anns.names_mapping == dict(raccoon=0) # as raccoons label was not used in any bounding boxes,
                                                 # plus there is not a file that lists all the available class in voc format
                                                 # there was a loss of information
                                                 # when converting from coco format to voc format

@pytest.mark.depends(on=['test_import_from_coco'])
def test_save_as_coco():
    persist_coco_path = str(os.path.join('tests', 'pybboxes', 'annotations', 'persist_as_coco_test.json'))
    sample_coco_dataset.save_as_coco(export_file=persist_coco_path)

    coco = COCO(persist_coco_path)

    assert len(coco.getImgIds()) == 196
    assert len(coco.getCatIds()) == 2

@pytest.mark.depends(on=['test_import_from_coco'])
def test_save_as_yolo():
    sample_coco_dataset.save_as_yolo(sample_yolo_dataset_path)

    assert count_files(sample_yolo_dataset_path, extensions=['.txt'])['txt'] == 197 # 196 annotation files, 1 classes.txt file

@pytest.mark.depends(on=['test_import_from_coco'])
def test_save_as_voc():
    sample_coco_dataset.save_as_voc(sample_voc_dataset_path)

    assert count_files(sample_voc_dataset_path, extensions=['.xml'])['xml'] == 196 # 196 annotation files

@pytest.fixture(scope='session', autouse=True)
def cleanup():
    # setup code here
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type='dataset')
    files = [file for file in files if ('.json' in file or '.jpg' in file)] # filter .gitattributes and README.md

    annotationfilename = files.pop(0) # annotations_coco.json
    downloadfile(annotationfilename, local_dir=os.path.dirname(sample_coco_dataset_path)) # download annotation file in a separate folder

    # now download test dataset images
    with ThreadPoolExecutor() as executor:
        partial_downloadfile = partial(downloadfile, local_dir=sample_images)
        futures = [executor.submit(partial_downloadfile, filename) for filename in files]
        with tqdm(total=len(futures), desc='downloading test set for unit testing', unit='file') as pbar:
            for future in as_completed(futures):
                pbar.set_description_str = future.result()
                pbar.update(1) # update the progress bar for each completed download

    yield

    # clean up the folders that we created after all the tests have ran
    shutil.rmtree(sample_voc_dataset_path)
    shutil.rmtree(sample_yolo_dataset_path)
    os.remove(persist_coco_test_path) # remove the test file