import os
import shutil
import tarfile
import pandas as pd
from PIL import Image
import scipy.io as scio
from contextlib import redirect_stdout
# multi-thread
import time
import queue
import threading

# global vars
train_file = 'ILSVRC2012_img_train.tar'
val_file = 'ILSVRC2012_img_val.tar'
test_file = 'ILSVRC2012_img_test_v10102019.tar'
dev_kit_file = 'ILSVRC2012_devkit_t12.tar.gz'

class SynsetsQueue:
    def __init__(self, length=25):
        self.queue = queue.Queue(length)
        self.locking = threading.Lock()
        self.max_length = length
        self.processed = 0
        self.stored = 0
        self.work = True # flag to terminate threads
    # end __init__

    def toggle_lock(self):
        if self.locking.locked():
            self.locking.release()
        else:
            self.locking.acquire()
    # end toggle_lock

    def get(self):
        data = None
        self.toggle_lock()
        if not self.empty():
            data = self.queue.get()

        self.toggle_lock()
        return data
    # end get

    def add(self, synsets):
        self.toggle_lock() # lock

        if isinstance(synsets, list):
            for s in synsets:
                self._add(s)
        else:
            self._add(synsets)

        self.toggle_lock() # unlock
        return True
    # end add

    def _add(self, data):
        if self.full():
            self.wait_for_space()
        self.queue.put(data)
        self.stored += 1
        return True
    # end _add

    def full(self):
        return self.queue.full()
    # end full

    def empty(self):
        return self.queue.empty()
    # end empty

    def space_available(self):
        return (self.max_length - self.queue.qsize())
    # end space_available

    def wait_for_space(self, wait_time=2):
        self.toggle_lock() # unlock
        while self.space_available() >= (self.max_length // 2):
            time.sleep(wait_time)
        self.toggle_lock() # lock
        return True
    # end wait_for_space
# end SynsetsQueue

class UntarThread(threading.Thread):
    def __init__(self, threadID, q, source_path, target_path, filename=None, del_source=False):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = "TH{}".format(threadID)
        self.q = q
        self.source_path = source_path
        self.target_path = target_path
        self.filename = filename
        self.del_source = del_source
    # end __init__

    def run(self):
        print ("{} thread started.".format(self.name))
        while self.q.work:
            data = self.q.get()
            if data is None:
                continue
            else:
                if self.filename is None:
                    fn = data
                    member = None
                else:
                    fn = self.filename
                    member = data
                msge = self.untar_to_folder(fn, self.source_path, self.target_path, member=member, del_source=self.del_source)
                self.q.processed += 1
                print("[{}] {:04d}/~{:04d} {} {}".format(
                self.name, self.q.processed, self.q.stored, data, msge))
            time.sleep(1)
        print ("{} thread finished.".format(self.name))
    # end run

    def untar_to_folder(self, filename, source_path, target_path, member=None, del_source=False):
        name = filename.replace('.tar', '')
        source_file = os_path(source_path, filename)
        target = os_path(target_path, name)

        if member is None:
            if os.path.isdir(target):
                return 'already extracted! skipping...'
            else:
                print("[{}] processing {}...".format(self.name, filename))
            tf = tarfile.open(source_file)
            tf.extractall(target)
            tf.close()
        else:
            if os.path.isfile(os_path(target, member)):
                return 'already exists! skipping...'
            else:
                print("[{}] processing {} from {}...".format(self.name, member, filename))
            tf = tarfile.open(source_file)
            tf.extract(member, path=target)
            tf.close()

        if del_source and os.path.isfile(source_file):
            os.remove(source_file)

        return 'extracted successfully!'
    # end untar_to_folder
# end UntarThread

def os_path(*args):
    return os.path.join(*args)
# end os_path

def to_human_readable(name):
    readable = name.split(',')[0].strip().replace(' ', '_')
    readable = readable.replace("'", '').lower()
    return readable
# end to_human_readable

def dev_kit_path(source_path):
    target_path = dev_kit_file.replace('.tar.gz', '')
    if not os.path.isdir(target_path):
        tf = tarfile.open(os_path(source_path, dev_kit_file), 'r:gz')
        tf.extractall(source_path)
        tf.close()
    return os_path(source_path, target_path, 'data')
# end dev_kit_path

def meta_info(source_path):
    meta_filename = 'meta.mat'
    n_classes = 1000
    # read dataset metadata
    meta_data = scio.loadmat(os_path(dev_kit_path(source_path), meta_filename), squeeze_me=True)
    synsets = meta_data['synsets']

    ids_qt = list(zip(synsets['WNID'][:n_classes],
                map(to_human_readable, synsets['words'][:n_classes]),
                synsets['num_train_images'][:n_classes]))

    return pd.DataFrame(data=ids_qt, columns=['WNID', 'human_readable', 'num_train_images'])
# end meta_info

def check_train_extraction(source_path, target_path):
    # read extracted dirs
    n_files = []
    directories = []
    for d in os.listdir(target_path):#directories:
        dt = os_path(target_path, d)
        if os.path.isdir(dt):
            directories.append(d)
            n_files.append(len(os.listdir(dt)))
    extracted_files = dict(zip(directories, n_files))

    # read dataset metadata
    df = meta_info(source_path)
    print("[INFO] {} of {} WNIDs extracted!".format(len(extracted_files), len(df)))

    # check number images
    differences = []
    missing = []
    for index, row in df.iterrows():
        wnid = row['WNID']
        if wnid in extracted_files:
            if extracted_files[wnid] != row['num_train_images']:
                differences.append(wnid)
        else:
            missing.append(wnid)

    if len(missing) > 0:
        print("[INFO] {} WNIDs are missing!".format(len(missing)))
    if len(differences) > 0:
        print("[INFO] {} extracted WNIDs have differences!".format(len(differences)))

    return missing, differences
# end check_train_extraction

def process_threads(workq, source_path, target_path, items=None, filename=None, max_threads=4, del_source=False):
    # ensure functionality
    if (filename is None and items is None):
        raise ValueError('Error, filename or items must be setted.')

    if not os.path.isdir(target_path):
        os.makedirs(target_path)

    if items is None:
        # tar file
        train_tar = tarfile.open(os_path(source_path, filename))
        items = train_tar.getnames()
        train_tar.close()

    # force to initiate processing
    workq.work = True

    # main file threads init
    threads = []
    for i in range(max_threads):
       th = UntarThread(i, workq, source_path, target_path, filename=filename, del_source=del_source)
       th.start()
       threads.append(th)

    # queue for main file
    workq.add(items)

    # wait for queue empty
    while not workq.empty():
       pass

    # stop threads
    workq.work = False

    # wait for thread finish untar
    for th in threads:
       th.join()

    #reset queue
    workq.processed = 0
    workq.stored = 0
# end process_threads

def extract_from_mainfile(sq, source_path, target_path, items=None, filename=train_file, workers=4):
    if os.path.isfile(os_path(source_path, filename)):
        print("Found {} at {}".format(filename, source_path))
    else:
        raise ValueError("Source path invalid! no {} file was founded.".format(filename))

    print("Extracting from mainfile {}".format(filename))
    process_threads(sq, source_path, target_path, items=items, filename=filename, max_threads=workers)
    print("File {} has been uncompressed successfully!".format(filename))
# end extract_from_mainfile

def extract_subfiles(sq, untar_target, items, workers=4, del_source=True):
    print("Extracting subfiles...")
    process_threads(sq, untar_target, untar_target, items=items, max_threads=workers, del_source=del_source)
    print("All files were extracted!")
# end extract_subfiles

def append_extention(fn):
    return "{}.tar".format(fn)
# end append_extention

def untar_train(source_path, target_path, q_size=50, workers=4):
    # work queue and target location
    sq = SynsetsQueue(length=q_size)
    untar_target = os_path(target_path, train_file.replace('.tar', ''))
    if not os.path.isdir(untar_target):
        os.makedirs(untar_target)

    # check for continue or correction
    missing, differences = check_train_extraction(source_path, untar_target)

    if len(missing) > 0:
        missing = list(map(append_extention, missing))
        extract_from_mainfile(sq, source_path, target_path, items=missing, workers=workers)
        time.sleep(1)
        extract_subfiles(sq, untar_target, items=missing, workers=workers)

    if len(differences) > 0:
        print("Processing differences...\nRemoving old folders...")
        differences = list(map(append_extention, differences))
        # remove existing folders
        for d in differences:
            shutil.rmtree(os_path(untar_target, d))
        extract_from_mainfile(sq, source_path, target_path, items=differences, workers=workers)
        time.sleep(1)
        extract_subfiles(sq, untar_target, items=differences, workers=workers)
# end untar_train

def untar_val(source_path, target_path, q_size=25, workers=4):
    # work queue and target location
    sq = SynsetsQueue(length=q_size)
    untar_target = os_path(target_path, val_file.replace('.tar', ''))
    # check if already untar
    if os.path.isdir(untar_target) and len(os.listdir(untar_target)) == 50000:
        print('Validation file was already extracted!')
        return False

    extract_from_mainfile(sq, source_path, target_path, filename=val_file, workers=workers)
# end untar_val

def untar_test(source_path, target_path, q_size=25, workers=4):
    # work queue and target location
    sq = SynsetsQueue(length=q_size)
    t_file = test_file.replace('.tar', '').split('_')
    del t_file[-1] # remove version for the folder name
    t_file = '_'.join(t_file)
    untar_target = os_path(target_path, t_file)

    if os.path.isdir(untar_target) and len(os.listdir(untar_target)) == 100000:
        print('Testing file was already extracted!')
        return False

    extract_from_mainfile(sq, source_path, target_path, filename=test_file, workers=workers)
# end untar_test

def generate_csv(source_path, target_path):
    csv_path = os_path(target_path, 'dataset.csv')
    if os.path.isfile(csv_path):
        options = ['n', 'y']
        msge = "File {} already exists!, regenerate? [y]/n: ".format(csv_path)
        force = str(input(msge) or "y").lower()
        while force not in options:
            force = str(input("Invalid! {}".format(msge)) or "y").lower()
        force = bool(options.index(force))
        if not force:
            print('Exiting!')
            return False
        print('Regenerating dataset.csv')
    else:
        print('Creating dataset.csv')
    # training set
    img_path = train_file.replace('.tar', '')
    metadata = meta_info(source_path)

    df = None
    # training data
    for idx, row in metadata.iterrows():
        wnid = row['WNID']
        f_path = os_path(img_path, wnid)
        # images id
        path_imgs = os.listdir(os_path(target_path, f_path))
        class_df = pd.DataFrame(data=path_imgs, columns=['image_id'])
        # insert class data
        class_df.insert(0, 'folder_path', f_path)
        class_df.insert(2, 'class', wnid)
        class_df.insert(3, 'class_name', row['human_readable'])
        # append main dataframe
        df = pd.concat([df, class_df])

    df.insert(4, 'purpose', 'train')

    # validation data
    val_gt_file = 'ILSVRC2012_validation_ground_truth.txt'
    f_path = val_file.replace('.tar', '')
    # images id
    path_imgs = os.listdir(os_path(target_path, f_path))
    # classes from id to wnid
    wnid_val = pd.read_csv(os_path(dev_kit_path(source_path), val_gt_file), header=None)
    classes = []
    classes_name = []
    for idx, row in wnid_val.iterrows():
        i = row[0]-1
        classes.append(metadata['WNID'][i])
        classes_name.append(metadata['human_readable'][i])

    # create dataframe
    class_df = pd.DataFrame(data=path_imgs, columns=['image_id'])
    class_df.insert(0, 'folder_path', f_path)
    class_df.insert(2, 'class', classes)
    class_df.insert(3, 'class_name', classes_name)
    class_df.insert(4, 'purpose', 'val')
    # append main dataframe
    df = pd.concat([df, class_df])
    print(df)

    df.to_csv(csv_path, index=False)
    print('Saved at', csv_path)
# end generate_csv

def check_images(target_path, purpose='train'):
    # csv read
    csv_path = os_path(target_path, 'dataset.csv')
    print('Reading from file: {}'.format(csv_path))
    dataset_df = pd.read_csv(csv_path)

    if purpose is not None and 'purpose' in dataset_df:
        dataset_df = dataset_df[dataset_df['purpose'] == purpose]

    problems_idx = []
    with open('info.log', 'w') as f:
            with redirect_stdout(f):
                for idx, row in dataset_df.iterrows():
                    img_path = os_path(target_path,
                                    row['folder_path'],
                                    row['image_id'])
                    print(idx, row['image_id'])
                    try:
                        image = Image.open(img_path).convert('RGB')
                    except OSError as e:
                        print(e)
                        problems_idx.append(idx)
                        print('problem:', img_path)

                print(problems_idx)

# end check_images

if __name__ == '__main__':
    header_msg = \
"""#                                                              #
# For ImageNet use the next files must be downloaded first:    #
#    1) Development kit (Task 1 & 2) -> the metadata.          #
#    2) Training images (Task 1 & 2) -> the training set.      #
#    3) Validation images (all tasks) -> the validation set.   #
#    4) Test images (all tasks) -> the test set.               #
#                                                              #
# These files are available in the ImageNet's site             #
# http://www.image-net.org/download                            #
# All the files must be contained in the same path before      #
# execute the script.                                          #
# This script is intented to unpack the classification's task  #
# image set (Task 1).                                          #
#                                                              #"""
    print("#"*64)
    print(header_msg)
    print("#"*64)
    # params
    self_path = os.path.dirname(os.path.abspath(__file__)) # self file's path
    source_path = input('ImageNet source path [./imagenet]: ') or './imagenet'
    source_path = source_path.strip()
    if os.path.isdir(source_path):
        destination_path = os.path.abspath(os_path(self_path, '..', 'datasets', 'ImageNetDataset'))
        #untar_train(source_path, destination_path, workers=4)
        #untar_val(source_path, destination_path, workers=1)
#        untar_test(source_path, destination_path, workers=1)
        #generate_csv(source_path, destination_path)
        check_images(destination_path)
    else:
        print("The entered location was not found! Exiting...")
