import struct
import numpy as np
import os
import tables

""""
Load and preprocess data from the DVS Gesture dataset, based on scripts by Emre Neftci.
"""


def gather_gestures_stats(hdf5_grp):
    from collections import Counter
    labels = []
    for d in hdf5_grp:
        labels += hdf5_grp[d]['labels'][:, 0].tolist()
    count = Counter(labels)
    stats = np.array(list(count.values()))
    stats = stats / stats.sum()
    return stats


def gather_aedat(directory, start_id, end_id, filename_prefix='user'):
    if not os.path.isdir(directory):
        raise FileNotFoundError("DVS Gestures Dataset not found,"
                                " looked at: {}".format(directory))
    import glob
    fns = []
    for i in range(start_id, end_id):
        search_mask = directory + '/' + filename_prefix + "{0:02d}".format(i) + '*.aedat'
        glob_out = glob.glob(search_mask)
        if len(glob_out) > 0:
            fns += glob_out
    return fns


def gather_aedat_by_light(directory):
    if not os.path.isdir(directory):
        raise FileNotFoundError("DVS Gestures Dataset not found, "
                                "looked at: {}".format(directory))
    import glob
    search_mask = directory + '/*.aedat'
    all_aedat = glob.glob(search_mask)

    fluorescent_led = [file for file in all_aedat if "fluorescent_led" in file]
    train_files = fluorescent_led
    fluorescent = [file for file in all_aedat if
                   "fluorescent" in file and file not in train_files]
    train_files += fluorescent
    led = [file for file in all_aedat if
           "led" in file and file not in train_files]
    train_files += led
    lab = [file for file in all_aedat if
           "lab" in file and file not in train_files]
    train_files += lab

    test_files = [file for file in all_aedat if file not in train_files]
    return train_files, test_files


def gather_aedat_by_light_and_id(directory, last_id_train):
    id_train = ["{0:02d}".format(i) for i in range(1, last_id_train)]
    id_test = ["{0:02d}".format(i) for i in range(last_id_train, 30)]
    all_train_files, all_test_files = gather_aedat_by_light(directory)

    train_files = []
    for i in id_train:
        train_files += [file for file in all_train_files if
                        i in file.split('\\')[-1]]

    test_files = []
    for i in id_test:
        test_files += [file for file in all_test_files if
                       i in file.split('\\')[-1]]

    return train_files, test_files


def aedat_to_events(filename, type='uint32'):
    """"
    type: datatype of the the obtained array. With uint32, the max value is 4,294,967,295
    This corresponds to samples of maximum length ~4,294s with a sampling rate of 1mus, enough for any current neuromorphic dataset
    """

    label_filename = filename[:-6] + '_labels.csv'
    labels = np.loadtxt(label_filename, skiprows=1, delimiter=',', dtype='int64')
    events = []
    with open(filename, 'rb') as f:
        for i in range(5):
            f.readline()
        while True:
            data_ev_head = f.read(28)
            if len(data_ev_head) == 0: break

            eventtype = struct.unpack('H', data_ev_head[0:2])[0]
            eventsource = struct.unpack('H', data_ev_head[2:4])[0]
            eventsize = struct.unpack('I', data_ev_head[4:8])[0]
            eventoffset = struct.unpack('I', data_ev_head[8:12])[0]
            eventtsoverflow = struct.unpack('I', data_ev_head[12:16])[0]
            eventcapacity = struct.unpack('I', data_ev_head[16:20])[0]
            eventnumber = struct.unpack('I', data_ev_head[20:24])[0]
            eventvalid = struct.unpack('I', data_ev_head[24:28])[0]

            if (eventtype == 1):
                event_bytes = np.frombuffer(f.read(eventnumber * eventsize), 'uint32')
                event_bytes = event_bytes.reshape(-1, 2)

                x = ((event_bytes[:, 0] >> 17) & 0x00001FFF)
                y = ((event_bytes[:, 0] >> 2) & 0x00001FFF)
                p = (event_bytes[:, 0] >> 1) & 0x00000001
                t = event_bytes[:, 1]
                events.append([t, x, y, p])
            else:
                f.read(eventnumber * eventsize)

    events = np.column_stack(events)

    events = events.astype(type).T
    clipped_events = {i: None for i in range(len(labels))}

    for i, l in enumerate(labels):
        start = np.searchsorted(events[:, 0], l[1])
        end = np.searchsorted(events[:, 0], l[2])
        clipped_events[i] = events[start:end]
        try:
            clipped_events[i][:, 0] -= clipped_events[i][0, 0]
        except IndexError:
            print(l, start, end, events[start, 0], events[end, 0])
            return None, None
    labels[:, 0] -= 1

    return clipped_events, labels[:, 0].astype('uint8')


def create_events_hdf5(path_to_hdf5, path_to_data, dtype='uint32', split='id',
                       split_idx=24):
    if split == 'id':
        fns_train = gather_aedat(path_to_data, 1, split_idx)
        fns_test = gather_aedat(path_to_data, split_idx, 30)
    elif split == 'light':
        fns_train, fns_test = gather_aedat_by_light(path_to_data)
    else:
        fns_train, fns_test = gather_aedat_by_light_and_id(path_to_data, 15)

    hdf5_file = tables.open_file(path_to_hdf5, 'w')
    hdf5_file.create_group(where=hdf5_file.root, name='train')
    train_labels_array = hdf5_file.create_earray(where=hdf5_file.root.train,
                                                 name='labels',
                                                 atom=tables.Atom.from_dtype(
                                                     np.dtype('uint8')),
                                                 shape=(0,))

    print("processing training data...")
    last_idx_train = 0
    for file_d in fns_train:
        events, labels = aedat_to_events(file_d, dtype)
        if labels is not None:
            for i in range(len(labels)):
                hdf5_file.create_earray(where=hdf5_file.root.train,
                                        name=str(i + last_idx_train),
                                        atom=tables.Atom.from_dtype(
                                            events[i].dtype),
                                        obj=events[i])

            train_labels_array.append(labels)
            last_idx_train += len(labels)

    hdf5_file.create_group(where=hdf5_file.root, name='test')
    test_labels_array = hdf5_file.create_earray(where=hdf5_file.root.test,
                                                name='labels',
                                                atom=tables.Atom.from_dtype(
                                                    np.dtype('uint8')),
                                                shape=(0,))

    print("processing testing data...")
    last_idx_test = 0
    for file_d in fns_test:
        events, labels = aedat_to_events(file_d, dtype)
        if labels is not None:
            for i in range(len(labels)):
                hdf5_file.create_earray(where=hdf5_file.root.test,
                                        name=str(i + last_idx_test),
                                        atom=tables.Atom.from_dtype(
                                            events[i].dtype),
                                        obj=events[i])
            test_labels_array.append(labels)
            last_idx_test += len(labels)

    stats_train_data = np.array([len(hdf5_file.root.train.labels[:]), 128])
    stats_train_label = np.array([len(hdf5_file.root.train.labels[:]), 11])

    stats_test_data = np.array([len(hdf5_file.root.test.labels[:]), 128])
    stats_test_label = np.array([len(hdf5_file.root.test.labels[:]), 11])

    hdf5_file.create_group(where=hdf5_file.root, name='stats')
    hdf5_file.create_array(where=hdf5_file.root.stats, name='train_data',
                           atom=tables.Atom.from_dtype(stats_train_data.dtype),
                           obj=stats_train_data)
    hdf5_file.create_array(where=hdf5_file.root.stats, name='train_label',
                           atom=tables.Atom.from_dtype(stats_train_label.dtype),
                           obj=stats_train_label)
    hdf5_file.create_array(where=hdf5_file.root.stats, name='test_data',
                           atom=tables.Atom.from_dtype(stats_test_data.dtype),
                           obj=stats_test_data)
    hdf5_file.create_array(where=hdf5_file.root.stats, name='test_label',
                           atom=tables.Atom.from_dtype(stats_test_label.dtype),
                           obj=stats_test_label)

def create_data(path_to_hdf5='../data/mnist_dvs_events.hdf5',
                path_to_data=None, dtype='uint32', split='id', split_idx=24):
    if os.path.exists(path_to_hdf5):
        print("File {} exists: not re-converting data".format(path_to_hdf5))
    elif (not os.path.exists(path_to_hdf5)) & (path_to_data is not None):
        print("converting DvsGestures to h5file")
        create_events_hdf5(path_to_hdf5, path_to_data, dtype, split, split_idx)
    else:
        print('Either an hdf5 file or DvsGestures data must be specified')

home = r"C:\Users\K1804053\OneDrive - King's College London\PycharmProjects"
create_data(path_to_hdf5=home + r"\datasets\DvsGesture\dvs_gestures_events_small_.hdf5",
            path_to_data=home + r"\datasets\DvsGesture",
            split='id',
            split_idx=10
            )
