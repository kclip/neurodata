import numpy as np
import os
import struct
import tables


""""
Load and preprocess data from the DVS Gesture dataset, based on scripts by Emre Neftci and Garrick Orchard
"""


def gather_aedat(directory, start_id, end_id):
    if not os.path.isdir(directory):
        raise FileNotFoundError("Dataset not found, looked at: {}".format(directory))

    dirs = [r'/' + dir_ for dir_ in os.listdir(directory)]

    fns = [[] for _ in range(10)]

    for i in range(10):
        for j in range(start_id, end_id):
            for dir_ in dirs:
                if dir_.find(str(i)) != -1:
                    fns[i].append(directory + dir_ + '/scale4' + '/' + ('mnist_%d_scale04_' % i) + "{0:04d}".format(j) + '.aedat')
    return fns


def aedat_to_events(datafile, min_pxl_value=48, max_pxl_value=73, dtype='uint32'):
    # constants
    aeLen = 8  # 1 AE event takes 8 bytes
    readMode = '>II'  # struct.unpack(), 2x ulong, 4B+4B
    xmask = 0x00fe
    ymask = 0x7f00
    pmask = 0x1

    aerdatafh = open(datafile, 'rb')
    k = 0  # line number
    p = 0  # pointer, position on bytes
    statinfo = os.stat(datafile)

    length = statinfo.st_size

    # header
    lt = aerdatafh.readline()
    while lt and lt[:1] == b'#':
        p += len(lt)
        k += 1
        lt = aerdatafh.readline()
        continue

    # variables to parse
    events = []
    # read data-part of file
    aerdatafh.seek(p)
    s = aerdatafh.read(aeLen)
    p += aeLen

    while p < length:
        addr, ts = struct.unpack(readMode, s)

        # parse event's data
        x_addr = 128 - 1 - ((xmask & addr) >> 1)
        y_addr = ((ymask & addr) >> 8)
        a_pol = (addr & pmask)

        if (x_addr >= min_pxl_value) & (x_addr <= max_pxl_value) & (y_addr >= min_pxl_value) & (y_addr <= max_pxl_value):
            events.append([ts, x_addr - min_pxl_value, y_addr - min_pxl_value, a_pol])

        aerdatafh.seek(p)
        s = aerdatafh.read(aeLen)
        p += aeLen

    events = np.column_stack(events).astype(dtype)
    label = np.array(int(datafile[-20])).astype('uint8')

    return events.T, label


def create_events_hdf5(path_to_hdf5, path_to_data=r'/processed_polarity', min_pxl_value=48, max_pxl_value=73, dtype='uint32'):
    fns_train = gather_aedat(path_to_data, 1, 901)
    fns_test = gather_aedat(path_to_data, 901, 1001)

    hdf5_file = tables.open_file(path_to_hdf5, 'w')

    hdf5_file.create_group(where=hdf5_file.root, name='train')
    train_labels_array = hdf5_file.create_earray(where=hdf5_file.root.train, name='labels', atom=tables.Atom.from_dtype(np.dtype('uint8')), shape=(0,))

    print("processing training data...")

    last_idx_train = 0
    for i, digit in enumerate(fns_train):
        for file in digit:
            print(file)
            events, label = aedat_to_events(file, min_pxl_value, max_pxl_value, dtype)

            hdf5_file.create_earray(where=hdf5_file.root.train, name=str(last_idx_train),  atom=tables.Atom.from_dtype(events.dtype), obj=events)
            train_labels_array.append(label[None])
            last_idx_train += 1


    hdf5_file.create_group(where=hdf5_file.root, name='test')
    test_labels_array = hdf5_file.create_earray(where=hdf5_file.root.test, name='labels', atom=tables.Atom.from_dtype(np.dtype('uint8')), shape=(0,))

    print("processing testing data...")
    last_idx_test = 0
    for i, digit in enumerate(fns_test):
        for file in digit:
            print(file)
            events, label = aedat_to_events(file, min_pxl_value, max_pxl_value, dtype)

            hdf5_file.create_earray(where=hdf5_file.root.test, name=str(last_idx_test),  atom=tables.Atom.from_dtype(events.dtype), obj=events)
            test_labels_array.append(label[None])
            last_idx_test += 1

    stats_train_data = np.array([9000, (1 + max_pxl_value - min_pxl_value)])
    stats_train_label = np.array([9000, 10])

    stats_test_data = np.array([1000, (1 + max_pxl_value - min_pxl_value)])
    stats_test_label = np.array([1000, 10])

    hdf5_file.create_group(where=hdf5_file.root, name='stats')
    hdf5_file.create_array(where=hdf5_file.root.stats, name='train_data', atom=tables.Atom.from_dtype(stats_train_data.dtype), obj=stats_train_data)
    hdf5_file.create_array(where=hdf5_file.root.stats, name='train_label', atom=tables.Atom.from_dtype(stats_train_label.dtype), obj=stats_train_label)
    hdf5_file.create_array(where=hdf5_file.root.stats, name='test_data', atom=tables.Atom.from_dtype(stats_test_data.dtype), obj=stats_test_data)
    hdf5_file.create_array(where=hdf5_file.root.stats, name='test_label', atom=tables.Atom.from_dtype(stats_test_label.dtype), obj=stats_test_label)

    hdf5_file.close()


def create_data(path_to_hdf5='../data/mnist_dvs_events.hdf5', path_to_data=None):
    if os.path.exists(path_to_hdf5):
        print("File {} exists: not re-converting data".format(path_to_hdf5))
    elif (not os.path.exists(path_to_hdf5)) & (path_to_data is not None):
        print("converting MNIST-DVS to h5file")
        create_events_hdf5(path_to_hdf5, path_to_data)
    else:
        print('Either an hdf5 file or MNIST DVS data must be specified')


create_data(path_to_hdf5=r"\datasets\mnist-dvs\mnist_dvs_events_new.hdf5",
            path_to_data=r"\datasets\mnist-dvs\processed_polarity"
            )
