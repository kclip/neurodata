from torch.utils import data
import tables

from neurodata.utils.misc import *


class NeuromorphicDataset(data.Dataset):
    def __init__(
            self,
            path,
            train=True,
            size=[1],
            classes=[0],
            n_classes=0,
            sample_length=2e6,
            dt=1000,
            ds=1,
            polarity=True,
    ):

        self.path = path
        self.train = train
        self.size = size
        self.dt = dt
        self.sample_length = sample_length
        self.T = int(sample_length / dt)
        self.classes = classes
        if n_classes > 0:
            self.n_classes = len(classes)
        else:
            self.n_classes = n_classes
        self.ds = ds
        self.polarity = polarity

        super(NeuromorphicDataset, self).__init__()

        dataset = tables.open_file(path)

        if train:
            self.group = 'train'
            self.x_max = dataset.root.stats.train_data[1] // ds
        else:
            self.group = 'test'
            self.x_max = dataset.root.stats.test_data[1] // ds

        self.valid_idx = find_indices_for_labels(dataset.root[self.group],
                                                 self.classes)
        self.n_examples = len(self.valid_idx)
        dataset.close()

    def __len__(self):
        return self.n_examples

    def __getitem__(self, key):
        idx = self.valid_idx[key]
        dataset = tables.open_file(self.path)
        data, target = get_batch_example(dataset.root[self.group], idx,
                                         T=self.T,
                                         sample_length=self.sample_length,
                                         ds=self.ds, classes=self.classes,
                                         size=self.size, dt=self.dt,
                                         x_max=self.x_max,
                                         polarity=self.polarity)
        dataset.close()

        return data, target


def get_batch_example(hdf5_group, idx, T=80, sample_length=2e6, dt=1000,
                      ds=1, classes=[0], size=[1, 26, 26], x_max=1,
                      polarity=True):
    data = np.zeros([T] + size, dtype='float')
    label = hdf5_group.labels[idx]

    addrs = hdf5_group[str(idx)]
    idx_end = np.searchsorted(addrs[:, 0], sample_length)

    addrs = addrs[:idx_end]

    ts = np.arange(dt, addrs[-1, 0] + dt, dt)
    bucket_start = 0

    for i, t in enumerate(ts):
        bucket_end = np.searchsorted(addrs[:, 0], t)

        ee = addrs[bucket_start:bucket_end]

        pol, x, y = ee[:, 3], ee[:, 1] // ds, ee[:, 2] // ds

        try:
            if len(size) == 3:
                data[i, pol, x, y] = 1.
            elif len(size) == 2:
                data[i, pol, (x * x_max + y).astype(int)] = 1.
            elif len(size) == 1:
                if polarity:
                    data[i, (pol + 2 * (x * x_max + y)).astype(int)] = 1.
                else:
                    data[i, (x * x_max + y).astype(int)] = 1.
        except:
            i_max = np.argmax((pol + 2 * (x * x_max + y)))
            print(x[i_max], y[i_max], pol[i_max])
            raise IndexError

        bucket_start = bucket_end

    return torch.FloatTensor(data), make_output_from_labels(label, T,
                                                            classes, size)


def create_dataloader(path, batch_size=32, size=[1], classes=[0], n_classes=0,
                      sample_length_train=2e6, sample_length_test=2e6, dt=1000,
                      polarity=True, ds=1,
                      shuffle_train=True, shuffle_test=False, **dl_kwargs):
    train_dataset = NeuromorphicDataset(path,
                                        train=True,
                                        size=size,
                                        classes=classes,
                                        n_classes=n_classes,
                                        sample_length=sample_length_train,
                                        dt=dt,
                                        ds=ds,
                                        polarity=polarity
                                        )

    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=shuffle_train, **dl_kwargs)

    test_dataset = NeuromorphicDataset(path,
                                       train=False,
                                       size=size,
                                       classes=classes,
                                       n_classes=n_classes,
                                       sample_length=sample_length_test,
                                       dt=dt,
                                       ds=ds,
                                       polarity=polarity
                                       )

    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=shuffle_test, **dl_kwargs)

    return train_dl, test_dl
