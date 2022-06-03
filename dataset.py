from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img


if __name__ == '__main__':
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
    )
    dset256 = MultiResolutionDataset('/home/sunlab/2021Studets/gaochao/datasets/style/lmdb_mul', transform=transform, resolution=256)
    dset512 = MultiResolutionDataset('/home/sunlab/2021Studets/gaochao/datasets/style/lmdb_mul', transform=transform, resolution=512)
    dset1024 = MultiResolutionDataset('/home/sunlab/2021Studets/gaochao/datasets/style/lmdb_mul', transform=transform, resolution=1024)

    loader256 = DataLoader(dset256, batch_size=4, num_workers=0)
    loader512 = DataLoader(dset512, batch_size=4, num_workers=0)
    loader1024 = DataLoader(dset1024, batch_size=4, num_workers=0)

    _,img256 = next(enumerate(loader256))
    _,img512 = next(enumerate(loader512))
    _,img1024 = next(enumerate(loader1024))

    transforms.ToPILImage()(img256[0]).save('./doc/img256.png')
    transforms.ToPILImage()(img512[0]).save('./doc/img512.png')
    transforms.ToPILImage()(img1024[0]).save('./doc/img1024.png')


    print("ok")
