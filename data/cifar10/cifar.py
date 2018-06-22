import glob
import hashlib
import os
import sys
import tarfile
import urllib.request

import pyprind


class Cifar10:
    def __init__(self, path=''):
        self.path = path
        if not self.path:
            this_path = os.path.abspath(os.path.join(__file__, os.pardir))
            self.path = this_path
        self.save_file_name = 'cifar.tar.gz'
        self.url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        self.md5_sum = 'c58f30108f718f92721af3b95e74349a'
        self.last_percent_reported = None
        self.bar = None
        self.files = []

    def __download_progress_hook(self, count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        if self.last_percent_reported != percent:
            bar.update()
            self.last_percent_reported = percent

    def load(self):
        global bar
        dest_path = os.path.join(self.path, self.save_file_name)
        is_download = True
        is_success = False
        if os.path.isfile(dest_path):
            download_md5 = hashlib.md5(open(dest_path, 'rb').read()).hexdigest()
            if self.md5_sum == download_md5:
                is_download = False
                is_success = True
        if is_download:
            bar = pyprind.ProgBar(100, monitor=True, title='Downloading')
            filename, _ = urllib.request.urlretrieve(self.url, dest_path, reporthook=self.__download_progress_hook)
            download_md5 = hashlib.md5(open(dest_path, 'rb').read()).hexdigest()
            if self.md5_sum == download_md5:
                print('\n')
                print('{} is downloaded.'.format(filename))
                is_success = True
            else:
                print('\n')
                print('{} is failed. md5 not match. please retry.'.format(filename))
        if is_success:
            with tarfile.open(dest_path) as fo:
                fo.extractall(self.path)
            for file_path in glob.glob("%s/*" % os.path.join(self.path,'cifar-10-batches-py')):
                self.files.append(file_path)


def main(*args):
    obj = Cifar10()
    # obj.load()


if __name__ == '__main__':
    main(*sys.argv[1:])
