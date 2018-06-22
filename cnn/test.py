import sys, os
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)

from data.cifar10.cifar import Cifar10
cif = Cifar10()
print(cif.path)
cif.load()