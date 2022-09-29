import os
import numpy as np
#import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

#from tqdm import tqdm
from knn import Knn
import urllib.request
import gzip
img_size = 784

def display_image(image, title):
    image = image.squeeze()
    plt.figure()
    plt.title(title)
    plt.imshow(image, cmap=plt.cm.gray_r)



def load_mnist(root='./mnist'):

    # TODO Load the MNIST dataset
    # 1. Download the MNIST dataset from
    #    http://yann.lecun.com/exdb/mnist/
    url_base = 'http://yann.lecun.com/exdb/mnist/'
    key_file = {
        'train_img': 'train-images-idx3-ubyte.gz',
        'train_label': 'train-labels-idx1-ubyte.gz',
        'test_img': 't10k-images-idx3-ubyte.gz',
        'test_label': 't10k-labels-idx1-ubyte.gz'
    }

    dataset_dir = os.path.dirname(os.path.abspath(__file__))
    save_file = dataset_dir + "/mnist.pkl"
    i = 0
    for file_name in key_file.values():
        file_path = dataset_dir + "/" + file_name

        # if os.path.exists(file_path):
        #     continue

        # print("Downloading " + file_name + " ... ")
        # headers = {"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:47.0) Gecko/20100101 Firefox/47.0"}
        # request = urllib.request.Request('http://yann.lecun.com/exdb/mnist/' + file_name, headers=headers)
        # response = urllib.request.urlopen(request).read()
        # with open(file_path, mode='wb') as f:
        #     f.write(response)
        #print("Done")
    # 2. Unzip the MNIST dataset into the
    #    mnist directory.
        print("Converting " + file_name + " to NumPy Array ...")
        with gzip.open(file_path, 'rb') as f:
            objects = np.frombuffer(f.read(), np.uint8, offset=8*(2-i%2))
        print("Done")
    # 3. Load the MNIST dataset into the
    #    X_train, y_train, X_test, y_test
    #    variables.
        if file_name == 'train-images-idx3-ubyte.gz' :
            X_train = objects
        elif file_name == 'train-labels-idx1-ubyte.gz':
            y_train = objects
        elif file_name == 't10k-images-idx3-ubyte.gz':
            X_test = objects
        elif file_name == 't10k-labels-idx1-ubyte.gz':
            y_test = objects
        i += 1
    #.reshape(60000, 28, 28)
    #y_train.reshape(60000, 1)
    #X_test.reshape(10000, 28, 28)
    #y_test.reshape(10000,1)
    # Tr = float(X_train / 255.0)
    # Te = float(X_test / 255.0)
    # for key in (X_train):
    #     dataset[key] = dataset[key].astype(np.float32)
    #     dataset[key] /= 255.0
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    return X_train.reshape(60000, 28, 28) / 255.0,y_train.reshape(60000, ),X_test.reshape(10000, 28, 28) / 255.0,y_test.reshape(10000,)
    # Input:).astype(np.float16)
    # root: str, the directory of mnist

    # Output:
    # X_train: np.array, shape (6e4, 28, 28)
    # y_train: np.array, shape (6e4,)
    # X_test: np.array, shape (1e4, 28, 28)
    # y_test: np.array, shape (1e4,)

    # Hint:
    # 1. Use np.fromfile to load the MNIST dataset(notice offset).
    # 2. Use np.reshape to reshape the MNIST dataset.

    # YOUR CODE HERE
    # raise NotImplementedError
    ...

    # End of todo


def main():
    X_train, y_train, X_test, y_test = load_mnist()

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    #print(X_train[0],'\n',y_train[0],)
    for i in range(0,100):
        print(y_test[i])
    knn = Knn()
    knn.fit(X_train, y_train)
    #y_pred = np.ones(10000) * -1
    y_pred = knn.predict(X_test)
    correct = sum((y_test - y_pred) == 0)
    #
    print('==> correct:', correct)
    print('==> total:', len(X_test))
    print('==> acc:', correct / len(X_test))
    #
    # # plot pred samples
    # fig, ax = plt.subplots(nrows=4, ncols=5, sharex='all', sharey='all')
    # fig.suptitle('Plot predicted samples')
    # ax = ax.flatten()
    # for i in range(20):
    #     img = X_test[i]
    #     ax[i].set_title(y_pred[i])
    #     ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    # ax[0].set_xticks([])
    # ax[0].set_yticks([])
    # plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    main()
