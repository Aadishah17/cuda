import os
import urllib.request
import gzip
import shutil

def download_mnist(target_dir="data/mnist"):
    os.makedirs(target_dir, exist_ok=True)
    
    files = {
        "train-images-idx3-ubyte": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz"
    }

    for name, url in files.items():
        dest_path = os.path.join(target_dir, name)
        if os.path.exists(dest_path):
            print(f"File {name} already exists. Skipping.")
            continue
            
        gz_path = dest_path + ".gz"
        print(f"Downloading {url} ...")
        try:
            urllib.request.urlretrieve(url, gz_path)
            print(f"Decompressing {gz_path} ...")
            with gzip.open(gz_path, 'rb') as f_in:
                with open(dest_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(gz_path)
            print(f"Successfully processed {name}")
        except Exception as e:
            print(f"Error downloading/decompressing {name}: {e}")
            if os.path.exists(gz_path):
                os.remove(gz_path)

if __name__ == "__main__":
    download_mnist()
