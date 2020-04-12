from __future__ import print_function
import time
import numpy as np
import pandas as pd
# from sklearn.manifold import TSNE
from tsnecuda import TSNE
# from MulticoreTSNE import MulticoreTSNE as TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.utils import make_grid
import torch
from evolution.generator import Generator
from evolution.gan_train import GanTrain
from evolution.config import config
from metrics.fid.inception import InceptionV3
from metrics.fid.fid_score import get_activations
import logging
import glob
import os
import argparse
from util import tools
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)


def tsne_gan(images_per_model=8192, batch_size=128, exec_dir="runs/Jan24_00-29-04_cdv-5", dataset_name="MNIST",
             generations=[0, 9, 19, 29, 39, 49], frow=120, fcol=120, perplexity=30, n_iter=2000, win_size=1):
    print("cuda", torch.cuda.is_available())
    run_dirs = [(exec_dir, None)] + list(zip([exec_dir] * len(generations), generations))
    print("generations", generations)
    print("images_per_model", images_per_model)

    df, image_shape = load_data(batch_size, dataset_name, images_per_model, run_dirs)
    act, images = calc_activations(df, image_shape, batch_size)
    tsne_results = apply_tsne(act, perplexity, n_iter)

    df['tsne_x'], df['tsne_y'] = tsne_results[:, 0], tsne_results[:, 1]
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x="tsne_x", y="tsne_y", hue="model", data=df, legend="full", alpha=0.2)
    plt.savefig(os.path.join(exec_dir, "tsne_scatter.png"))
    generate_images(fcol, frow, image_shape, df, win_size=win_size)


def generate_images(fcol, frow, image_shape, df, win_size=1):
    df["tsne_x_int"] = ((fcol - 1) * (df["tsne_x"] - np.min(df["tsne_x"])) / np.ptp(df["tsne_x"])).astype(int)
    df["tsne_y_int"] = ((frow - 1) * (df["tsne_y"] - np.min(df["tsne_y"])) / np.ptp(df["tsne_y"])).astype(int)
    yy, xx = np.mgrid[-win_size:win_size + 1, -win_size:win_size + 1]
    all_possibilities = np.vstack([xx.reshape(-1), yy.reshape(-1)]).T
    all_possibilities = all_possibilities.tolist()
    all_possibilities.sort(key=lambda x: (max(abs(x[0]), abs(x[1])), abs(x[0]) + abs(x[1])))
    all_possibilities.pop(0)
    print(all_possibilities)

    print(df.groupby(by="model"))
    for model, group in df.groupby(by="model"):
        ordered_images = np.zeros((frow, fcol, *image_shape))
        overlap, show = 0, 0
        for i, row in group.iterrows():
            x, y = row["tsne_x_int"], row["tsne_y_int"]
            possibilities = list(all_possibilities)
            while np.sum(ordered_images[x, y]) != 0 and len(possibilities):
                dx, dy = possibilities.pop(0)
                x, y = np.clip(x + dx, 0, fcol - 1), np.clip(y + dy, 0, frow - 1)
            if np.sum(ordered_images[x, y]) == 0:
                show += 1
                ordered_images[x, y] = (row[get_features(image_shape)].values.reshape((-1, *image_shape)) + 1) / 2
            else:
                overlap += 1
        print(f"overlap for {model}: {overlap}, show: {show}")
        print(ordered_images.shape)
        ordered_images = np.flipud(np.transpose(ordered_images, (1, 0, 2, 3, 4))).reshape(frow * fcol, *image_shape)

        grid = make_grid(torch.tensor(ordered_images), nrow=frow).numpy()
        grid = np.transpose(grid, (1, 2, 0))
        plt.figure(figsize=(20, 20))
        run_dir, generation = group.iloc[0]['run_dir'], group.iloc[0]['generation']
        print("run_dir", run_dir, generation)
        save_path = "tsne_dataset.png" if generation is None else f"tsne_gen_{generation:03}.png"
        plt.imsave(os.path.join(run_dir, save_path), grid)
    return df


def apply_tsne(act, perplexity, n_iter):
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=n_iter)
    tsne_results = tsne.fit_transform(act)
    # del act
    print(f"Time elapsed: {time.time() - time_start} seconds")
    return tsne_results


def calc_activations(df, image_shape, batch_size):
    inception_model = tools.cuda(InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]))
    images = get_image_data(df, image_shape).reshape((-1, *image_shape))
    print(images.shape)
    act = get_activations(images, inception_model, batch_size=batch_size, dims=2048, cuda=tools.is_cuda_available(),
                          verbose=True)
    del inception_model
    torch.cuda.empty_cache()
    print(act.shape)
    return act, images


def get_features(image_shape):
    return list(range(np.prod(image_shape)))


def get_image_data(df, image_shape):
    return df[get_features(image_shape)].values


def load_data(batch_size, dataset_name, images_per_model, run_dirs):
    df = pd.DataFrame()
    image_shape = None
    noise_data = None
    for run_dir, generation in run_dirs:
        target_size = len(df) + images_per_model
        if generation is None:
            config.gan.dataset = dataset_name
            #     config.gan.dataset_resize = [64, 64]
            dataset = GanTrain.create_dataset()
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=True)
            # load images from dataset
            for images, labels in train_loader:
                image_shape = images.shape[1:]
                df_new = pd.DataFrame(images.numpy().reshape((-1, np.prod(image_shape))))
                df_new["model"] = dataset_name
                df_new["run_dir"] = run_dir
                df_new["generation"] = None
                df_new["y"] = np.zeros(len(images)) if len(labels.shape) > 1 else labels.numpy()
                df = df.append(df_new)
                if len(df) >= target_size:
                    break
        else:
            if noise_data is None:
                noise_data = Generator().generate_noise(images_per_model, volatile=True)
                print("noise data created", noise_data.shape)
            last_model = sorted(glob.glob(os.path.join(run_dir, "generations", f"{generation:03}", "generator.pkl")))[
                -1]
            best_generator = tools.cuda(Generator.load(last_model))
            n = 0
            while len(df) < target_size:
                noise = noise_data[n:min(n+batch_size, len(noise_data))]
                n += batch_size
                images = best_generator(noise).detach().cpu().numpy()
                image_shape = images.shape[1:]
                df_new = pd.DataFrame(images.reshape((-1, np.prod(image_shape))))
                df_new["model"] = f"{run_dir}|{generation}"
                df_new["run_dir"] = run_dir
                df_new["generation"] = generation
                df_new["y"] = np.zeros(len(images))
                df = df.append(df_new)
                del noise
                if len(df) >= target_size:
                    break
            best_generator = best_generator.cpu()
            torch.cuda.empty_cache()
    print(df.describe())
    return df, image_shape


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply t-SNE.')
    parser.add_argument('-i', "--images", type=int, help='images per model', default=8192)
    parser.add_argument('-b', "--batch", type=int, help='batch size', default=128)
    parser.add_argument('-p', "--path", help='execution path')
    parser.add_argument('-d', "--dataset", help='dataset name', default="MNIST")
    parser.add_argument('-g', "--generations", help='generations', nargs="+", type=int, default=[0, 9, 19, 29, 39, 49])
    parser.add_argument('-r', "--rows", type=int, help='rows', default=120)
    parser.add_argument('-c', "--cols", type=int, help='cols', default=120)
    parser.add_argument('-k', "--perplexity", type=int, help='perplexity', default=30)
    parser.add_argument('-n', "--iter", type=int, help='iterations', default=2000)
    parser.add_argument('-w', "--win", type=int, help='win size', default=1)
    args = parser.parse_args()
    print(args)
    tsne_gan(images_per_model=args.images, batch_size=args.batch, exec_dir=args.path, dataset_name=args.dataset,
             generations=args.generations, frow=args.rows, fcol=args.cols, perplexity=args.perplexity,
             n_iter=args.iter, win_size=args.win)
