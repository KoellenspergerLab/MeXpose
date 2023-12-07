import os

import argparse
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler, MinMaxScaler


def detect_delimiter(csv_file_path):
    with open(csv_file_path, 'r') as f:
        first_line = f.readline()
    sniffer = csv.Sniffer()
    delimiter = sniffer.sniff(first_line).delimiter
    return delimiter


def generate_info(cluster_label, channel, df):
    mean_val = df.loc[df["label"] == cluster_label, channel].mean()
    min_val = df.loc[df["label"] == cluster_label, channel].min()
    max_val = df.loc[df["label"] == cluster_label, channel].max()

    if cluster_label == 0:
        cluster_name = "background (cluster 0)"
    else:
        cluster_name = f"cluster_{cluster_label}"

    info = (
        f"The average single pixel value in counts for {channel} in the {cluster_name} is: {mean_val}\n"
        f"The minimum single pixel value in counts for {channel} in {cluster_name} is: {min_val}\n"
        f"The maximum single pixel value in counts for {channel} in {cluster_name} is: {max_val}"
    )
    return info


def main():
    parser = argparse.ArgumentParser(description="Pixel Script")
    parser.add_argument("--data_csv", required=True,
                        help="Path to the data CSV file.")
    parser.add_argument("--setup_csv", required=True,
                        help="Path to the setup CSV file.")
    parser.add_argument("--image_dims", required=True,
                        help="Dimensions for the generated images, format: width,height")
    parser.add_argument("--no_cluster", action="store_true",
                        help="Skip clustering.")
    parser.add_argument("--no_heatmap", action="store_true",
                        help="Skip heatmap generation.")
    parser.add_argument("--n_clusters", type=int, default=3,
                        help="Number of clusters for KMeans.")
    parser.add_argument("--no_transform", action="store_true",
                        help="Disables square root transformation of data.")
    parser.add_argument('--scaling_method', type=str, choices=['robust', 'minmax'], default='robust',
                        help='Choose the scaling method for clustering. Options are "robust" and "minmax". Default is "robust".')
    parser.add_argument("--correlation_method", type=str, choices=['pearson', 'spearman'], default='spearman',
                        help="Select the desired correlation method. Options are 'pearson' and 'spearman'. Default is 'spearman'.")

    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = os.path.join(os.path.dirname(args.data_csv), 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_delimiter = detect_delimiter(args.data_csv)
    data_df = pd.read_csv(args.data_csv, delimiter=data_delimiter)

    setup_delimiter = detect_delimiter(args.setup_csv)
    setup_df = pd.read_csv(
        args.setup_csv, delimiter=setup_delimiter)

    # Get image dimensions
    width, height = map(int, args.image_dims.split(','))

    info_strings = []

    # Set scaling method based on user choice
    if args.scaling_method == 'robust':
        scaler = RobustScaler()

    elif args.scaling_method == 'minmax':
        scaler = MinMaxScaler()

    # Clustering
    if not args.no_cluster:
        cluster_channels = setup_df.loc[setup_df['cluster']
                                        == 1, 'channels'].tolist()

        cluster_scaled = scaler.fit_transform(data_df[cluster_channels])
        kmeans = KMeans(n_clusters=args.n_clusters, n_init='auto')
        kmeans.fit(cluster_scaled)
        labels = kmeans.labels_

        # Adding cluster labels to DataFrame
        data_df["label"] = labels

        # Generate cluster statistics
        for i in range(0, labels.max() + 1):
            for channel in cluster_channels:
                info = generate_info(i, channel, data_df)
                print(info)
                info_strings.append(info)
            print()
            info_strings.append("")

        # Save statistics to a text file
        with open(os.path.join(output_dir, 'statistics.txt'), 'w') as f:
            f.write("\n".join(info_strings))

        # Generate the image based on clusters
        image = np.zeros((height, width, 3), dtype=np.uint8)

        cmap = plt.cm.viridis

        for i, label in enumerate(labels):
            row = i // width
            col = i % width
            normalized_label = label / (args.n_clusters - 1)
            color = cmap(normalized_label)[:3]
            color = np.array(color)
            color = (color * 255).astype(np.uint8)
            image[row, col, :] = color[::-1]

        plt.imshow(image)
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'clustered_image.png'),
                    dpi=600, bbox_inches='tight', pad_inches=0)

    # Heatmap Generation
    if not args.no_heatmap:
        heatmap_channels = setup_df.loc[setup_df['heatmap']
                                        == 1, 'channels'].tolist()

        if not args.no_transform:
            data_df[heatmap_channels] = data_df[heatmap_channels].applymap(lambda x: np.sqrt(max(0, x)))

        heatmap_scaled = scaler.fit_transform(data_df[heatmap_channels])
        heatmap_scaled_df = pd.DataFrame(heatmap_scaled, columns=heatmap_channels)

        # Perform correlation
        if args.correlation_method == 'spearman':
            correlation_matrix =heatmap_scaled_df.corr(method="spearman")

        elif args.correlation_method == 'pearson':
            correlation_matrix =heatmap_scaled_df.corr(method="pearson")

        sns.set(font_scale=0.5)
        fig, ax = plt.subplots()
        sns.heatmap(correlation_matrix,
                    vmin=-1,
                    vmax=1,
                    center=0,
                    cmap=sns.diverging_palette(20, 220, n=200),
                    square=True,
                    annot=True,
                    ax=ax,
                    )

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45,
                           horizontalalignment="right")

        plt.tight_layout()
        plt.savefig(os.path.join(
            output_dir, args.correlation_method + '_correlation_heatmap.png'), dpi=600)


if __name__ == "__main__":
    main()
