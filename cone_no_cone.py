from cnnlearning import *
from learning_utils import *
import os
from scipy.spatial import Voronoi


def get_random_points_in_voronoi_diagram(centroids):
    vor = Voronoi(centroids, qhull_options='Qbb Qc Qx', incremental=False)
    vor.close()

    edges = np.array(vor.ridge_vertices)

    edges_start = edges[:, 0]
    edges_end = edges[:, 1]

    vertices_start = vor.vertices[edges_start]
    vertices_end = vor.vertices[edges_end]

    t = np.random.rand(vertices_start.shape[0])

    random_vertices = t[:, np.newaxis] * vertices_start + (1 - t[:, np.newaxis]) * vertices_end
    random_vertices = random_vertices[edges_start != -1]

    random_vertices = random_vertices[random_vertices[:, 0] >= 0]
    random_vertices = random_vertices[random_vertices[:, 0] <= 200]
    random_vertices = random_vertices[random_vertices[:, 1] >= 0]
    random_vertices = random_vertices[random_vertices[:, 1] <= 200]

    return random_vertices


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    SHARED_CONES = os.path.join(".", "data", "ConesShare")
    SHARED_VIDEOS_PATH = os.path.join(".", "data", "Shared_Videos")
    OUTPUT_FOLDER = os.path.join(".", "data", "output")
    TRAINED_MODEL_FOLDER = os.path.join(OUTPUT_FOLDER, "trained_models")

    video_filenames = [file for file in
                       [f for f in os.listdir(SHARED_VIDEOS_PATH) if f.endswith('avi') and 'OA790nm' in f]]
    marked_video_filenames = [os.path.join(SHARED_VIDEOS_PATH, file) for file in video_filenames if 'marked' in file]
    raw_video_filenames = [os.path.join(SHARED_VIDEOS_PATH, file) for file in video_filenames if 'marked' not in file]

    csv_filenames = [os.path.join(SHARED_VIDEOS_PATH, file) for file in
                     [f for f in os.listdir(SHARED_VIDEOS_PATH) if f.endswith('csv') and 'OA790nm' in f]]

    print("BLOOD CELLS")
    print("-----------")
    print("RAW VIDEOS:")
    print(*marked_video_filenames, sep="\n")
    print()

    print("MARKED VIDEOS:")
    print(*raw_video_filenames, sep="\n")
    print()

    print("CSV FILES:")
    print(*csv_filenames, sep="\n")

    cone_images_filenames = [os.path.join(SHARED_CONES, file) for file in
                             [f for f in os.listdir(SHARED_CONES) if f.endswith('tif')]]
    cone_csv_filenames = [os.path.join(SHARED_CONES, file) for file in
                          [f for f in os.listdir(SHARED_CONES) if f.endswith('txt')]]

    print()
    print("CONES")
    print("-----")
    print("TIFF:")
    print(*cone_images_filenames, sep="\n")
    print()

    print("CSV:")
    print(*cone_csv_filenames, sep="\n")

    cone_image_size = (33, 33)

    model = CNN(
        convolutional=
        nn.Sequential(
            nn.Conv2d(1, 32, padding=2, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),

            nn.Conv2d(32, 32, padding=2, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, padding=1, stride=2),

            nn.Conv2d(32, 64, padding=2, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, padding=1, stride=2),
        ),
        dense=
        nn.Sequential(
            nn.Linear(1024, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(64),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.Linear(32, 2),
            #   nn.Softmax()
        )
    ).to(device)
    model.load_state_dict(torch.load('cone_model.pt'))
    model = model.eval()

    sample_cone_image = plt.imread(cone_images_filenames[0]).astype(np.float32) / 255
    plt.imshow(sample_cone_image)

    probability_map = get_frame_probability_map(sample_cone_image, model, height=33, width=33)
    plt.imshow(probability_map)
    plt.show()


if __name__ == "__main__":
    main()
