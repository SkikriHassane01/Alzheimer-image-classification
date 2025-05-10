import os
import random
from PIL import Image
import matplotlib.pyplot as plt
from src.utils.logger import get_logger
logger = get_logger("3_exploratory_analysis")

def plot_class_distribution(data_dir: str) -> None:
    """
    Plot the distribution of samples in each class directory.

    :param data_dir: Directory containing class subdirectories.
    """
    if not os.path.exists(data_dir):
        logger.error("Data directory '%s' does not exist.", data_dir)
        return

    # Filter to include only directories representing classes.
    classes = sorted([c for c in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, c))])
    counts = [len(os.listdir(os.path.join(data_dir, c))) for c in classes]
    distribution = dict(zip(classes, counts))
    logger.info("Class distribution: %s", distribution)

    plt.figure(figsize=(8, 6))
    plt.bar(classes, counts, color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Samples per Class')
    plt.tight_layout()
    figures_dir = os.path.join("figures", "eda")
    os.makedirs(figures_dir, exist_ok=True)
    fig_path = os.path.join(figures_dir, "class_distribution.png")
    plt.savefig(fig_path)
    logger.info("Saved class distribution plot to %s", fig_path)
    plt.close()

def show_random_samples(data_dir: str, class_name: str, num: int = 9) -> None:
    """
    Display a grid of random samples from a specified class.

    :param data_dir: Directory containing class subdirectories.
    :param class_name: Name of the class to display images from.
    :param num: Number of random samples to display (default is 9).
    """
    class_path = os.path.join(data_dir, class_name)
    if not os.path.exists(class_path):
        logger.error("Class directory '%s' does not exist.", class_path)
        return

    # List only files in the class directory.
    image_files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
    if not image_files:
        logger.warning("No images found in class directory '%s'.", class_path)
        return

    samples = random.sample(image_files, min(len(image_files), num))
    # Determine grid layout for a near-square display.
    grid_columns = int(num ** 0.5) if int(num ** 0.5) ** 2 == num else 3
    grid_rows = (len(samples) + grid_columns - 1) // grid_columns

    plt.figure(figsize=(grid_columns * 3, grid_rows * 3))
    for idx, fname in enumerate(samples):
        img_path = os.path.join(class_path, fname)
        try:
            img = Image.open(img_path).convert('L')
        except Exception as e:
            logger.error("Failed to open image '%s': %s", img_path, e)
            continue
        ax = plt.subplot(grid_rows, grid_columns, idx + 1)
        ax.imshow(img, cmap='gray')
        ax.set_title(class_name, fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    figures_dir = os.path.join("figures", "eda")
    os.makedirs(figures_dir, exist_ok=True)
    fig_path = os.path.join(figures_dir, f"random_samples_{class_name.replace(' ', '_')}.png")
    plt.savefig(fig_path)
    logger.info("Saved random samples plot for %s to %s", class_name, fig_path)
    plt.close()

if __name__ == "__main__":
    data_dir = os.path.join("Data", "raw")
    plot_class_distribution(data_dir)
    show_random_samples(data_dir, "Mild Dementia", num=9)
    show_random_samples(data_dir, "Non Demented", num=9)
    show_random_samples(data_dir, "Moderate Dementia", num=9)
    show_random_samples(data_dir, "Very Mild Dementia", num=9)
Z