"""ALCEO principal module.
"""
__version__ = '0.1.0'

def main():
    # %%
    from pathlib import Path


    dataset_root = Path("/home/gsech/Source/alceo/data/second_dataset/train")
    dataset_root.exists()
    # %%
    from glob import glob
    train_images = list(map(Path, glob(str(dataset_root / "im1" / "*.png"))))
    print(f"Found {len(train_images)} in dataset!")
    # %% Check all pairs are sane!

    for image1_path in train_images:
        image_id = image1_path.stem

        image2_path = dataset_root / "im2" / f"{image_id}.png"
        label1_path = dataset_root / "label1" / f"{image_id}.png"
        label2_path = dataset_root / "label2" / f"{image_id}.png"

        if not image1_path.exists() or not image2_path.exists() or not label1_path.exists() or not label2_path.exists():
            print(f"ID {image_id} is broken!")


if __name__ == '__main__':
    main()