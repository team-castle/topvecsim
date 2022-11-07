from fire import Fire

from topvecsim.ml import train_save_top2vec, load_train_save_umap
from topvecsim.load_data import load_all_data_from_disk


def main():
    try:
        fire_obj = Fire(
            {
                "train": {"top2vec": train_save_top2vec, "umap": load_train_save_umap},
                "load_data": load_all_data_from_disk,
            }
        )
    except KeyboardInterrupt:
        print("\n\n\tBye!\n\n")


if __name__ == "__main__":
    main()
