from fire import Fire

from topvecsim.ml import train_save_top2vec, load_train_save_umap


def main():
    try:
        fire_obj = Fire(
            {"train": {"top2vec": train_save_top2vec, "umap": load_train_save_umap}}
        )
    except KeyboardInterrupt:
        print("\n\n\tBye!\n\n")


if __name__ == "__main__":
    main()
