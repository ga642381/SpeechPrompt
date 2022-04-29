import argparse
import yaml
from pathlib import Path
from importlib.machinery import SourceFileLoader
import IPython


def main(args):
    with open(f"{args.downstream}/config.yaml", "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)

    preprocessor_module = SourceFileLoader(
        "preprocessor", f"{args.downstream}/preprocessor.py"
    ).load_module()

    preprocessor = preprocessor_module.Preprocessor(config)
    eval(f"preprocessor.{args.action}")()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--downstream", help="which downstream task to preprocess")
    parser.add_argument("--action")

    args = parser.parse_args()
    main(args)
