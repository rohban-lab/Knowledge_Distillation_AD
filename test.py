from argparse import ArgumentParser
from utils.utils import get_config
from dataloader import load_data, load_localization_data
from test_functions import detection_test, localization_test
from models.network import get_networks

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml', help="training configuration")


def main():
    args = parser.parse_args()
    config = get_config(args.config)
    vgg, model = get_networks(config, load_checkpoint=True)

    # Localization test
    if config['localization_test']:
        test_dataloader, ground_truth = load_localization_data(config)
        roc_auc = localization_test(model=model, vgg=vgg, test_dataloader=test_dataloader, ground_truth=ground_truth,
                                    config=config)

    # Detection test
    else:
        _, test_dataloader = load_data(config)
        roc_auc = detection_test(model=model, vgg=vgg, test_dataloader=test_dataloader, config=config)
    last_checkpoint = config['last_checkpoint']
    print("RocAUC after {} epoch:".format(last_checkpoint), roc_auc)


if __name__ == '__main__':
    main()
