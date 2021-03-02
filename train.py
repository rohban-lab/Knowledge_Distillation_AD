from argparse import ArgumentParser

from torch import nn
from random import randrange
from models.network import get_networks
from test import *
from utils.utils import *
from dataloader import *
from pathlib import Path
from torch.autograd import Variable
import pickle
from test_functions import detection_test

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml', help="training configuration")


def train_with_complete_loss(config):
    direction_loss_only = config["direction_loss_only"]
    assert direction_loss_only == False, "Direction Only is true and this block should not run"

    normal_class = config["normal_class"]
    learning_rate = float(config['learning_rate'])
    num_epochs = config["num_epochs"]
    lamda = config['lamda']

    checkpoint_path = "./outputs/{}/{}/checkpoints/".format(config['experiment_name'], config['dataset_name'])

    # create directory
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    train_dataloader, test_dataloader = load_data(config)
    vgg, model = get_networks(config)

    # Criteria And Optimizers
    criterion = nn.MSELoss()
    similarity_loss = torch.nn.CosineSimilarity()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    losses = []
    roc_aucs = []
    for epoch in range(num_epochs + 1):  #
        model.train()
        epoch_loss = 0
        for data in train_dataloader:
            X = data[0]
            if X.shape[1] == 1:
                X = X.repeat(1, 3, 1, 1)
            X = Variable(X).cuda()
            # X = Variable(X)
            # Precit the output for Given input
            # adv_X = fgsm_attack(X, model)
            # print(len(output_pred))
            output_pred = model.forward(X)
            output_real = vgg(X)
            y_pred_0, y_pred_1, y_pred_2, y_pred_3 = output_pred[3], output_pred[6], output_pred[9], output_pred[12]
            y_0, y_1, y_2, y_3 = output_real[3], output_real[6], output_real[9], output_real[12]
            # Compute Cross entropy loss
            # print(y_pred_0.shape)
            # print(y_0.shape)
            abs_loss_0 = criterion(y_pred_0, y_0)
            loss_0 = torch.mean(1 - similarity_loss(y_pred_0.view(y_pred_0.shape[0], -1), y_0.view(y_0.shape[0], -1)))
            abs_loss_1 = criterion(y_pred_1, y_1)
            loss_1 = torch.mean(1 - similarity_loss(y_pred_1.view(y_pred_1.shape[0], -1), y_1.view(y_1.shape[0], -1)))
            abs_loss_2 = criterion(y_pred_2, y_2)
            loss_2 = torch.mean(1 - similarity_loss(y_pred_2.view(y_pred_2.shape[0], -1), y_2.view(y_2.shape[0], -1)))
            abs_loss_3 = criterion(y_pred_3, y_3)
            loss_3 = torch.mean(1 - similarity_loss(y_pred_3.view(y_pred_3.shape[0], -1), y_3.view(y_3.shape[0], -1)))

            total_loss = loss_0 + loss_1 + loss_2 + loss_3 + lamda * (abs_loss_0 + abs_loss_1 + abs_loss_2 + abs_loss_3)
            dist_loss = 0

            total_loss += dist_loss

            # Add loss to the list
            epoch_loss += total_loss.item()
            losses.append(total_loss.item())
            # Clear the previous gradients
            optimizer.zero_grad()
            # Compute gradients
            total_loss.backward()
            # Adjust weights
            optimizer.step()
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
        if epoch % 10 == 0:
            roc_auc = detection_test(model, vgg, test_dataloader, config)
            roc_aucs.append(roc_auc)
            print("RocAUC at epoch {}:".format(epoch), roc_auc)
        if epoch % 10 == 0:
            torch.save(model.state_dict(),
                       '{}Cloner_{}_epoch_{}.pth'.format(checkpoint_path, normal_class, epoch))
            torch.save(optimizer.state_dict(),
                       '{}Opt_{}_epoch_{}.pth'.format(checkpoint_path, normal_class, epoch))
            with open('{}Auc_{}_epoch_{}.pickle'.format(checkpoint_path, normal_class, epoch),
                      'wb') as f:
                pickle.dump(roc_aucs, f)


def main():
    args = parser.parse_args()
    config = get_config(args.config)
    direction_loss_only = config["direction_loss_only"]
    if not direction_loss_only:
        train_with_complete_loss(config)


if __name__ == '__main__':
    main()
