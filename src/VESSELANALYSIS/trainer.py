import torch
import numpy as np
import time
from datetime import timedelta
from matplotlib import pyplot as plt


def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, log_interval, start_epoch=0,
        save_progress_path=None, show_plots=True):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    train_losses = []
    val_losses = []
    for epoch in range(start_epoch, n_epochs):
        print("Starting Epoch", epoch)
        scheduler.step()

        # Train stage
        train_loss = train_epoch(train_loader, model, loss_fn, optimizer, log_interval)
        train_losses.append(train_loss)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)

        if val_loader is not None:
            val_loss = test_epoch(val_loader, model, loss_fn)
            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, val_loss)

        print(message)

        if save_progress_path is not None:
            if len(val_losses) <= 1 or val_losses[-1] < np.min(np.array(val_losses[:-1])):
                print("Saving model weights")
                state = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }
                torch.save(state, save_progress_path + rf"\training_state_{epoch}.pth")

            with open(save_progress_path + "/progress.txt", "a") as progres_file:
                progres_file.write(message + "\n\n")

        if epoch > 0 and show_plots:
            plt.plot(train_losses, color='orange', label='train_loss')
            plt.plot(val_losses, color='green', label='val_loss')
            plt.title("Loss progression")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()

    if save_progress_path is not None:
        plt.plot(train_losses, color='orange', label='train_loss')
        plt.plot(val_losses, color='green', label='val_loss')
        plt.axvline(np.argmin(np.array(val_losses)), color='red')
        plt.title("Loss progression")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(save_progress_path + r"\loss_progress.png")

    print("Best validation loss: {:.4f}".format(np.min(np.array(val_losses))))


def train_epoch(train_loader, model, loss_fn, optimizer, log_interval):
    model.train()
    losses = []
    total_loss = 0
    start_time = time.time()

    print("Will sample from train_loader")
    for batch_idx, data in enumerate(train_loader):
        # print("batch_idx", batch_idx, "data", data)
        view_a, view_b, positive_pairs, negative_pairs, gt = reformat_data(data)
        graph_a = train_loader.dataset.graphs_data[view_a]
        graph_b = train_loader.dataset.graphs_data[view_b]

        optimizer.zero_grad()
        outputs_a, h_a = model(graph_a.x, graph_a.edge_index)
        outputs_b, h_b = model(graph_b.x, graph_b.edge_index)

        positive_outputs_a = outputs_a[positive_pairs[:, 0]]
        positive_outputs_b = outputs_b[positive_pairs[:, 1]]
        negative_outputs_a = outputs_a[negative_pairs[:, 0]]
        negative_outputs_b = outputs_b[negative_pairs[:, 1]]
        selected_outputs_a = torch.cat((positive_outputs_a, negative_outputs_a), 0)
        selected_outputs_b = torch.cat((positive_outputs_b, negative_outputs_b), 0)

        loss = loss_fn(selected_outputs_a, selected_outputs_b, gt)
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch_idx > 0 and batch_idx % log_interval == 0:
            elapsed_time = time.time() - start_time
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tElapsed time: {}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses), str(timedelta(seconds=elapsed_time)))
            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss


def test_epoch(val_loader, model, loss_fn):
    with torch.no_grad():
        model.eval()
        val_loss = 0
        for batch_idx, data in enumerate(val_loader):
            view_a, view_b, positive_pairs, negative_pairs, gt = reformat_data(data)
            graph_a = val_loader.dataset.graphs_data[view_a]
            graph_b = val_loader.dataset.graphs_data[view_b]

            outputs_a, h_a = model(graph_a.x, graph_a.edge_index)
            outputs_b, h_b = model(graph_b.x, graph_b.edge_index)

            positive_outputs_a = outputs_a[positive_pairs[:, 0]]
            positive_outputs_b = outputs_b[positive_pairs[:, 1]]
            negative_outputs_a = outputs_a[negative_pairs[:, 0]]
            negative_outputs_b = outputs_b[negative_pairs[:, 1]]
            selected_outputs_a = torch.cat((positive_outputs_a, negative_outputs_a), 0)
            selected_outputs_b = torch.cat((positive_outputs_b, negative_outputs_b), 0)

            loss = loss_fn(selected_outputs_a, selected_outputs_b, gt)
            val_loss += loss.item()

    return val_loss


def reformat_data(data):
    view_a = data[0][0][0]
    view_b = data[0][1][0]
    positive_pairs = torch.squeeze(data[1])
    negative_pairs = torch.squeeze(data[2])
    gt = torch.cat((torch.ones(positive_pairs.shape[0]), -torch.ones(negative_pairs.shape[0])), 0)
    return view_a, view_b, positive_pairs, negative_pairs, gt
