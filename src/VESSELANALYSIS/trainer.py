import torch
import numpy as np
import time
from datetime import timedelta
from matplotlib import pyplot as plt
import torch.nn.functional as F


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
        best_epoch = np.argmin(np.array(val_losses))
        plt.plot(train_losses, color='orange', label='train_loss')
        plt.plot(val_losses, color='green', label='val_loss')
        plt.axvline(best_epoch, color='red')
        plt.title("Loss progression")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(save_progress_path + r"\loss_progress.png")

        training_state = torch.load(save_progress_path + rf"\training_state_{best_epoch}.pth")
        model.load_state_dict(training_state["model"])

        plt.figure()
        plt.suptitle("Training set node feature vectors")
        graphs_data = train_loader.graphs_data if hasattr(train_loader, 'graphs_data') else train_loader.dataset.graphs_data
        for i in range(2):
            plt.subplot(1, 2, i+1)
            plt.title(f"{'Normalized' if i == 0 else 'Original'} feature vectors")
            for view, graph in graphs_data.items():
                outputs, h = model(graph.x, graph.edge_index)
                if i == 0:
                    outputs = F.normalize(outputs, dim=-1)
                numpy_outputs = outputs.detach().numpy()
                plt.scatter(numpy_outputs[:, 0], numpy_outputs[:, 1], label=view)
            plt.legend()
        plt.show()

        # Evaluation metric (top-1 and top-5)
        if hasattr(train_loader, 'all_positive_node_pairs'):
            for view_pair, positive_pairs in train_loader.all_positive_node_pairs.items():
                graph_a = graphs_data[view_pair[0]]
                graph_b = graphs_data[view_pair[1]]
                outputs_a, h = model(graph_a.x, graph_a.edge_index)
                outputs_b, h = model(graph_b.x, graph_b.edge_index)
                dist = torch.cdist(outputs_a, outputs_b)
                topk = torch.topk(-dist, k=5, dim=1)[1]
                top1 = 0
                top5 = 0
                for positive_pair in positive_pairs:
                    if positive_pair[1] in topk[positive_pair[0]]:
                        top5 += 1
                        if positive_pair[1] == topk[positive_pair[0]][0]:
                            top1 += 1
                top1_percent = int(top1 * 1000 / len(positive_pairs)) / 1000
                top5_percent = int(top5 * 1000 / len(positive_pairs)) / 1000
                print(f"{view_pair} results: top-1 = {top1} ({top1_percent}%), top-5 = {top5} ({top5_percent}%)")


    print("Best validation loss: {:.4f}".format(np.min(np.array(val_losses))))


def train_epoch(train_loader, model, loss_fn, optimizer, log_interval):
    model.train()
    losses = []
    total_loss = 0
    start_time = time.time()

    print("Will sample from train_loader")
    for batch_idx, data in enumerate(train_loader):
        # print("batch_idx", batch_idx, "data", data)
        old_training = len(data) == 3  # (views, positive pairs, negative pairs) while new training only has 2 views
        if old_training:
            view_a, view_b, positive_pairs, negative_pairs, gt = reformat_data(data)
            graph_a = train_loader.dataset.graphs_data[view_a]
            graph_b = train_loader.dataset.graphs_data[view_b]
        else:
            view_a, view_b, positive_pairs = reformat_data(data, train_loader.all_positive_node_pairs)
            graph_a = train_loader.graphs_data[view_a]
            graph_b = train_loader.graphs_data[view_b]

        optimizer.zero_grad()
        outputs_a, h_a = model(graph_a.x, graph_a.edge_index)
        outputs_b, h_b = model(graph_b.x, graph_b.edge_index)

        if old_training:
            positive_outputs_a = outputs_a[positive_pairs[:, 0]]
            positive_outputs_b = outputs_b[positive_pairs[:, 1]]
            negative_outputs_a = outputs_a[negative_pairs[:, 0]]
            negative_outputs_b = outputs_b[negative_pairs[:, 1]]
            selected_outputs_a = torch.cat((positive_outputs_a, negative_outputs_a), 0)
            selected_outputs_b = torch.cat((positive_outputs_b, negative_outputs_b), 0)
            loss = loss_fn(selected_outputs_a, selected_outputs_b, gt)
        else:
            loss = loss_fn(outputs_a, outputs_b, positive_pairs)

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
            old_training = len(data) == 3  # (views, positive pairs, negative pairs) while new training only has 2 views
            if old_training:
                view_a, view_b, positive_pairs, negative_pairs, gt = reformat_data(data)
                graph_a = val_loader.dataset.graphs_data[view_a]
                graph_b = val_loader.dataset.graphs_data[view_b]
            else:
                view_a, view_b, positive_pairs = reformat_data(data, val_loader.all_positive_node_pairs)
                graph_a = val_loader.graphs_data[view_a]
                graph_b = val_loader.graphs_data[view_b]

            outputs_a, h_a = model(graph_a.x, graph_a.edge_index)
            outputs_b, h_b = model(graph_b.x, graph_b.edge_index)

            if old_training:
                positive_outputs_a = outputs_a[positive_pairs[:, 0]]
                positive_outputs_b = outputs_b[positive_pairs[:, 1]]
                negative_outputs_a = outputs_a[negative_pairs[:, 0]]
                negative_outputs_b = outputs_b[negative_pairs[:, 1]]
                selected_outputs_a = torch.cat((positive_outputs_a, negative_outputs_a), 0)
                selected_outputs_b = torch.cat((positive_outputs_b, negative_outputs_b), 0)
                loss = loss_fn(selected_outputs_a, selected_outputs_b, gt)
            else:
                loss = loss_fn(outputs_a, outputs_b, positive_pairs)
            val_loss += loss.item()

    return val_loss


def reformat_data(data, all_positive_pairs=None):
    if all_positive_pairs is None:
        view_a = data[0][0][0]
        view_b = data[0][1][0]
        positive_pairs = torch.squeeze(data[1])
        negative_pairs = torch.squeeze(data[2])
        gt = torch.cat((torch.ones(positive_pairs.shape[0]), -torch.ones(negative_pairs.shape[0])), 0)
        return view_a, view_b, positive_pairs, negative_pairs, gt
    else:
        view_a = data[0][0]
        view_b = data[1][0]
        positive_pairs = all_positive_pairs[(view_a, view_b)]
        return view_a, view_b, positive_pairs
