# =============== ADDED BY CS10-2: Minjun Jung BEGIN *************** #

from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import itertools
import copy
import logging
from datetime import datetime

# https://datascience.stackexchange.com/questions/78449/plotting-scikit-learn-confusion-matrix-returns-no-values-in-the-last-class
# (↑) Original source of the confusion matrix.
"""
    However, our project necessitated calculation of F1 score as the final metrics,
    and F1 score is dependent on Precision and Recall value.
    Therefore to get a better idea about specifically which pred-actual labels our model is failing at
    I made modifications from the above code to give results as:
        percentage over sum(preds) -> Precision    [when mode="v"] for "vertical"
        percentage over sum(trues) -> Recall       [when mode="h"] for "horizontal"

    Although technically they are not exactly Precisions and Recalls.
    Only values at diagonal positions correspond to the real definition of Precs & Recs.
"""
def getConfusionMatrix(y_true, y_pred, labels=None, mode="", figsize=(12, 12), title="", show=False, dir="./"):
    cm = confusion_matrix(y_true, y_pred)

    # Due to unbalanced distribution of labels simply showing counts is not very comprehensive.
    # Instead it'll calculate percentage over sum(row) or sum(col).
    pro_cm = cm[:, :].astype(np.float64)
    if mode.lower() == "v":
        denoms = np.array([np.sum(pro_cm[:, j]) for j in range(pro_cm.shape[1])])
    elif mode.lower() == "h":
        denoms = np.array([np.sum(pro_cm[i, :]) for i in range(pro_cm.shape[0])])
    else:
        return cm  # For any other options just return raw confusion matrix.

    # Iterate cells and divide them by sum of the column or row.
    if mode.lower() == "v":
        for i in range(pro_cm.shape[0]):
            for j in range(pro_cm.shape[1]):
                pro_cm[i, j] = (pro_cm[i, j] * 100 / denoms[j])
    else:  # Only remaining other option is "h"
        for i in range(pro_cm.shape[0]):
            for j in range(pro_cm.shape[1]):
                pro_cm[i, j] = (pro_cm[i, j] * 100 / denoms[i])

    # To prevent 0 / 0 (div by 0) occurring when either sum(preds) or sum(trues) is 0
    orig_pro_cm = copy.deepcopy(pro_cm)  # Back up pro_cm so that we can deal with NaN Prec, Rec more wisely.
    pro_cm[np.isnan(pro_cm)] = 0

    # pyplot settings
    ticks = np.arange(len(labels))
    diag_type = "Prec =\n" if mode.lower() == "v" else "Rec =\n"
    fmt = lambda x: format(x, ".2f") + " %"  # Show percentage until 2 decimal point in cells
    thresh = 50  # White letters instead of black when cell gets darker than 50%

    # Plot size & title
    plt.figure(figsize=figsize)  # (n, n) => n*100 x n*100 pixels
    plt.imshow(pro_cm, interpolation='nearest', cmap=plt.cm.Greens)
    title += "\nsum(col) = 100%" if mode.lower() == "v" else "\nsum(row) = 100%"
    plt.title(title)

    # Plot axises config
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    # If label names are too long: in our use case if they contain '_' supposed to
    # act as word separator then write each words separated by '_' into new lines.
    labels = [name.replace("_", "\n") for name in labels.values()]
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)
    plt.colorbar()
    plt.clim(0, 100)  # Color bar fixed to 0 to 100 representing percentage.

    # Populating cells
    for (i, j) in itertools.product(range(pro_cm.shape[0]), range(pro_cm.shape[1])):
        if i == j:
            plt.text(j, i, diag_type + fmt(pro_cm[i, j]), ha="center", va="center", fontweight="bold",
                     color='white' if pro_cm[i, j] > thresh else 'black')
        else:
            plt.text(j, i, fmt(pro_cm[i, j]), ha="center", va="center",
                     color='white' if pro_cm[i, j] > thresh else 'black')

    file_title = title.split("\n")
    # Put this image into its own designated time-stamped folder too.
    plt.savefig(dir + "/" + datetime.now().strftime('%Y-%m-%d_%H%M%S') + " " +\
        file_title[0] + " [" + file_title[1] + "].png")

    # plt.show() has to come after file saving, otherwise the saved file will be blank.
    if show:
        matplotlib.rcParams['interactive'] == True  # Open new window to show plt
        plt.show()

    plt.close()
    return cm, orig_pro_cm.diagonal()  # Diagonals are either Precisions or Recalls.

def getPerformance(y_true, y_pred, labels, title="", dir="./"):
    # Calculate Accuracy
    cm = confusion_matrix(y_true, y_pred)
    acc = sum(cm.diagonal()) / len(y_true) * 100  # Sync to percentage

    # Calculate Precs & Recs
    prec = getConfusionMatrix(y_true, y_pred, labels=labels, mode="v", title=title, dir=dir)[1]
    rec = getConfusionMatrix(y_true, y_pred, labels=labels, mode="h", title=title, dir=dir)[1]

    # Calculate F1 using numpy.ndarray features
    f1 = (2 * prec * rec) / (prec + rec)

    # Print out the results - in more nicely formatted table form
    logging.info("********** Performance Report **********")
    max_label_len = max(map(lambda x: len(x), labels.values()))
    max_label_len = max(max_label_len, len("Label"))  # In case label names are short
    heading = (" {:^" + str(max_label_len) + "} |  Prec  |  Rec   |   F1   ").format("Label")
    logging.info( heading)
    logging.info("-" * len(heading))

    for key, value in labels.items():
        logging.info((" {:^" + str(max_label_len) + "} | {:>6} | {:>6} | {:>6} ").format(
            value, format(prec[key], ".2f"), format(rec[key], ".2f"), format(f1[key], ".2f")
        ))

    logging.info("-" * len(heading))
    logging.info((" {:^" + str(max_label_len) + "} | {:>6} | {:>6} | {:>6} ").format(
        "Mean", format(np.nanmean(prec), ".2f"), format(np.nanmean(rec), ".2f"), format(np.nanmean(f1), ".2f")
        # F1 can have nan, but using np.nanmean() resolves this.
    ))

    logging.info("Accuracy: " + format(acc, ".2f") + " %")
    logging.info("********** Report Finished **********")

    return acc, prec, rec, f1

def printEpochWise(means):
    epoch, prec, rec, f1 = means["epoch"], means["prec"], means["rec"], means["f1"]
    logging.info(f"<<<<<<<<<< Mean Epoch Performance >>>>>>>>>>")
    logging.info(" Epoch |  Prec  |  Rec   |   F1   ")
    logging.info("----------------------------------")

    for i in range(len(epoch)):
        logging.info((" {:>5} | {:>6} | {:>6} | {:>6} ").format(
            epoch[i], format(prec[i], ".2f"), format(rec[i], ".2f"), format(f1[i], ".2f")
        ))

    logging.info(f"<<<<<<<<<< Epoch Report Finished >>>>>>>>>>")


import torch

def convert_to_one_hot(targets, num_classes, on_value=1.0, off_value=0.0):
    targets = targets.long().view(-1, 1)
    return torch.full(
        (targets.size()[0], num_classes), off_value, device=targets.device
    ).scatter_(1, targets, on_value)

@torch.no_grad()
def predict_val(val_loader, curr_model, cfg):
    y_true, y_pred = [], []
    model = curr_model
    model.eval()  # Set the model to eval mode freezing weights

    for inputs, labels in val_loader:
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            labels = labels.cuda()
            inputs = inputs.cuda(non_blocking=True)

        # Make predictions
        labels = torch.unsqueeze(convert_to_one_hot(labels,cfg.MODEL.NUM_CLASSES),dim=1)
        preds = model(inputs)
        labels = torch.squeeze(torch.mean(labels,dim=1,keepdim=True),dim=1)
        preds = torch.squeeze(torch.mean(preds,dim=1,keepdim=True),dim=1)
        clean_labels = torch.argmax(labels,dim=1)
        clean_preds = torch.argmax(preds,dim=1)

        # Append to result lists
        y_true += list(clean_labels.detach().cpu().numpy())
        y_pred += list(clean_preds.detach().cpu().numpy())

    return y_true, y_pred

def plotElbow(metrics_dict , x_label="epochs", y_label="metrics",title="Elbow Chart",dir="./"):
    print(f"metrics_dict: {metrics_dict}")
    fig, ax = plt.subplots( nrows=1, ncols=1,figsize=(16,8) )  # create figure & 1 axis
    f1_line, = ax.plot(metrics_dict["epoch"], metrics_dict["f1"], color="red" )
    f1_line.set_label("f1")
    prec_line, = ax.plot(metrics_dict["epoch"], metrics_dict["prec"], color="green" )
    prec_line.set_label("precision")
    rec_line, = ax.plot(metrics_dict["epoch"], metrics_dict["rec"],  color="blue" )
    rec_line.set_label("recall")

    ax.legend()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    # plt.plot([1,2,3],[1,4,5])
    # plt.figure(figsize=(16,8))
    # plt.plot(metrics_dict["epoch"], metrics_dict["f1"], label="f1", color="red" )
    # plt.plot(metrics_dict["epoch"], metrics_dict["prec"], label="precision", color="red" )
    # plt.plot(metrics_dict["epoch"], metrics_dict["rec"], label="recall", color="red" )
    # plt.legend()
    # fig = plt.gcf()
    fig.savefig(f"{dir}/{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{title.replace(' ','_')}.png")
    # plt.close()

# =============== ADDED BY CS10-2: Minjun Jung END =============== #


if __name__ == "__main__":
    trues = [0, 1, 2, 1, 2, 0, 1, 2, 1, 1, 0, 0, 2, 1]
    preds = [0, 2, 2, 1, 2, 0, 0, 2, 1, 1, 1, 0, 2, 1]

    # Make sure that the keys of dict is in ascending order.
    # otherwise, it will mess up confusion_matrix() function's result.
    labels = {0: "hoho", 1: "haha", 2: "huhu"}

    getPerformance(trues, preds, labels, title="Test Plot")
    pass
