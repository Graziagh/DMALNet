from sklearn.metrics import accuracy_score, precision_score, recall_score, \
     f1_score, classification_report, confusion_matrix, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import numpy as np
from models.mymodels import *
import time

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def cal_metrics(confusion_matrix):
    n_classes = confusion_matrix.shape[0]
    metrics_result = []
    for i in range(n_classes):     
        ALL = np.sum(confusion_matrix)
        
        TP = confusion_matrix[i, i]
        
        FP = np.sum(confusion_matrix[:, i]) - TP
        
        FN = np.sum(confusion_matrix[i, :]) - TP
        
        TN = ALL - TP - FP - FN
        # precision, sensitivity, f1_score, specificity
        metrics_result.append([TP/(TP+FP), TP/(TP+FN), (2*TP)/(2*TP+FN+FP), TN/(TN+FP)])

    metrics_np = np.array(metrics_result)
    res = metrics_np.mean(axis=0)
    return res


def evaluate_one(loader_train, model, criterion, device, logFile, epoch):
    batch_time = AverageMeter()
    # data_time = AverageMeter()
    losses = AverageMeter()
    # top1 = AverageMeter()
    # top5 = AverageMeter()

    train_labels = []
    train_preds = []

    # switch to evaluate mode
    model.eval()

    end = time.time()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader_train):
            # measure data loading time
            # data_time.update(time.time() - end)

            inputs = inputs.to(device)
            labels = labels.to(device)

            # compute output
            output, _, _, _, _, _ = model(inputs)
            loss = criterion(output, labels)

            # measure accuracy and record loss
            _, pred = output.data.topk(1, 1, True, True)
            preds = pred.squeeze(1)
            train_labels.extend(labels.detach().cpu().numpy())
            train_preds.extend(preds.detach().cpu().numpy())
            # ACC = accuracy_score(labels, preds)
            losses.update(loss.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                     epoch, i, len(loader_train), batch_time=batch_time, loss=losses))

        # print acc, p, r, f1
        matrix = confusion_matrix(train_labels, train_preds)
        accuracy = accuracy_score(train_labels, train_preds)
        precision, sensitivity, f1_score, specificity = cal_metrics(matrix)
        with open(logFile, "a") as f:
            print("[Epoch:{}] accuracy:{:.4f} precision:{:.4f} sensitivity:{:.4f} "
                  "f1_score:{:.4f} specificity:{:.4f}".format(epoch,
                                                              accuracy,
                                                              precision,
                                                              sensitivity,
                                                              f1_score,
                                                              specificity), file=f)

        print("[Epoch:{}] accuracy:{:.4f} precision:{:.4f} sensitivity:{:.4f} "
              "f1_score:{:.4f} specificity:{:.4f}".format(epoch,
                                                          accuracy,
                                                          precision,
                                                          sensitivity,
                                                          f1_score,
                                                          specificity))

    return accuracy


def evaluate_two(loader_train, model_first, model_temp, model_second, criterion, device, logFile, epoch):
    batch_time = AverageMeter()
    # data_time = AverageMeter()
    losses = AverageMeter()
    # top1 = AverageMeter()
    # top5 = AverageMeter()

    train_labels = []
    train_preds = []

    # switch to evaluate mode
    model_first.eval()
    model_temp.eval()
    model_second.eval()

    end = time.time()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader_train):
            # measure data loading time
            # data_time.update(time.time() - end)

            inputs = inputs.to(device)
            labels = labels.to(device)

            # compute output
            output, _, op1, op2, op3, op4 = model_first(inputs)
            # new_inputs = build_img2(model_temp, op1)
            new_inputs = build_img(model_temp, op1, op2, op3)
            # new_inputs = build_img(model_temp, op1, op2, op4)
            new_inputs = new_inputs.to(device)
            output2, _, _, _, _, _ = model_second(new_inputs)
            loss = criterion(output2, labels)

            # measure accuracy and record loss
            _, pred = output2.data.topk(1, 1, True, True)
            preds = pred.squeeze(1)
            train_labels.extend(labels.detach().cpu().numpy())
            train_preds.extend(preds.detach().cpu().numpy())
            # ACC = accuracy_score(labels, preds)
            losses.update(loss.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                print('[Train]Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                     epoch, i, len(loader_train), batch_time=batch_time, loss=losses))

        # print acc, p, r, f1
        matrix = confusion_matrix(train_labels, train_preds)
        accuracy = accuracy_score(train_labels, train_preds)
        precision, sensitivity, f1_score, specificity = cal_metrics(matrix)
        with open(logFile, "a") as f:
            print("[Epoch:{}] accuracy:{:.4f} precision:{:.4f} sensitivity:{:.4f} "
                  "f1_score:{:.4f} specificity:{:.4f}".format(epoch,
                                                              accuracy,
                                                              precision,
                                                              sensitivity,
                                                              f1_score,
                                                              specificity), file=f)

        print("[Epoch:{}] accuracy:{:.4f} precision:{:.4f} sensitivity:{:.4f} "
              "f1_score:{:.4f} specificity:{:.4f}".format(epoch,
                                                          accuracy,
                                                          precision,
                                                          sensitivity,
                                                          f1_score,
                                                          specificity))

    return accuracy



def evaluate_fusion(loader_train, model_first, model_temp, model_second, model_fusion, criterion, device, logFile, epoch):
    batch_time = AverageMeter()
    # data_time = AverageMeter()
    losses = AverageMeter()
    # top1 = AverageMeter()
    # top5 = AverageMeter()

    train_labels = []
    train_preds = []

    # switch to evaluate mode
    model_first.eval()
    model_temp.eval()
    model_second.eval()
    model_fusion.eval()

    end = time.time()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader_train):
            # measure data loading time
            # data_time.update(time.time() - end)

            inputs = inputs.to(device)
            labels = labels.to(device)

            # compute output
            _, out_pool, op1, op2, op3, op4 = model_first(inputs)
            # new_inputs = build_img2(model_temp, op1)
            new_inputs = build_img(model_temp, op1, op2, op3)
            # new_inputs = build_img(model_temp, op1, op2, op4)
            new_inputs = new_inputs.to(device)
            _, out_pool2, _, _, _, _ = model_second(new_inputs)
            output3 = model_fusion(out_pool, out_pool2)
            loss = criterion(output3, labels)

            # measure accuracy and record loss
            _, pred = output3.data.topk(1, 1, True, True)
            preds = pred.squeeze(1)
            train_labels.extend(labels.detach().cpu().numpy())
            train_preds.extend(preds.detach().cpu().numpy())
            # ACC = accuracy_score(labels, preds)
            losses.update(loss.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                     epoch, i, len(loader_train), batch_time=batch_time, loss=losses))

        # print acc, p, r, f1
        matrix = confusion_matrix(train_labels, train_preds)
        accuracy = accuracy_score(train_labels, train_preds)
        precision, sensitivity, f1_score, specificity = cal_metrics(matrix)
        with open(logFile, "a") as f:
            print("[Epoch:{}] accuracy:{:.4f} precision:{:.4f} sensitivity:{:.4f} "
                  "f1_score:{:.4f} specificity:{:.4f}".format(epoch,
                                                              accuracy,
                                                              precision,
                                                              sensitivity,
                                                              f1_score,
                                                              specificity), file=f)

        print("[Epoch:{}] accuracy:{:.4f} precision:{:.4f} sensitivity:{:.4f} "
              "f1_score:{:.4f} specificity:{:.4f}".format(epoch,
                                                          accuracy,
                                                          precision,
                                                          sensitivity,
                                                          f1_score,
                                                          specificity))

    return accuracy



def train_first(loader_train, model, criterion, optimizer, device, logFile, epoch):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # top1 = AverageMeter()
    # top5 = AverageMeter()

    train_labels = []
    train_preds = []

    # switch to train mode
    model.train()

    end = time.time()

    for i, (inputs, labels) in enumerate(loader_train):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs = inputs.to(device)
        labels = labels.to(device)

        # compute output
        output, _, _, _, _, _ = model(inputs)
        # print(labels.dtype)
        # print(output.dtype)
        loss = criterion(output, labels)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        _, pred = output.data.topk(1, 1, True, True)
        preds = pred.squeeze(1)
        train_labels.extend(labels.detach().cpu().numpy())
        train_preds.extend(preds.detach().cpu().numpy())
        # ACC = accuracy_score(labels, preds)
        losses.update(loss.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(loader_train), batch_time=batch_time, data_time=data_time, loss=losses))

    # print acc, p, r, f1
    matrix = confusion_matrix(train_labels, train_preds)
    accuracy = accuracy_score(train_labels, train_preds)
    precision, sensitivity, f1_score, specificity = cal_metrics(matrix)
    with open(logFile, "a") as f:
        print("[Epoch:{}]  accuracy:{:.4f} precision:{:.4f} sensitivity:{:.4f} "
              "f1_score:{:.4f} specificity:{:.4f}".format(epoch,
                                                          accuracy,
                                                          precision,
                                                          sensitivity,
                                                          f1_score,
                                                          specificity), file=f)

    print("[Epoch:{}]  accuracy:{:.4f} precision:{:.4f} sensitivity:{:.4f} "
          "f1_score:{:.4f} specificity:{:.4f}".format(epoch,
                                                      accuracy,
                                                      precision,
                                                      sensitivity,
                                                      f1_score,
                                                      specificity))




def train_second(loader_train, model_first, model_temp, model_second, criterion, temp_optimizer, second_optimizer,
                 device, logFile, epoch):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # top1 = AverageMeter()
    # top5 = AverageMeter()

    train_labels = []
    train_preds = []

    # switch to train mode
    model_first.eval()
    model_second.train()
    model_temp.train()

    end = time.time()

    for i, (inputs, labels) in enumerate(loader_train):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs = inputs.to(device)
        labels = labels.to(device)

        # compute output
        with torch.no_grad():
            output, _, op1, op2, op3, op4 = model_first(inputs)
        # new_inputs = build_img2(model_temp, op1)
        new_inputs = build_img(model_temp, op1, op2, op3)
        # new_inputs = build_img(model_temp, op1, op2, op4)
        new_inputs = new_inputs.to(device)
        output2, _, _, _, _, _ = model_second(new_inputs)
        loss = criterion(output2, labels)

        # compute gradient and do Adam step
        second_optimizer.zero_grad()
        temp_optimizer.zero_grad()
        loss.backward()
        second_optimizer.step()
        temp_optimizer.step()

        # measure accuracy and record loss
        _, pred = output2.data.topk(1, 1, True, True)
        preds = pred.squeeze(1)
        train_labels.extend(labels.detach().cpu().numpy())
        train_preds.extend(preds.detach().cpu().numpy())
        # ACC = accuracy_score(labels, preds)
        losses.update(loss.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(loader_train), batch_time=batch_time, data_time=data_time, loss=losses))

    # print acc, p, r, f1
    matrix = confusion_matrix(train_labels, train_preds)
    accuracy = accuracy_score(train_labels, train_preds)
    precision, sensitivity, f1_score, specificity = cal_metrics(matrix)
    with open(logFile, "a") as f:
        print("[Epoch:{}]  accuracy:{:.4f} precision:{:.4f} sensitivity:{:.4f} "
              "f1_score:{:.4f} specificity:{:.4f}".format(epoch,
                                                          accuracy,
                                                          precision,
                                                          sensitivity,
                                                          f1_score,
                                                          specificity), file=f)

    print("[Epoch:{}]  accuracy:{:.4f} precision:{:.4f} sensitivity:{:.4f} "
          "f1_score:{:.4f} specificity:{:.4f}".format(epoch,
                                                      accuracy,
                                                      precision,
                                                      sensitivity,
                                                      f1_score,
                                                      specificity))



def train_fusion(loader_train, model_first, model_temp, model_second, model_fusion, criterion, fusion_optimizer,
                 device, logFile, epoch):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # top1 = AverageMeter()
    # top5 = AverageMeter()

    train_labels = []
    train_preds = []

    # switch to train mode
    model_first.eval()
    model_second.eval()
    model_temp.eval()
    model_fusion.train()

    end = time.time()

    for i, (inputs, labels) in enumerate(loader_train):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs = inputs.to(device)
        labels = labels.to(device)

        # compute output
        with torch.no_grad():
            _, out_pool, op1, op2, op3, op4 = model_first(inputs)
            # new_inputs = build_img2(model_temp, op1)
            new_inputs = build_img(model_temp, op1, op2, op3)
            # new_inputs = build_img(model_temp, op1, op2, op4)
            new_inputs = new_inputs.to(device)
            _, out_pool2, _, _, _, _ = model_second(new_inputs)
        output3 = model_fusion(out_pool, out_pool2)
        loss = criterion(output3, labels)

        # compute gradient and do Adam step
        fusion_optimizer.zero_grad()
        loss.backward()
        fusion_optimizer.step()

        # measure accuracy and record loss
        _, pred = output3.data.topk(1, 1, True, True)
        preds = pred.squeeze(1)
        train_labels.extend(labels.detach().cpu().numpy())
        train_preds.extend(preds.detach().cpu().numpy())
        # ACC = accuracy_score(labels, preds)
        losses.update(loss.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(loader_train), batch_time=batch_time, data_time=data_time, loss=losses))

    # print acc, p, r, f1
    matrix = confusion_matrix(train_labels, train_preds)
    accuracy = accuracy_score(train_labels, train_preds)
    precision, sensitivity, f1_score, specificity = cal_metrics(matrix)
    with open(logFile, "a") as f:
        print("[Epoch:{}]  accuracy:{:.4f} precision:{:.4f} sensitivity:{:.4f} "
              "f1_score:{:.4f} specificity:{:.4f}".format(epoch,
                                                          accuracy,
                                                          precision,
                                                          sensitivity,
                                                          f1_score,
                                                          specificity), file=f)

    print("[Epoch:{}]  accuracy:{:.4f} precision:{:.4f} sensitivity:{:.4f} "
          "f1_score:{:.4f} specificity:{:.4f}".format(epoch,
                                                      accuracy,
                                                      precision,
                                                      sensitivity,
                                                      f1_score,
                                                      specificity))




# def train_last(loader_train, model_first, model_temp, model_second, model_fusion, criterion, first_optimizer,
#                temp_optimizer, second_optimizer, fusion_optimizer, device, logFile, epoch):
#
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     losses = AverageMeter()
#     # top1 = AverageMeter()
#     # top5 = AverageMeter()
#
#     train_labels = []
#     train_preds = []
#
#     # switch to train mode
#     model_first.train()
#     model_second.train()
#     model_temp.train()
#     model_fusion.train()
#
#     end = time.time()
#
#     for i, (inputs, labels) in enumerate(loader_train):
#         # measure data loading time
#         data_time.update(time.time() - end)
#
#         inputs = inputs.to(device)
#         labels = labels.to(device)
#
#         # compute output
#         _, out_pool, op1, op2, op3, op4 = model_first(inputs)
#         new_inputs = build_img(model_temp, op1, op2, op3)
#         new_inputs = new_inputs.to(device)
#         _, out_pool2 = model_second(new_inputs)
#         output3 = model_fusion(out_pool, out_pool2)
#         loss = criterion(output3, labels)
#
#         # compute gradient and do Adam step
#         first_optimizer.zero_grad()
#         temp_optimizer.zero_grad()
#         second_optimizer.zero_grad()
#         fusion_optimizer.zero_grad()
#         loss.backward()
#         first_optimizer.step()
#         temp_optimizer.step()
#         second_optimizer.step()
#         fusion_optimizer.step()
#
#
#
#         # measure accuracy and record loss
#         _, pred = output3.data.topk(1, 1, True, True)
#         preds = pred.squeeze(1)
#         train_labels.extend(labels.detach().cpu().numpy())
#         train_preds.extend(preds.detach().cpu().numpy())
#         # ACC = accuracy_score(labels, preds)
#         losses.update(loss.item(), inputs.size(0))
#
#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()
#
#         if i % 10 == 0:
#             print('Epoch: [{0}][{1}/{2}]\t'
#                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
#                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
#                 epoch, i, len(loader_train), batch_time=batch_time, data_time=data_time, loss=losses))
#
#     # print acc, p, r, f1
#     matrix = confusion_matrix(train_labels, train_preds)
#     accuracy = accuracy_score(train_labels, train_preds)
#     precision, sensitivity, f1_score, specificity = cal_metrics(matrix)
#     with open(logFile, "a") as f:
#         print("[Epoch:{}]  accuracy:{:.4f} precision:{:.4f} sensitivity:{:.4f} "
#               "f1_score:{:.4f} specificity:{:.4f}".format(epoch,
#                                                           accuracy,
#                                                           precision,
#                                                           sensitivity,
#                                                           f1_score,
#                                                           specificity), file=f)
#
#     print("[Epoch:{}]  accuracy:{:.4f} precision:{:.4f} sensitivity:{:.4f} "
#           "f1_score:{:.4f} specificity:{:.4f}".format(epoch,
#                                                       accuracy,
#                                                       precision,
#                                                       sensitivity,
#                                                       f1_score,
#                                                       specificity))

def evaluate_other_model(loader_train, model, criterion, device, logFile, epoch):
    batch_time = AverageMeter()
    # data_time = AverageMeter()
    losses = AverageMeter()
    # top1 = AverageMeter()
    # top5 = AverageMeter()

    train_labels = []
    train_preds = []

    # switch to evaluate mode
    model.eval()

    end = time.time()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader_train):
            # measure data loading time
            # data_time.update(time.time() - end)

            inputs = inputs.to(device)
            labels = labels.to(device)

            # compute output
            output, _, _, _, _, _ = model(inputs)
            # output = output.logits
            loss = criterion(output, labels)

            # measure accuracy and record loss
            _, pred = output.data.topk(1, 1, True, True)
            preds = pred.squeeze(1)
            train_labels.extend(labels.detach().cpu().numpy())
            train_preds.extend(preds.detach().cpu().numpy())
            # ACC = accuracy_score(labels, preds)
            losses.update(loss.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                     epoch, i, len(loader_train), batch_time=batch_time, loss=losses))

        # print acc, p, r, f1
        matrix = confusion_matrix(train_labels, train_preds)
        accuracy = accuracy_score(train_labels, train_preds)
        precision, sensitivity, f1_score, specificity = cal_metrics(matrix)
        with open(logFile, "a") as f:
            print("[Epoch:{}] accuracy:{:.4f} precision:{:.4f} sensitivity:{:.4f} "
                  "f1_score:{:.4f} specificity:{:.4f}".format(epoch,
                                                              accuracy,
                                                              precision,
                                                              sensitivity,
                                                              f1_score,
                                                              specificity), file=f)

        print("[Epoch:{}] accuracy:{:.4f} precision:{:.4f} sensitivity:{:.4f} "
              "f1_score:{:.4f} specificity:{:.4f}".format(epoch,
                                                          accuracy,
                                                          precision,
                                                          sensitivity,
                                                          f1_score,
                                                          specificity))

    return accuracy




def train_other_model(loader_train, model, criterion, optimizer, device, logFile, epoch):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # top1 = AverageMeter()
    # top5 = AverageMeter()

    train_labels = []
    train_preds = []

    # switch to train mode
    model.train()

    end = time.time()

    for i, (inputs, labels) in enumerate(loader_train):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs = inputs.to(device)
        labels = labels.to(device)

        # compute output
        output, _, _, _, _, _ = model(inputs)
        # print(labels.dtype)
        # print(output.dtype)
        # output = output.logits
        loss = criterion(output, labels)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        _, pred = output.data.topk(1, 1, True, True)
        preds = pred.squeeze(1)
        train_labels.extend(labels.detach().cpu().numpy())
        train_preds.extend(preds.detach().cpu().numpy())
        # ACC = accuracy_score(labels, preds)
        losses.update(loss.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(loader_train), batch_time=batch_time, data_time=data_time, loss=losses))

    # print acc, p, r, f1
    matrix = confusion_matrix(train_labels, train_preds)
    accuracy = accuracy_score(train_labels, train_preds)
    precision, sensitivity, f1_score, specificity = cal_metrics(matrix)
    with open(logFile, "a") as f:
        print("[Epoch:{}]  accuracy:{:.4f} precision:{:.4f} sensitivity:{:.4f} "
              "f1_score:{:.4f} specificity:{:.4f}".format(epoch,
                                                          accuracy,
                                                          precision,
                                                          sensitivity,
                                                          f1_score,
                                                          specificity), file=f)

    print("[Epoch:{}]  accuracy:{:.4f} precision:{:.4f} sensitivity:{:.4f} "
          "f1_score:{:.4f} specificity:{:.4f}".format(epoch,
                                                      accuracy,
                                                      precision,
                                                      sensitivity,
                                                      f1_score,
                                                      specificity))