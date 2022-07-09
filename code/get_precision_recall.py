import argparse
import os

import torch
from torchvision import transforms
from retinanet import model
from tqdm import tqdm
from retinanet.dataloader import CSVDataset, Resizer, Normalizer
import numpy as  np
assert torch.__version__.split('.')[0] == '1'
print('CUDA available: {}'.format(torch.cuda.is_available()))
import matplotlib.pyplot as plt

def compute_overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def _get_detections(dataset, retinanet, score_threshold=0.05, max_detections=100, save_path=None):
    """ Get the detections from the retinanet using the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]
    # Arguments
        dataset         : The generator used to run images through the retinanet.
        retinanet           : The retinanet to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(dataset.num_classes())] for j in range(len(dataset))]

    retinanet.eval()
    nms_threshold = 0.8
    mode = 'min'
    print("________________"+"nms_threshold = "+str(nms_threshold)+" mode = "+mode+"_______________")
    with torch.no_grad():

        for index in tqdm(range(len(dataset))):
            data = dataset[index]
            scale = data['scale']

            # run network
            if torch.cuda.is_available():
                scores, labels, boxes = retinanet(data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
            else:
                scores, labels, boxes = retinanet(data['img'].permute(2, 0, 1).float().unsqueeze(dim=0))
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
            boxes = boxes.cpu().numpy()
            # select indices which have a score above the threshold
            indices = np.where(scores > score_threshold)[0]
            indices = crop_first_nms(boxes,labels,indices,nms_threshold=nms_threshold,mode=mode)

            # correct boxes for image scale
            boxes /= scale
            if indices.shape[0] > 0:
                # select those scores
                scores = scores[indices]

                # find the order with which to sort the scores
                scores_sort = np.argsort(-scores)[:max_detections]

                # select detections
                image_boxes = boxes[indices[scores_sort], :]
                image_scores = scores[scores_sort]
                image_labels = labels[indices[scores_sort]]
                image_detections = np.concatenate(
                    [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

                # copy detections to all_detections
                for label in range(dataset.num_classes()):
                    all_detections[index][label] = image_detections[image_detections[:, -1] == label, :-1]
            else:
                # copy detections to all_detections
                for label in range(dataset.num_classes()):
                    all_detections[index][label] = np.zeros((0, 5))

            # print('{}/{}'.format(index + 1, len(dataset)), end='\r')

    return all_detections

def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]
    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(len(generator))]

    for i in range(len(generator)):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()

        print('{}/{}'.format(i + 1, len(generator)), end='\r')

    return all_annotations

def crop_first_nms(bboxes, classification,inds,nms_threshold=0.5, mode='union'):
    pass
    weed_inds = np.array([i for i in inds if classification[i] ==0], dtype=int)
    crop_inds = np.array([i for i in inds if classification[i] ==1],dtype=int)
    crop_x1 = bboxes[crop_inds, 0]
    crop_y1 = bboxes[crop_inds, 1]
    crop_x2 = bboxes[crop_inds, 2]
    crop_y2 = bboxes[crop_inds, 3]

    weed_x1 = bboxes[weed_inds, 0]
    weed_y1 = bboxes[weed_inds, 1]
    weed_x2 = bboxes[weed_inds, 2]
    weed_y2 = bboxes[weed_inds, 3]

    crop_boxes = bboxes[crop_inds]
    weed_boxes = bboxes[weed_inds]

    crop_areas = (crop_x2 - crop_x1 + 1) * (crop_y2 - crop_y1 + 1)
    weed_areas = (weed_x2-weed_x1+1) * (weed_y2-weed_y1+1)
    keep = np.array([],dtype=int)
    for i in range(len(crop_inds)):
        keep = np.append(keep,crop_inds[i])
        if len(weed_inds)==0:
            continue

        # xx1 = weed_x1[weed_inds].clamp(min= crop_x1[i])
        # yy1 = weed_y1[weed_inds].clamp(min=crop_y1[i])
        #
        # xx2 = weed_x2[weed_inds].clamp(max=crop_x2[i])
        # yy2 = weed_y2[weed_inds].clamp(max=crop_y2[i])
        xx1=np.maximum(crop_x1[i],weed_x1[weed_inds])
        yy1=np.maximum(crop_y1[i],weed_y1[weed_inds])


        xx2 = np.minimum(crop_x2[i], weed_x2[weed_inds])
        yy2 = np.minimum(crop_y2[i], weed_y2[weed_inds])
        w= np.maximum(xx2-xx1+1,0)
        h = np.maximum((yy2 - yy1 + 1),0)
        # w = (xx2 - xx1 + 1).clamp(min=0)
        # h = (yy2 - yy1 + 1).clamp(min=0)
        inter = w * h
        if mode == 'union':
            ovr = inter / (crop_areas[i] + weed_areas[weed_inds] - inter)
        elif mode == 'min':
            ovr = inter / np.maximum(crop_areas[i],weed_areas[weed_inds])
        ovr_idxs = np.where(ovr<=nms_threshold)[0]#(ovr>threshold).nonzero().squeeze().cpu()
        weed_inds = weed_inds[ovr_idxs]

    new_inds = np.append(weed_inds,keep)
    return new_inds



def get_pr( generator,
        retinanet,
        iou_threshold=0.5,
        score_threshold=0.5,
        max_detections=100,
        save_path=None):
    pass

    all_detections = _get_detections(generator, retinanet, score_threshold=score_threshold,max_detections=max_detections, save_path=save_path)
    all_annotations = _get_annotations(generator)

    average_precisions = {}
    precisions = {}
    recalls = {}

    # 计算每一类的FP和TP
    final_scores = {}
    for label in range(generator.num_classes()):
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0
        score05_idx = 0
        # 遍历每一张图片的预测结果
        for i in range(len(generator)):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            num_annotations += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                # 如果无gt，则为fp
                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue
                # 计算交并比
                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        recalls[label] = recall
        precisions[label] = precision

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

        scores = np.sort(scores)
        scores = scores[::-1]
        final_scores[label] = scores
        # scores = scores.tolist()
        for i, score in enumerate(scores):
            if score <= 0.5:
                score05_idx = max(i - 1, 0)
                print(i - 1)
                break

    print('\n evaluation precision:')
    mAP = 0.0
    for label in range(generator.num_classes()):
        if len(precisions[label]) == 0 or len(recalls[label]) == 0:
            continue
        label_name = generator.label_to_name(label)
        mAP += average_precisions[label][0]
        print('{}: {}'.format(label_name, average_precisions[label][0]))
        print("Precision: ", precisions[label][-1])
        print("Recall: ", recalls[label][-1])

    if save_path != None:
        import pandas as pd
        class_names = ['weed', 'sugar beet']
        for label in range(generator.num_classes()):
            plt.plot(recalls[label], precisions[label], label=class_names[label])
            pr_dic = {class_names[label] + 'rec': recalls[label], class_names[label] + 'prec': precisions[label],
                      class_names[label] + 'score': final_scores[label]}
            df = pd.DataFrame(pr_dic)
            df.to_csv(save_path + "/" + class_names[label] + "_PR_Curve.csv")
            # scores_dic = {class_names[label]+'score':final_scores[label]}
            # df = pd.DataFrame(scores_dic)
            # df.to_csv(save_path + "/" + class_names[label] + "_scores.csv")
        # naming the x axis
        plt.xlabel('Recall')
        # naming the y axis
        plt.ylabel('Precision')
        # giving a title to my graph
        plt.title('Precision Recall curve')
        plt.legend(loc="upper right", fontsize='medium')
        # function to show the plot
        plt.savefig(os.path.join(save_path, 'precision_recall.jpg'))
        plt.close()
    mAP = mAP / generator.num_classes()
    print("{}:{}".format("mAP", str(mAP)))
    return average_precisions, precisions, recalls

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--csv_annotations_path', help='Path to CSV annotations',default='./dataset/test_.csv')
    parser.add_argument('--model_path', help='Path to model', type=str,default=r'F:\sugarbeer_weed_detection\Ablation\visualize\csv_retinanet_8.pt')
    parser.add_argument('--images_path',help='Path to images directory',type=str)
    parser.add_argument('--class_list_path',help='Path to classlist csv',type=str,default='./dataset/class_list.csv')
    parser.add_argument('--iou_threshold',help='IOU threshold used for evaluation',type=str, default='0.5')
    parser.add_argument('--score_threshold', help='score threshold used for evaluation', type=str, default='0.5')
    parser = parser.parse_args(args)

    #dataset_val = CocoDataset(parser.coco_path, set_name='val2017',transform=transforms.Compose([Normalizer(), Resizer()]))
    dataset_val = CSVDataset(parser.csv_annotations_path,parser.class_list_path,transform=transforms.Compose([Normalizer(), Resizer()]))
    # Create the model
    #retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)
    retinanet=torch.load(parser.model_path)

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        #retinanet.load_state_dict(torch.load(parser.model_path))
        retinanet = torch.nn.DataParallel(retinanet).cuda()

    else:
        retinanet.load_state_dict(torch.load(parser.model_path))
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = False
    retinanet.eval()
    retinanet.module.freeze_bn()
    print(get_pr(dataset_val, retinanet,iou_threshold=float(parser.iou_threshold),save_path='tmp'))

if __name__ == '__main__':
    main()

