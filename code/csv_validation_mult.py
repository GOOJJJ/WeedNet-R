import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import csv
import torch
from torchvision import transforms
from retinanet import model
from retinanet.dataloader import CSVDataset, Resizer, Normalizer
from retinanet import csv_eval
import numpy as np
assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main():

    csv_annotations_path=r'./dataset/test.csv'
    model_dir=r'path to pt files dir'
    class_list_path =r'./dataset/class_list.csv'
    iou_threshold='0.5'

    save_dir=os.path.join(os.getcwd(), 'result')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    result_dir='from_'+str.split(model_dir,'/')[-1]
    save_path=os.path.join(save_dir,result_dir)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    model_files=[file for file in os.listdir(model_dir) if file.endswith('.pt') and 'final' not in file]

    model_number=[ str.split(file.split('.pt')[0],'_')[-1].strip() for file in model_files ]
    model_number= [int(i) for i in model_number]
    models={}
    for i,key in enumerate(model_number):
        models[key]=model_files[i]
    valid_num = len(models)
    dataset_val = CSVDataset(csv_annotations_path,class_list_path,transform=transforms.Compose([Normalizer(), Resizer()]))

    evaluation_index=['model_name','weed_AP','crop_AP','mAP', 'weed_precision', 'crop_precision', 'weed_recall', 'crop_recall']
    results=[]
    use_gpu = True

    with open(os.path.join(save_path, "info" + ".txt"), 'w') as f:
        f.write('Model from: %s\n\n'%(model_dir))
        f.write('baseline:Adam,bs=4,epoch=15,init_lr=1e-4')
        model = torch.load(os.path.join(model_dir, models[0]))
        f.write('Model architecture:\n %s'%(model))
        f.write('\n\n')

    with open(os.path.join(save_path, 'result.csv'), 'a', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(evaluation_index)
        for i in range(valid_num):
            print('evaluate {}/{}'.format(i + 1, valid_num))
            result=[]
            result.append(models[i].split('.')[0])
            model_path= os.path.join(model_dir,models[i])
            mAP,precisions,recalls=evaluate_mult(model_path,dataset_val,use_gpu,iou_threshold)
            mAP=[ mAP[key][0] for key in mAP.keys()]
            result.extend(mAP)
            result.append(np.mean(mAP))
            precisions=[precisions[key][-1] for key in precisions.keys() ]
            result.extend(precisions)
            recalls = [recalls[key][-1] for key in recalls.keys()]
            result.extend(recalls)
            results.append(result)

            f_csv.writerow(result)

            with open(os.path.join(save_path, "info" + ".txt"), 'a') as f:
                f.write(models[i].split('.')[0]+' eval result :\n')
                f.write('weed AP = %6f\ncrop AP = %6f\nmAP = %6f\n\n'%(result[1],result[2],(result[1]+result[2])/2))

    print(results)

def evaluate_mult(model_path,dataset_val,use_gpu,iou_threshold,save_path=None):
    # Create the model
    retinanet = torch.load(model_path)
    #print(retinanet)
    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        #retinanet.load_state_dict(torch.load(parser.model_path))
        retinanet = torch.nn.DataParallel(retinanet).cuda()

    else:
        retinanet.load_state_dict(torch.load(model_path))
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = False
    retinanet.eval()
    retinanet.module.freeze_bn()

    return  csv_eval.evaluate(dataset_val, retinanet,iou_threshold=float(iou_threshold),save_path=save_path)

if __name__ == '__main__':
    main()
