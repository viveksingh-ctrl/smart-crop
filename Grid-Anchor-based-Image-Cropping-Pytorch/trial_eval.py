from croppingModel import build_crop_model
from trial_cropping_dataset import setup_test_dataset
import os
import torch
import cv2
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import argparse
import time

## fastapi conversion
from fastapi import FastAPI
from pydantic import BaseModel
import matplotlib.pyplot as plt
###
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def smart_crop(input_img, output_dir):
    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

    # arguments


    batch_size = 1
    num_workers = 0
    cuda = False
    # output_dir = 'trial_test_result'
    net_path = 'pretrained_model/mobilenet_0.625_0.583_0.553_0.525_0.785_0.762_0.748_0.723_0.783_0.806.pth'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if torch.cuda.is_available():
        if False:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if not False:
            print("WARNING: It looks like you have a CUDA device, but aren't " +
                  "using CUDA.\nRun with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    dataset = setup_test_dataset(dataset_dir=input_img)

    def naive_collate(batch):
        return batch[0]

    def output_file_name(input_path, idx):
        name = os.path.basename(input_path)
        segs = name.split('.')
        assert len(segs) >= 2
        return '%scrop_%d.%s' % ('.'.join(segs[:-1]), idx, segs[-1])

    for epoch in range(0,1):

        net = build_crop_model(scale='multi',#scale='single',
                               alignsize=9, reddim=8, loadweight=False, model='mobilenetv2',downsample=4)
        net.load_state_dict(torch.load(net_path, map_location = torch.device('cpu')))
        net.eval()

        # if args.cuda:
        #     net = torch.nn.DataParallel(net,device_ids=[0])
        #     cudnn.benchmark = True
        #     net = net.cuda()

        data_loader = data.DataLoader(dataset, batch_size,
                                      num_workers=num_workers,
                                      collate_fn=naive_collate,
                                      shuffle=False)
        print('-'*50)
        for id, sample in enumerate(data_loader):

            imgpath = sample['imgpath']
            image = sample['image']
            bboxes = sample['sourceboxes']
            resized_image = sample['resized_image']
            tbboxes = sample['tbboxes']

            if len(tbboxes['xmin'])==0:
                continue

            roi = []

            for idx in range(0,len(tbboxes['xmin'])):
                roi.append((0, tbboxes['xmin'][idx],tbboxes['ymin'][idx],tbboxes['xmax'][idx],tbboxes['ymax'][idx]))

            resized_image = torch.unsqueeze(torch.as_tensor(resized_image), 0)
            if cuda:
                resized_image = Variable(resized_image.cuda())
                roi = Variable(torch.Tensor(roi))
            else:
                resized_image = Variable(resized_image)
                roi = Variable(torch.Tensor(roi))

            t0 = time.time()
            out = net(resized_image,roi)
            t1 = time.time()
            print('timer: %.4f sec.' % (t1 - t0))
            id_out = sorted(range(len(out)), key=lambda k: out[k], reverse = True)
            # print(id_out)
            # print(id)
            for id in range(0,1):
                top_box = bboxes[id_out[id]]
                top_crop = image[int(top_box[0]):int(top_box[2]),int(top_box[1]):int(top_box[3])]
                # plt.imshow(top_crop[])

                imgname = imgpath[0].split('/')[-1]
                output_name = output_file_name(imgpath, id+1)
                cv2.imwrite(os.path.join(output_dir,
                                         output_name),
                            top_crop[:,:,(2, 1, 0)])
            statement = imgpath+' -> Image cropped successfully at '+output_dir+'/'+output_name
            return statement

smart_crop('test/github_profile_pic.jpeg', 'smart_crop_result')
# app = FastAPI()
# class Item(BaseModel):
#     image_path:str
#
# @app.post('/crop_image/')
# async def create_item(item:Item):
#     item_dict = item.dict()
#
#     # test('test/img3.jpeg')
#     x = test(item_dict['image_path'])
#     return x
