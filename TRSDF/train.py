# coding=gbk

from utils.logger import setup_logger
from datasets import make_dataloader
from model.make_model import make_model
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from processor import do_train
import random
import torch
import numpy as np
import os
import argparse
# from timm.scheduler import create_scheduler
from config import cfg
from thop import profile


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument('--base_keep_rate', type=float, default=0.7,
                        help='Base keep rate (default: 0.7)')  # ��ӵ�
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--resume_and_load', default='',  # ./logs/occ_duke_vit_transreid_stride/transformer_120.pth
                        help='load the .pth you have trained. fill in your .pth path')
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)  # ��������ѵ��Ȩ�ص��ļ���

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID  # ���õ�ǰʹ�õ�GPU�豸��Ϊ0���豸
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    # ��ģ�Ͳ���ʱ����������
    # model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num).cuda()


    # ���������������  117.70M
    # param_count = sum(p.numel() for p in model.parameters()) / 1000000.0
    # print(param_count)

    # ����FLOPS��Params  flops: 702479.20 M, params: 95.10 M
    # dummy_input = torch.randn(32, 3, 256, 128).cuda()
    # flops, params = profile(model, (dummy_input,))
    # print('flops: ', flops, 'params: ', params)
    # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))




    start_epoch = 1

    # if args.resume_and_load:              # �ֶ���ȡepoch,����ѵ��
    #     checkpoint = torch.load(args.resume_and_load)
    #     # model.load_state_dict(checkpoint['state_dict'])
    #     model.load_state_dict(checkpoint)
    #     start_epoch = int(args.resume_and_load.split("_")[-1].split(".")[0]) + 1






    # print(model)
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)  # ʹ��SGD�Ż����Ż�ģ�Ͳ�����������ʧ����

    scheduler = create_scheduler(cfg, optimizer)

    # if args.resume_and_load:                                           # ����model,epoch,optimizer��������,����ѵ��
    #     checkpoint = torch.load(args.resume_and_load)  # ���ضϵ�
    #     model.load_state_dict(checkpoint['model'])  # ����ģ�Ϳ�ѧϰ����
    #     optimizer.load_state_dict(checkpoint['optimizer'])  # �����Ż�������
    #     start_epoch = checkpoint['epoch']  # ���ÿ�ʼ��epoch


    if args.resume_and_load:
        # ���ضϵ�ģ��
        checkpoint_path = args.resume_and_load
        train_state = torch.load(checkpoint_path)
        # ���ضϵ��״̬
        model.load_state_dict(train_state['model_state_dict'])
        optimizer.load_state_dict(train_state['optimizer_state_dict'])
        start_epoch = train_state['epoch'] + 1



    do_train(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_func,
        num_query, args.local_rank,
    start_epoch
    )
