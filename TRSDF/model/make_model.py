import torch
import torch.nn as nn
from .backbones.resnet import ResNet, Bottleneck
import copy
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
import matplotlib.pyplot as plt
from .classifier_norm import Classifier_norm,Classifier_norm_circle


# 洗牌操作定义的函数
def shuffle_unit(features, shift, group, begin=1):  # group:2, shift:5

    batchsize = features.size(0)  # 64
    dim = features.size(-1)  # 768
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)  # 移位后得到的隐藏特征,同时去掉了class_token
    # 对应原文中shift操作！features[:, begin-1+shift:]:[64, 238, 768] , features[:, begin:begin-1+shift]:[64, 4, 768]

    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)  # [64, 2, 121, 768]
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()  # [64, 121, 2, 768]
    x = x.view(batchsize, -1, dim)  # [64, 242, 768]

    return x

# def window_partition(x, window_size: int):
#
#     B, H, W, C = x.shape
#     x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
#     # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
#     # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
#     windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
#     return windows


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER  # 默认为False
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT  # before

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT  # before
        self.in_planes = 768


        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate= cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None, cam_label= None, view_label=None):
        global_feat = self.base(x, cam_label=cam_label, view_label=view_label)

        feat = self.bottleneck(global_feat)  # 经过BN-Neck后的feat

        if self.training:  # True
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)

            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))



class SENet(nn.Module):
    def __init__(self, channel, ratio=0.25):
        super(SENet, self).__init__()

        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel * ratio)),
            nn.ReLU(),
            nn.Linear(int(channel * ratio), channel),
            nn.Sigmoid()
        )
        # self.fc1 = nn.Linear(channel, channel * ratio)
        # self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(channel * ratio, channel)
        # self.sigmoid = nn.Sigmoid()
    def forward(self, x):  # input:[32, 46, 768]
        B, N, C = x.size()
        x = x.transpose(1, 2)
        branch = self.global_avg_pooling(x)
        branch = branch.view(B, C)

        weight = self.fc(branch).view(B, C, 1)
        # weight = self.fc1(branch)
        # weight = self.relu(weight)
        # weight = self.fc2(weight)
        # weight = self.sigmoid(weight).view(B, C, 1, 1)

        scale = weight * x
        scale = scale.transpose(1, 2)
        return scale



class build_transformer_local(nn.Module):
    def __init__(self, num_classes,   camera_num, view_num, cfg, factory, rearrange):
        super(build_transformer_local, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER  # 训练时是否使用ArcFace loss，默认值为False
        self.neck = cfg.MODEL.NECK  # 训练时是否使用BNNeck
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.ZeroPad2D = nn.ZeroPad2d(padding=(1, 0, 0, 0))
        self.window_size = 2

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:  # default:True
            camera_num = camera_num
        else:
            camera_num = 0

        if cfg.MODEL.SIE_VIEW:  # default:False
            view_num = view_num
        else:
            view_num = 0

        # 调用TransReID函数
        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH, distilled=False, fuse_token=True, keep_rate=(1, 1, 1, 0.8, 1, 1, 0.8, 1, 1, 0.8, 1, 1))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)  # 加载在imagenet数据集上的预训练参数
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        block = self.base.blocks[-1]  # 论文中最后一层transformer layer(l-1的后一层)
        layer_norm = self.base.norm  # LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)  # copy.deepcopy()函数是一个深复制函数。就是从输入变量完全复刻一个相同的变量，无论怎么改变新变量，原有变量的值都不会受到影响。
        )
        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE  # 默认为softmax
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_1.apply(weights_init_classifier)
            self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_2.apply(weights_init_classifier)
            self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_3.apply(weights_init_classifier)
            self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_4.apply(weights_init_classifier)

            self.classifier_norm = Classifier_norm(self.in_planes, self.num_classes)
            self.classifier_norm.apply(weights_init_classifier)
            self.classifier_norm_1 = Classifier_norm(self.in_planes, self.num_classes)
            self.classifier_norm_1.apply(weights_init_classifier)
            self.classifier_norm_2 = Classifier_norm(self.in_planes, self.num_classes)
            self.classifier_norm_2.apply(weights_init_classifier)
            self.classifier_norm_3 = Classifier_norm(self.in_planes, self.num_classes)
            self.classifier_norm_3.apply(weights_init_classifier)
            self.classifier_norm_4 = Classifier_norm(self.in_planes, self.num_classes)
            self.classifier_norm_4.apply(weights_init_classifier)

            self.classifier_norm_circlesoftmax = Classifier_norm_circle(self.in_planes, self.num_classes)
            self.classifier_norm_circlesoftmax.apply(weights_init_classifier)
            self.classifier_norm_circlesoftmax_1 = Classifier_norm_circle(self.in_planes, self.num_classes)
            self.classifier_norm_circlesoftmax_1.apply(weights_init_classifier)
            self.classifier_norm_circlesoftmax_2 = Classifier_norm_circle(self.in_planes, self.num_classes)
            self.classifier_norm_circlesoftmax_2.apply(weights_init_classifier)
            self.classifier_norm_circlesoftmax_3 = Classifier_norm_circle(self.in_planes, self.num_classes)
            self.classifier_norm_circlesoftmax_3.apply(weights_init_classifier)
            self.classifier_norm_circlesoftmax_4 = Classifier_norm_circle(self.in_planes, self.num_classes)
            self.classifier_norm_circlesoftmax_4.apply(weights_init_classifier)


        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)

        self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP  # 默认为2
        print('using shuffle_groups size:{}'.format(self.shuffle_groups))
        self.shift_num = cfg.MODEL.SHIFT_NUM  # 默认为5  # 对应原文中shift的位数 m
        print('using shift_num size:{}'.format(self.shift_num))
        self.divide_length = cfg.MODEL.DEVIDE_LENGTH  # 默认为4  # 将隐藏特征分为4组进行计算local_feature k
        print('using divide_length size:{}'.format(self.divide_length))
        self.rearrange = rearrange  # True

        self.senet = SENet(768, 0.25)

    def forward(self, x, label=None, cam_label=None, view_label=None):  # label is unused if self.cos_layer == 'no'

        features = self.base(x, cam_label=cam_label, view_label=view_label)

        # features, index_visual = self.base(x, cam_label=cam_label, view_label=view_label)  # [64, 243, 768]

        # 不可视化时注释一下
        # return index_visual


        # global branch
        b1_feat = self.b1(features)  # [64, 243, 768]  243=22*11 + 1
        global_feat = b1_feat[:, 11]  # 取b1_feat第一维的所有数据  [64, 768]

        # JPM branch
        # feature_length = features.size(1) - 1  # 242，对应原文去除cls_token进行shift和shuffle # 69.2的feature_length

        feature_length = 242 # 12.19新加的-------69.2

        patch_length = feature_length // self.divide_length  # 242/4 = 60
        token = features[:, 10:11]  # 取出token为第0维到第1维的所有数据,也就是class_token

        # if self.rearrange:
        #     x = shuffle_unit(features, self.shift_num, self.shuffle_groups)  # [64, 242, 768]
        # else:
        #     x = features[:, 1:]

        x = features[:, 242:]  # 只改了这一行，不做shift和shuffle操作的实验都要用到这一行!!!!  # 69.2的x






        # lf_1                                                                 # 原文
        b1_local_feat = x[:, :patch_length]  # (64, 60, 768)
        b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))  # (64, 61, 768)
        local_feat_1 = b1_local_feat[:, 0]  # (64, 768)

        # lf_2
        b2_local_feat = x[:, patch_length:patch_length * 2]
        b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        local_feat_2 = b2_local_feat[:, 0]

        # lf_3
        b3_local_feat = x[:, patch_length * 2:patch_length * 3]
        b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        local_feat_3 = b3_local_feat[:, 0]

        # lf_4
        b4_local_feat = x[:, patch_length * 3:patch_length * 4]
        b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        local_feat_4 = b4_local_feat[:, 0]


        # x = features[:, -242:]
        #
        # # lf_1                                                                 # 12.19改的，感觉以前的有一些问题  b1_local_feat[:, 1]结果不好  x = features[:, -242:]效果不好
        # b1_local_feat = x[:, :patch_length]  # (64, 60, 768)
        # b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))  # (64, 61, 768)
        # local_feat_1 = b1_local_feat[:, 0]  # (64, 768)
        #
        # # lf_2
        # b2_local_feat = x[:, patch_length:patch_length * 2]
        # b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        # local_feat_2 = b2_local_feat[:, 0]
        #
        # # lf_3
        # b3_local_feat = x[:, patch_length * 2:patch_length * 3]
        # b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        # local_feat_3 = b3_local_feat[:, 0]
        #
        # # lf_4
        # b4_local_feat = x[:, patch_length * 3:patch_length * 4]
        # b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        # local_feat_4 = b4_local_feat[:, 0]



        # patch_length_1_3 = feature_length // 3   # 重叠分局部特征 12.20========69.2
        # b1_local_feat = x[:, :patch_length_1_3]  # (64, 60, 768)
        # b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))  # (64, 61, 768)
        # local_feat_1 = b1_local_feat[:, 0]  # (64, 768)
        #
        # # lf_2
        # b2_local_feat = x[:, patch_length:patch_length+patch_length_1_3]
        # b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        # local_feat_2 = b2_local_feat[:, 0]
        #
        # # lf_3
        # b3_local_feat = x[:, patch_length * 2:patch_length * 2 + patch_length_1_3]
        # b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        # local_feat_3 = b3_local_feat[:, 0]
        #
        # # lf_4
        # b4_local_feat = x[:, patch_length_1_3 * 2:patch_length * 4]
        # b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        # local_feat_4 = b4_local_feat[:, 0]











        # b1_local_feat = x[:, :patch_length]  # (64, 60, 768)                                     #  SENet改法  在最后一个transformer_layer后面
        # b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))  # (64, 61, 768)
        # b1_local_feat = self.senet(b1_local_feat)
        # local_feat_1 = b1_local_feat[:, 0]  # (64, 768)
        #
        # # lf_2
        # b2_local_feat = x[:, patch_length:patch_length * 2]
        # b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        # b2_local_feat = self.senet(b2_local_feat)
        # local_feat_2 = b2_local_feat[:, 0]
        #
        # # lf_3
        # b3_local_feat = x[:, patch_length * 2:patch_length * 3]
        # b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        # b3_local_feat = self.senet(b3_local_feat)
        # local_feat_3 = b3_local_feat[:, 0]
        #
        # # lf_4
        # b4_local_feat = x[:, patch_length * 3:patch_length * 4]
        # b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        # b4_local_feat = self.senet(b4_local_feat)
        # local_feat_4 = b4_local_feat[:, 0]






        # b1_local_feat = x[:,:patch_length]  # (64, 60, 768)                                     #  SENet改法  在最后一个transformer_layer前面
        # b1_local_feat = self.senet(b1_local_feat)
        # b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))  # (64, 61, 768)
        # local_feat_1 = b1_local_feat[:, 0]  # (64, 768)
        #
        # # lf_2
        # b2_local_feat = x[:, patch_length:patch_length * 2]
        # b2_local_feat = self.senet(b2_local_feat)
        # b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        # local_feat_2 = b2_local_feat[:, 0]
        #
        # # lf_3
        # b3_local_feat = x[:, patch_length * 2:patch_length * 3]
        # b3_local_feat = self.senet(b3_local_feat)
        # b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        # local_feat_3 = b3_local_feat[:, 0]
        #
        # # lf_4
        # b4_local_feat = x[:, patch_length * 3:patch_length * 4]
        # b4_local_feat = self.senet(b4_local_feat)
        # b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        # local_feat_4 = b4_local_feat[:, 0]




        # b1_local_feat = x[:, :patch_length]  # (64, 60, 768)   头部                  # 实验十  师兄切块方法
        # b2_local_feat = x[:, patch_length:patch_length * 2]  # 2，3拼接为躯干
        # b3_local_feat = x[:, patch_length * 2:patch_length * 3]
        # b4_local_feat = x[:, patch_length * 3:patch_length * 4]  # 下半身
        #
        # head_feature = b1_local_feat
        # body_feature = torch.cat((b2_local_feat, b3_local_feat), dim=1)
        # lower_body_feature = b4_local_feat
        #
        # head_feature = self.b2(torch.cat((token, head_feature), dim=1))
        # head_feature_1 = head_feature[:, 0]
        #
        # body_feature = self.b2(torch.cat((token, body_feature), dim=1))
        # body_feature_1 = body_feature[:, 0]
        #
        # lower_body_feature = self.b2(torch.cat((token, lower_body_feature), dim=1))
        # lower_body_feature_1 = lower_body_feature[:, 0]
        #
        # feat = self.bottleneck(global_feat)
        #
        # head_feature_1_bn = self.bottleneck_1(head_feature_1)
        # body_feature_1_bn = self.bottleneck_2(body_feature_1)
        # lower_body_feature_1_bn = self.bottleneck_3(lower_body_feature_1)
        #
        # if self.training:
        #     if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
        #         cls_score = self.classifier(feat, label)
        #     else:
        #         cls_score = self.classifier(feat)
        #         cls_score_1 = self.classifier_1(head_feature_1_bn)
        #         cls_score_2 = self.classifier_2(body_feature_1_bn)
        #         cls_score_3 = self.classifier_3(lower_body_feature_1_bn)
        #     return [cls_score, cls_score_1, cls_score_2, cls_score_3,
        #             ], [global_feat, head_feature_1, body_feature_1,
        #                 lower_body_feature_1]  # global feature for triplet loss  # 不要经过bn的
        # else:
        #     if self.neck_feat == 'after':
        #         return torch.cat(
        #             [feat, head_feature_1_bn / 3, body_feature_1_bn / 3, lower_body_feature_1_bn / 3], dim=1)
        #     else:
        #         return torch.cat(
        #             [global_feat, head_feature_1 / 3, body_feature_1 / 3, lower_body_feature_1 / 3], dim=1)






        # x = features[:, 1:]  # 获取除class_token之外的feature  # Transreid_windows_partition更改部分！！！ 实验2！！！
        # # x = x.permute(0, 3, 1, 2)
        # x = x.view(-1, 22, 11, 768)
        # x = self.ZeroPad2D(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        #
        # windows = window_partition(x, self.window_size)
        #
        # windows1 = windows[:, 0].view(-1, 66, 768)
        # windows2 = windows[:, 1].view(-1, 66, 768)
        # windows3 = windows[:, 2].view(-1, 66, 768)
        # windows4 = windows[:, 3].view(-1, 66, 768)
        #
        #
        # b1_local_feat = self.b2(torch.cat((token, windows1), dim=1))  # (64, 61, 768)
        # local_feat_1 = b1_local_feat[:, 0]  # (64, 768)
        #
        # b2_local_feat = self.b2(torch.cat((token, windows2), dim=1))
        # local_feat_2 = b2_local_feat[:, 0]
        #
        # b3_local_feat = self.b2(torch.cat((token, windows3), dim=1))
        # local_feat_3 = b3_local_feat[:, 0]
        #
        # b4_local_feat = self.b2(torch.cat((token, windows4), dim=1))
        # local_feat_4 = b4_local_feat[:, 0]






        # patch_divide = patch_length // 3                                   # 实验十  SSM模块
        #
        # b1_local_feat = x[:, :patch_length]  # (64, 60, 768)
        # b2_local_feat = x[:, patch_length:patch_length * 2]
        # b3_local_feat = x[:, patch_length * 2:patch_length * 3]
        # b4_local_feat = x[:, patch_length * 3:patch_length * 4]
        #
        # b1_local_feat = b1_local_feat + b2_local_feat   # 以下两行做SSM+FRM用到，只做SSM时候请注释掉！！！！！
        # b4_local_feat = b3_local_feat + b4_local_feat
        #
        #
        # f_out = torch.cat((token, b1_local_feat, b2_local_feat, b3_local_feat, b4_local_feat), dim=1)  # 对应PFT原文FRM中f_out
        # f_out = self.b1(f_out)
        # feat = f_out[:, 0]
        # feat_bn = self.bottleneck(feat)
        #
        #
        # fusion_feature = b1_local_feat + b2_local_feat + b3_local_feat + b4_local_feat
        # GLF = torch.cat((token, fusion_feature), dim=1)  # 要经过self.b2的特征
        #
        # # patch_group_divide
        # b1_local_feat_1 = b1_local_feat[:, :patch_divide]
        # b1_local_feat_2 = b1_local_feat[:, patch_divide:patch_divide * 2]
        # b1_local_feat_3 = b1_local_feat[:, patch_divide * 2:patch_divide * 3]
        #
        # b2_local_feat_4 = b2_local_feat[:, :patch_divide]
        # b2_local_feat_5 = b2_local_feat[:, patch_divide:patch_divide * 2]
        # b2_local_feat_6 = b2_local_feat[:, patch_divide * 2:patch_divide * 3]
        #
        # b3_local_feat_7 = b3_local_feat[:, :patch_divide]
        # b3_local_feat_8 = b3_local_feat[:, patch_divide:patch_divide * 2]
        # b3_local_feat_9 = b3_local_feat[:, patch_divide * 2:patch_divide * 3]
        #
        # b4_local_feat_10 = b4_local_feat[:, :patch_divide]
        # b4_local_feat_11 = b4_local_feat[:, patch_divide:patch_divide * 2]
        # b4_local_feat_12 = b4_local_feat[:, patch_divide * 2:patch_divide * 3]
        #
        # left_feature = torch.cat((b1_local_feat_1, b2_local_feat_4, b3_local_feat_7, b4_local_feat_10), dim=1)  # 要经过self.b2的特征
        # middle_feature = torch.cat((b1_local_feat_2, b2_local_feat_5, b3_local_feat_8, b4_local_feat_11), dim=1)  # 要经过self.b2的特征
        # right_feature = torch.cat((b1_local_feat_3, b2_local_feat_6, b3_local_feat_9, b4_local_feat_12), dim=1)  # 要经过self.b2的特征
        #
        # left_feature = torch.cat((token, left_feature), dim=1)
        # middle_feature = torch.cat((token, middle_feature), dim=1)
        # right_feature = torch.cat((token, right_feature), dim=1)
        #
        # GLF = self.b2(GLF)
        # GLF = GLF[:, 0]
        #
        # left_feature = self.b2(left_feature)
        # left_feature = left_feature[:, 0]
        # middle_feature = self.b2(middle_feature)
        # middle_feature = middle_feature[:, 0]
        # right_feature = self.b2(right_feature)
        # right_feature = right_feature[:, 0]
        #
        # GLF_bn = self.bottleneck_1(GLF)
        # left_feature_bn = self.bottleneck_2(left_feature)
        # middle_feature_bn = self.bottleneck_3(middle_feature)
        # right_feature_bn = self.bottleneck_4(right_feature)
        #
        # if self.training:
        #     if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
        #         cls_score = self.classifier(feat_bn, label)
        #     else:
        #         cls_score = self.classifier(feat_bn)
        #         cls_score_1 = self.classifier_1(GLF_bn)
        #         cls_score_2 = self.classifier_2(left_feature_bn)
        #         cls_score_3 = self.classifier_3(middle_feature_bn)
        #         cls_score_4 = self.classifier_4(right_feature_bn)
        #     return [cls_score, cls_score_1, cls_score_2, cls_score_3, cls_score_4
        #             ], [feat, GLF, left_feature, middle_feature, right_feature]  # global feature for triplet loss
        # else:
        #         return torch.cat(
        #             [feat, GLF / 4, left_feature / 4, middle_feature / 4, right_feature / 4], dim=1)






        # b1_local_feat = x[:, :patch_length]  # (64, 60, 768)               # 实验九！！！记得删除shift和shuffle模块 FRM模块
        # b2_local_feat = x[:, patch_length:patch_length * 2]
        # b3_local_feat = x[:, patch_length * 2:patch_length * 3]
        # b4_local_feat = x[:, patch_length * 3:patch_length * 4]
        #
        #
        #
        # # lf_1
        # b1_local_feat = b1_local_feat + b2_local_feat
        # b4_local_feat = b3_local_feat + b4_local_feat
        #
        # f_out = torch.cat((token, b1_local_feat, b2_local_feat, b3_local_feat, b4_local_feat), dim=1)  # 对应PFT原文FRM中f_out
        # f_out = self.b1(f_out)
        # feat = f_out[:, 0]
        # feat_bn = self.bottleneck(feat)
        #
        #
        #
        # b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))  # (64, 61, 768)
        # local_feat_1 = b1_local_feat[:, 0]  # (64, 768)
        #
        # # lf_2
        # b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        # local_feat_2 = b2_local_feat[:, 0]
        #
        # # lf_3
        # b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        # local_feat_3 = b3_local_feat[:, 0]
        #
        # # lf_4
        # b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        # local_feat_4 = b4_local_feat[:, 0]
        #
        # # 4个经过BN层的局部特征
        # local_feat_1_bn = self.bottleneck_1(local_feat_1)
        # local_feat_2_bn = self.bottleneck_2(local_feat_2)
        # local_feat_3_bn = self.bottleneck_3(local_feat_3)
        # local_feat_4_bn = self.bottleneck_4(local_feat_4)
        #
        # if self.training:
        #     if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
        #         cls_score = self.classifier(feat_bn, label)
        #     else:
        #         cls_score = self.classifier(feat_bn)
        #         cls_score_1 = self.classifier_1(local_feat_1_bn)
        #         cls_score_2 = self.classifier_2(local_feat_2_bn)
        #         cls_score_3 = self.classifier_3(local_feat_3_bn)
        #         cls_score_4 = self.classifier_4(local_feat_4_bn)
        #     return [cls_score, cls_score_1, cls_score_2, cls_score_3,
        #                 cls_score_4
        #                 ], [feat, local_feat_1, local_feat_2, local_feat_3,
        #                     local_feat_4]  # global feature for triplet loss
        # else:
        #     if self.neck_feat == 'after':
        #         return torch.cat(
        #             [feat_bn, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4], dim=1)
        #     else:
        #         return torch.cat(
        #             [feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], dim=1)
























        feat = self.bottleneck(global_feat)  # 经过BN层的全局特征  [64, 768], 做实验10需要注释掉

        # 4个经过BN层的局部特征
        local_feat_1_bn = self.bottleneck_1(local_feat_1)
        local_feat_2_bn = self.bottleneck_2(local_feat_2)
        local_feat_3_bn = self.bottleneck_3(local_feat_3)
        local_feat_4_bn = self.bottleneck_4(local_feat_4)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
                cls_score_1 = self.classifier_1(local_feat_1_bn)
                cls_score_2 = self.classifier_2(local_feat_2_bn)
                cls_score_3 = self.classifier_3(local_feat_3_bn)
                cls_score_4 = self.classifier_4(local_feat_4_bn)
            # else:
            #     cls_score = self.classifier_norm(feat)                               # Trick1 classifier_norm层
            #     cls_score_1 = self.classifier_norm_1(local_feat_1_bn)
            #     cls_score_2 = self.classifier_norm_2(local_feat_2_bn)
            #     cls_score_3 = self.classifier_norm_3(local_feat_3_bn)
            #     cls_score_4 = self.classifier_norm_4(local_feat_4_bn)
            # else:
            #     cls_score = self.classifier_norm_circlesoftmax(feat)                  # Trick1+2  classifier_norm层 + circlesoftmax  有问题，circlesoftmax的forward函数缺少参数targets
            #     cls_score_1 = self.classifier_norm_circlesoftmax_1(local_feat_1_bn)
            #     cls_score_2 = self.classifier_norm_circlesoftmax_2(local_feat_2_bn)
            #     cls_score_3 = self.classifier_norm_circlesoftmax_3(local_feat_3_bn)
            #     cls_score_4 = self.classifier_norm_circlesoftmax_4(local_feat_4_bn)
            return [cls_score, cls_score_1, cls_score_2, cls_score_3,
                        cls_score_4], global_feat  # global feature for triplet loss  # 不要经过bn的
            # return cls_score
        else:
            if self.neck_feat == 'after':
                return torch.cat(
                    [feat, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4], dim=1)
            else:
                return torch.cat(
                    [global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], dim=1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID
}

def make_model(cfg, num_class, camera_num, view_num):
    if cfg.MODEL.NAME == 'transformer':
        if cfg.MODEL.JPM:
            # print(camera_num,view_num)
            model = build_transformer_local(num_class, camera_num, view_num, cfg, __factory_T_type, rearrange=cfg.MODEL.RE_ARRANGE)
            print('===========building transformer with JPM module ===========')
        else:
            model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
            print('===========building transformer===========')
    else:
        model = Backbone(num_class, cfg)
        print('===========building ResNet===========')
    return model
