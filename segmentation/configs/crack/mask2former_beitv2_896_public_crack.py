num_things_classes = 0
num_stuff_classes = 2
num_classes = 2
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoderMask2Former',
    pretrained='pretrained/beitv2_large_patch16_224_pt1k_ft21k.pth',
    backbone=dict(
        type='BEiTAdapter',
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        use_abs_pos_emb=False,
        use_rel_pos_bias=True,
        img_size=896,
        init_values=1e-06,
        drop_path_rate=0.3,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        with_cp=True,
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]]),
    decode_head=dict(
        type='Mask2FormerHead',
        in_channels=[1024, 1024, 1024, 1024],
        feat_channels=1024,
        out_channels=1024,
        in_index=[0, 1, 2, 3],
        num_things_classes=0,
        num_stuff_classes=2,
        num_queries=200,
        num_transformer_feat_level=3,
        pixel_decoder=dict(
            type='MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=1024,
                        num_heads=32,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=False,
                        norm_cfg=None,
                        init_cfg=None),
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=1024,
                        feedforward_channels=4096,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True),
                        with_cp=True),
                    operation_order=('self_attn', 'norm', 'ffn', 'norm')),
                init_cfg=None),
            positional_encoding=dict(
                type='SinePositionalEncoding', num_feats=512, normalize=True),
            init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=512, normalize=True),
        transformer_decoder=dict(
            type='DetrTransformerDecoder',
            return_intermediate=True,
            num_layers=9,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=1024,
                    num_heads=32,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False),
                ffn_cfgs=dict(
                    embed_dims=1024,
                    feedforward_channels=4096,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True,
                    with_cp=True),
                feedforward_channels=4096,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                 'ffn', 'norm')),
            init_cfg=None),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1, 1, 0.1]),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0)),
    train_cfg=dict(
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        assigner=dict(
            type='MaskHungarianAssigner',
            cls_cost=dict(type='ClassificationCost', weight=2.0),
            mask_cost=dict(
                type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
            dice_cost=dict(
                type='DiceCost', weight=5.0, pred_act=True, eps=1.0)),
        sampler=dict(type='MaskPseudoSampler')),
    test_cfg=dict(
        panoptic_on=True,
        semantic_on=False,
        instance_on=True,
        max_per_image=100,
        iou_thr=0.8,
        filter_low_score=True,
        mode='slide',
        crop_size=(896, 896),
        stride=(512, 512)),
    init_cfg=None)
dataset_type = 'CustomDataset'
#data_root = 'data/Kuba_nocrop_all'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (896, 896)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(3584, 896), ratio_range=(0.5, 2.0)), 
    dict(type='RandomCrop', crop_size=(896, 896)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(896, 896), pad_val=0, seg_pad_val=255),
    dict(type='ToMask'),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_semantic_seg', 'gt_masks', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(3584, 1792), #2048, 1408),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            #dict(type='ResizeToMultiple', size_divisor=32),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
            type='CustomDataset',
            data_root='data/Public_all',
            img_dir='images/training',
            ann_dir='annotations/training',
            img_suffix='.jpg',
            seg_map_suffix='.png',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations'),
                dict(
                    type='Resize',
                    img_scale=(3584, 896), 
                    ratio_range=(0.5, 2.0)),
                dict(type='RandomCrop', crop_size=(896, 896)),
                dict(type='RandomFlip', prob=0.5),
                dict(type='PhotoMetricDistortion'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='Pad', size=(896, 896), pad_val=0, seg_pad_val=255),
                dict(type='ToMask'),
                dict(type='DefaultFormatBundle'),
                dict(
                    type='Collect',
                    keys=['img', 'gt_semantic_seg', 'gt_masks', 'gt_labels'])
            ],
            classes=['background', 'crack']),
    val=dict(
        type='CustomDataset',
        data_root='data/Public_all',
        img_dir='images/val2',
        ann_dir='annotations/val2',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 1408),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    #dict(type='ResizeToMultiple', size_divisor=32),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=['background', 'crack']),
    test=dict(
        type='CustomDataset',
        data_root='data/Public_all',
        img_dir='images/test2',
        ann_dir='annotations/test2',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(3584, 1792), #2048, 1408),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    #dict(type='ResizeToMultiple', size_divisor=32),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=['background', 'crack']))
log_config = dict(
    interval=64, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'pretrained/mask2former_beitv2_adapter_large_896_80k_ade20k.pth' #'./work_dirs/mask2former_beitv2_896_lab_speckle/best_mIoU_iter_76000_2images.pth'
resume_from = None #'work_dirs/mask2former_beitv2_adapter_large_896_320k_Kuba_nocrop_all_ss_eval_on_1408_grad_acc/iter_308000.pth'  #'work_dirs/mask2former_beit_adapter_large_640_160k_Kuba_nocrop_all_ss_grad_acc_eval_on_1280/iter_2000.pth'
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=2e-05,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.9))
optimizer_config = dict(type="GradientCumulativeOptimizerHook", cumulative_iters=32)
lr_config = dict(policy='poly', warmup='linear',
                 warmup_iters=8000,
                 warmup_ratio=1e-6, power=1.0, min_lr=0.0, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=120000)
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=1)
evaluation = dict(interval=4000, metric='mIoU', pre_eval=True, save_best='mIoU')
pretrained = 'pretrained/beitv2_large_patch16_224_pt1k_ft21k.pth'
work_dir = './work_dirs/mask2former_beitv2_896_public_crack'
gpu_ids = range(0, 1)
auto_resume = False
