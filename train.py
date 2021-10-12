import argparse
import os.path as osp
import yaml
import logging
import numpy as np

import os
import torch
import torch.backends.cudnn as cudnn

import torch.nn.functional as F
from semseg.models.model_helper import ModelBuilder

import torch.distributed as dist

from semseg.utils.loss_helper import get_criterion
from semseg.utils.lr_helper import get_scheduler, get_optimizer

from semseg.utils.utils import AverageMeter, intersectionAndUnion, init_log, load_trained_model
from semseg.utils.utils import dynamic_copy_paste, set_random_seed, update_cutmix_bank
from semseg.utils.utils import generate_cutmix_mask, cal_category_confidence, sample_from_bank
from semseg.utils.utils import get_world_size, get_rank, synchronize

import random
from semseg.dataset.builder import get_loader
import time

parser = argparse.ArgumentParser(description="Semi-Supervised Semantic Segmentation")
parser.add_argument("--config", type=str, default="config.yaml")
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
logger =init_log('global', logging.INFO)
logger.propagate = 0


def main():
    global args, cfg
    args = parser.parse_args()
    seed = args.seed
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    
    cudnn.enabled = True
    cudnn.benchmark = True

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    distributed = num_gpus > 1
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()
    rank = get_rank()
    world_size = get_world_size()

    if rank == 0:
        logger.info(cfg)
    if args.seed is not None:
        print('set random seed to',args.seed)
        set_random_seed(args.seed)

    if not osp.exists(cfg['saver']['snapshot_dir']) and rank == 0:
        os.makedirs(cfg['saver']['snapshot_dir'])
    
    # Create network.
    model = ModelBuilder(cfg['net'])
    modules_back = [model.encoder]
    modules_head = [model.auxor, model.decoder]
    device = torch.device("cuda")
    model.to(device)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            find_unused_parameters=True,
        ) 
    if cfg['saver']['pretrain']:
        state_dict = torch.load(cfg['saver']['pretrain'], map_location='cpu')['model_state']
        print("Load trained model from ", str(cfg['saver']['pretrain']))
        load_trained_model(model, state_dict)
    
    if rank ==0:
        logger.info(model)

    # Teacher model
    model_teacher = ModelBuilder(cfg['net'])
    model_teacher.to(device)
    if distributed:
        model_teacher = torch.nn.parallel.DistributedDataParallel(
            model_teacher, device_ids=[args.local_rank], output_device=args.local_rank,
            find_unused_parameters=True,
        )
    
    for p in model_teacher.parameters():
        p.requires_grad = False

    criterion = get_criterion(cfg)
    cons = cfg['criterion'].get('cons',False)
    sample = False
    if cons:
        sample = cfg['criterion']['cons'].get('sample', False)
    if cons:
        criterion_cons = get_criterion(cfg, cons=True)
    else: 
        criterion_cons = torch.nn.CrossEntropyLoss(ignore_index=255)
    
    trainloader_sup, trainloader_unsup, valloader = get_loader(cfg, seed=seed)
    
    # Optimizer and lr decay scheduler
    cfg_trainer = cfg['trainer']
    cfg_optim = cfg_trainer['optimizer']

    params_list = []
    for module in modules_back:
        params_list.append(dict(params=module.parameters(), lr=cfg_optim['kwargs']['lr']))
    for module in modules_head:
        params_list.append(dict(params=module.parameters(), lr=cfg_optim['kwargs']['lr']*10))

    optimizer = get_optimizer(params_list, cfg_optim)
    lr_scheduler = get_scheduler(cfg_trainer, len(trainloader_unsup), optimizer)  # TODO
    
    acp = cfg['dataset'].get('acp', False)
    acm = cfg['dataset']['train'].get('acm', False)
    
    if acp or acm or sample:
        class_criterion = torch.rand(3, cfg['net']['num_classes']).type(torch.float32)
    if acm:
        cutmix_bank = torch.zeros(cfg['net']['num_classes'], trainloader_unsup.dataset.__len__()).cuda()
        
    # Start to train model
    best_prec = 0
    labeled_epoch = 0
    for epoch in range(cfg_trainer['epochs']):
        # Training
        t_start = time.time()
        if not acp and not acm and not sample:
            labeled_epoch = train(model, optimizer, lr_scheduler, criterion, trainloader_sup, epoch, 
                                labeled_epoch, model_teacher, trainloader_unsup, criterion_cons)
        elif acm:
            labeled_epoch, class_criterion, cutmix_bank = train(model, optimizer, lr_scheduler, criterion, trainloader_sup, epoch, 
                                labeled_epoch, model_teacher, trainloader_unsup, criterion_cons, class_criterion, cutmix_bank)         
        else:
            labeled_epoch, class_criterion = train(model, optimizer, lr_scheduler, criterion, trainloader_sup, epoch, 
                                labeled_epoch, model_teacher, trainloader_unsup, criterion_cons, class_criterion)
        # Validataion
        if cfg_trainer["eval_on"]:
            if rank ==0:
                logger.info("start evaluation")
            prec = validate(model_teacher,model, valloader, epoch)
            if rank == 0:
                if prec > best_prec:
                    best_prec = prec
                    state = {'epoch': epoch,
                         'model_state': model_teacher.state_dict(),
                         'optimizer_state': optimizer.state_dict()}
                    torch.save(state, osp.join(cfg['saver']['snapshot_dir'], 'best_'+str(seed)+'.pth'))
                    logger.info('Currently, the best val result is: {}'.format(best_prec))
        # note we also save the last epoch checkpoint
        if (epoch == (cfg_trainer['epochs'] - 1) or epoch == (cfg_trainer['epochs'] - 2)) and rank == 0:
            state = {'epoch': epoch,
                     'model_state': model_teacher.state_dict(),
                     'optimizer_state': optimizer.state_dict()}
            torch.save(state, osp.join(cfg['saver']['snapshot_dir'], 'epoch_' + str(epoch) + '_' + str(seed)+'.pth'))
            logger.info('Save Checkpoint {}'.format(epoch))
        t_end = time.time()
        if rank == 0:
            print('time for one epoch',t_end - t_start)

def train(model, optimizer, lr_scheduler, criterion, data_loader, epoch, labeled_epoch, model_teacher, data_loader_unsup, criterion_cons, class_criterion=None, cutmix_bank=None):
    model.train()
    model_teacher.train()

    data_loader.sampler.set_epoch(labeled_epoch)
    data_loader_unsup.sampler.set_epoch(epoch)
    data_loader_iter = iter(data_loader)
    data_loader_unsup_iter = iter(data_loader_unsup)

    num_classes, ignore_label = cfg['net']['num_classes'], cfg['dataset']['ignore_label']
    ema_decay_origin = cfg['net']['ema_decay']
    consist_weight = cfg['criterion'].get('consist_weight', 1)
    threshold = cfg['criterion'].get('threshold',0)
    cutmix = cfg['dataset']['train'].get('cutmix', False)
    acm = cfg['dataset']['train'].get('acm', False)
    acp = cfg['dataset'].get('acp', False)
    sample = False
    num_cat = 3
    if cfg['criterion'].get('cons', False):
        sample = cfg['criterion']['cons'].get('sample', False)
    if sample:
        class_momentum = cfg['criterion']['cons'].get('momentum', 0.999)
    if acp:
        all_cat = [i for i in range(num_classes)]
        ignore_cat = [0, 1, 2, 8, 10]
        target_cat = list(set(all_cat)-set(ignore_cat))
        class_momentum = cfg['dataset']['acp'].get('momentum', 0.999)
        num_cat = cfg['dataset']['acp'].get('number', 3)
    if acm:
        class_momentum = cfg['dataset']['train']['acm'].get('momentum', 0.999)
        area_thresh = cfg['dataset']['train']['acm'].get('area_thresh', 0.0001)  
        no_pad = cfg['dataset']['train']['acm'].get('no_pad', False)
        no_slim = cfg['dataset']['train']['acm'].get('no_slim', False)
        if 'area_thresh2' in cfg['dataset']['train']['acm'].keys():
            area_thresh2 = cfg['dataset']['train']['acm']['area_thresh2']
        else:
            area_thresh2 = area_thresh  
    rank, world_size = get_rank(), get_world_size()
    if acp or acm or sample:
        conf = 1 - class_criterion[0]
        conf = conf[target_cat]
        conf = (conf**0.5).numpy()
        conf_print = np.exp(conf)/np.sum(np.exp(conf))
        if rank == 0:
            print('epoch [',epoch,': ]', 'sample_rate_target_class_conf', conf_print)
        if rank == 0:
            print('epoch [',epoch,': ]', 'criterion_per_class' ,class_criterion[0])
            print('epoch [',epoch,': ]', 'sample_rate_per_class_conf' ,(1-class_criterion[0])/(torch.max(1-class_criterion[0])+1e-12))
            
    losses = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    #for step, batch in enumerate(data_loader_unsup):
    for step in range(len(data_loader_unsup)):
        i_iter = epoch * len(data_loader_unsup) + step
        lr = lr_scheduler.get_lr()
        lr_scheduler.step()
        if acp or acm:
            conf = 1 - class_criterion[0]
            conf = conf[target_cat]
            conf = (conf**0.5).numpy()
            conf = np.exp(conf)/np.sum(np.exp(conf))
            query_cat = []
            for rc_idx in range(num_cat):
                query_cat.append(np.random.choice(target_cat, p=conf))
            query_cat = list(set(query_cat))
        # get labeled input
        if acp:
            try:
                labeled_inputs = data_loader_iter.next()
            except:
                labeled_epoch += 1
                data_loader.sampler.set_epoch(labeled_epoch)
                data_loader_iter = iter(data_loader)
                labeled_inputs = data_loader_iter.next()
            if len(labeled_inputs) > 2:
                images_sup, labels_sup, paste_img, paste_label = labeled_inputs
                images_sup = images_sup.cuda()
                labels_sup = labels_sup.long().cuda()
                paste_img  = paste_img.cuda()
                paste_label = paste_label.long().cuda()        
                images_sup, labels_sup = dynamic_copy_paste(images_sup, labels_sup, paste_img, paste_label, query_cat)
                del paste_img, paste_label
            else:
                images_sup, labels_sup = labeled_inputs
                images_sup = images_sup.cuda()
                labels_sup = labels_sup.long().cuda()     
                images_sup, labels_sup = dynamic_copy_paste(images_sup, labels_sup, query_cat)           
        else:
            try:
                images_sup, labels_sup = data_loader_iter.next()
            except:
                labeled_epoch += 1
                data_loader.sampler.set_epoch(labeled_epoch)
                data_loader_iter = iter(data_loader)
                images_sup, labels_sup = data_loader_iter.next()
            images_sup = images_sup.cuda()
            labels_sup = labels_sup.long().cuda()
        # get unlabeled input
        if not cutmix and not acm:
            images_unsup_weak, _, images_unsup_strong, _ ,valid_mask= data_loader_unsup_iter.next()
            images_unsup_weak = images_unsup_weak.cuda()
            images_unsup_strong = images_unsup_strong.cuda()
            valid_mask = valid_mask.long().cuda()
        elif acm:
            image_unsup, _, img_id = data_loader_unsup_iter.next()
            prob_im = random.random()
            if image_unsup.shape[0] > 1:
                if prob_im>0.5:
                    image_unsup = image_unsup[0]
                    img_id = img_id[0]
                else:
                    image_unsup = image_unsup[1]
                    img_id = img_id[1]
            image_unsup = image_unsup.cuda()
            sample_id, sample_cat = sample_from_bank(cutmix_bank, class_criterion[0])
            image_unsup2, _, _ = data_loader_unsup.dataset.__getitem__(index=sample_id)
            image_unsup2 = image_unsup2.cuda()
            images_unsup = torch.cat([image_unsup.unsqueeze(0),image_unsup2.unsqueeze(0)],dim=0)
            images_unsup_weak = images_unsup.clone()
        else:
            # cutmix for unlabeled input
            images_unsup, _, valid_masks = data_loader_unsup_iter.next()
            images_unsup = images_unsup.cuda()
            valid_masks = valid_masks.long().cuda()
            images_unsup_weak = images_unsup.clone() 
            #construct strong and weak inputs for teacher and student model
            assert valid_masks.shape[0] == 2
            # images_unsup 2(B),3,H,W
            prob = random.random()      
            if prob > 0.5:
                valid_mask_mix = valid_masks[0]    # H, W
                images_unsup_strong = images_unsup[0] * valid_mask_mix + images_unsup[1] * (1 - valid_mask_mix)
                images_unsup_strong = images_unsup_strong.unsqueeze(0)
            else:
                valid_mask_mix = valid_masks[1]
                images_unsup_strong = images_unsup[1] * valid_mask_mix + images_unsup[0] * (1 - valid_mask_mix)
                images_unsup_strong = images_unsup_strong.unsqueeze(0)

        #student model forward
        preds_student_sup = model(images_sup)
        loss_sup_student = criterion(preds_student_sup,labels_sup)/ world_size 

        #teacher model forward
        with torch.no_grad():
            preds_teacher_sup = model_teacher(images_sup)
            preds_teacher_sup = preds_teacher_sup[0].detach()
            preds_teacher_unsup = model_teacher(images_unsup_weak)
            preds_teacher_unsup = preds_teacher_unsup[0].detach()
            if cutmix:
                if prob >0.5:
                    preds_teacher_unsup = preds_teacher_unsup[0] * valid_mask_mix + preds_teacher_unsup[1] * (1 - valid_mask_mix)
                else:
                    preds_teacher_unsup = preds_teacher_unsup[1] * valid_mask_mix + preds_teacher_unsup[0] * (1 - valid_mask_mix)
                preds_teacher_unsup = preds_teacher_unsup.unsqueeze(0)
            if acm:    
                valid_mask_mix = generate_cutmix_mask(preds_teacher_unsup[1].max(0)[1].cpu().numpy(), sample_cat, area_thresh, 
                                    no_pad=no_pad, no_slim=no_slim)
                images_unsup_strong = images_unsup[0] * (1 - valid_mask_mix) + images_unsup[1] * valid_mask_mix 
                #update cutmix bank
                cutmix_bank = update_cutmix_bank(cutmix_bank, preds_teacher_unsup, img_id, sample_id, area_thresh2)
                preds_teacher_unsup = preds_teacher_unsup[0] * (1-valid_mask_mix) + preds_teacher_unsup[1] * valid_mask_mix

                preds_teacher_unsup = preds_teacher_unsup.unsqueeze(0)
                images_unsup_strong = images_unsup_strong.unsqueeze(0)

            #compute consistency loss
            logits_teacher_sup = preds_teacher_sup.max(1)[1]            
            conf_sup = F.softmax(preds_teacher_sup, dim=1).max(1)[0]
            conf_teacher_sup_map = conf_sup
           
            logits_teacher_sup[conf_teacher_sup_map < threshold] = 255
            conf_unsup = F.softmax(preds_teacher_unsup, dim=1).max(1)[0]
            logits_teacher_unsup = preds_teacher_unsup.max(1)[1]
            if not cutmix and not acm:
                logits_teacher_unsup += valid_mask
                logits_teacher_unsup[logits_teacher_unsup > 20] = 255

            conf_teacher_unsup_map = conf_unsup
            logits_teacher_unsup[conf_teacher_unsup_map < threshold] = 255
        
        preds_student_unsup = model(images_unsup_strong)  
        with torch.no_grad():
            if acp or acm or sample:
                category_entropy = cal_category_confidence(preds_student_sup[0].detach(), preds_student_unsup[0].detach(), labels_sup, preds_teacher_unsup, num_classes)
                # perform momentum update
                class_criterion = class_criterion * class_momentum + category_entropy * (1 - class_momentum)
        if isinstance(criterion_cons, torch.nn.CrossEntropyLoss):
            loss_consistency1 = criterion_cons(preds_student_sup[0],logits_teacher_sup)/world_size
            loss_consistency2 = criterion_cons(preds_student_unsup[0],logits_teacher_unsup)/world_size
        elif sample:
            loss_consistency1 = criterion_cons(preds_student_sup[0],conf_sup, logits_teacher_sup, class_criterion[0])/world_size
            loss_consistency2 = criterion_cons(preds_student_unsup[0],conf_unsup, logits_teacher_unsup, class_criterion[0])/world_size
        else:
            loss_consistency1 = criterion_cons(preds_student_sup[0],conf_sup, logits_teacher_sup)/world_size
            loss_consistency2 = criterion_cons(preds_student_unsup[0],conf_unsup, logits_teacher_unsup)/world_size
        loss_consistency = loss_consistency1 + loss_consistency2
        loss = loss_sup_student + consist_weight * loss_consistency
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # get the output produced by model
        output = preds_student_sup[0] if cfg['net'].get('aux_loss', False) else preds_student_sup
        output = output.data.max(1)[1].cpu().numpy()
        target = labels_sup.cpu().numpy()
       
        # start to calculate miou
        intersection, union, target = intersectionAndUnion(output, target, num_classes, ignore_label)

        # gather all validation information
        reduced_intersection = torch.from_numpy(intersection).cuda()
        reduced_union = torch.from_numpy(union).cuda()
        reduced_target = torch.from_numpy(target).cuda()

        dist.all_reduce(reduced_intersection)
        dist.all_reduce(reduced_union)
        dist.all_reduce(reduced_target)

        intersection_meter.update(reduced_intersection.cpu().numpy())
        union_meter.update(reduced_union.cpu().numpy())
        target_meter.update(reduced_target.cpu().numpy())

        # gather all loss from different gpus
        reduced_loss = loss.clone()
        dist.all_reduce(reduced_loss)
        losses.update(reduced_loss.item())

        # update teacher model with EMA
        ema_decay = min(1-1/(i_iter+1),ema_decay_origin)
        for t_params, s_params in zip(model_teacher.parameters(), model.parameters()):
            t_params.mul_(ema_decay).add_(1-ema_decay, s_params.data)

        if i_iter % 50 == 0 and rank==0:
            iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
            accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            logger.info('iter = {} of {} completed, LR = {} loss = {}, mIoU = {}'
                        .format(i_iter, cfg['trainer']['epochs']*len(data_loader_unsup), lr, losses.avg, mIoU))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    if rank == 0:
        logger.info('=========epoch[{}]=========,Train mIoU = {}'.format(epoch, mIoU))
    if class_criterion is not None and cutmix_bank is None:
        return labeled_epoch, class_criterion
    elif cutmix_bank is not None:
        return labeled_epoch, class_criterion, cutmix_bank
    else:
        return labeled_epoch

def validate(model_teacher,model_student, data_loader, epoch):
    model_teacher.eval()
    model_student.eval()
    data_loader.sampler.set_epoch(epoch)

    num_classes, ignore_label = cfg['net']['num_classes'], cfg['dataset']['ignore_label']
    rank, world_size = get_rank(), get_world_size()

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    # meters for student 
    intersection_meter_student = AverageMeter()
    union_meter_student = AverageMeter()
    target_meter_student = AverageMeter()

    for step, batch in enumerate(data_loader):
        images, labels = batch
        images = images.cuda()
        labels = labels.long().cuda()
        with torch.no_grad():
            preds = model_teacher(images)
            preds_student = model_student(images)

        # get the output produced by model_teacher
        output = preds[0] if cfg['net'].get('aux_loss', False) else preds
        output = output.data.max(1)[1].cpu().numpy()
        target_origin = labels.cpu().numpy()

        # start to calculate miou
        intersection, union, target = intersectionAndUnion(output, target_origin, num_classes, ignore_label)

        # gather all validation information
        reduced_intersection = torch.from_numpy(intersection).cuda()
        reduced_union = torch.from_numpy(union).cuda()
        reduced_target = torch.from_numpy(target).cuda()

        dist.all_reduce(reduced_intersection)
        dist.all_reduce(reduced_union)
        dist.all_reduce(reduced_target)

        intersection_meter.update(reduced_intersection.cpu().numpy())
        union_meter.update(reduced_union.cpu().numpy())
        target_meter.update(reduced_target.cpu().numpy())

        # get the output produced by model_student
        output_student = preds_student[0] if cfg['net'].get('aux_loss',False) else preds_student
        output_student = output_student.data.max(1)[1].cpu().numpy()
        intersection_s, union_s, target_s = intersectionAndUnion(output_student,target_origin,num_classes,ignore_label)
        reduced_intersection_s = torch.from_numpy(intersection_s).cuda()
        reduced_union_s = torch.from_numpy(union_s).cuda()
        reduced_target_s = torch.from_numpy(target_s).cuda()

        dist.all_reduce(reduced_intersection_s)
        dist.all_reduce(reduced_union_s)
        dist.all_reduce(reduced_target_s)
        intersection_meter_student.update(reduced_intersection_s.cpu().numpy())
        union_meter_student.update(reduced_union_s.cpu().numpy())
        target_meter_student.update(reduced_target_s.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)

    iou_class_student = intersection_meter_student.sum / (union_meter_student.sum + 1e-10)
    accuracy_class_student = intersection_meter_student.sum / (target_meter_student.sum + 1e-10)
    mIoU_student = np.mean(iou_class_student)
    
    if rank == 0:
        print('teacher mIoU', mIoU)
        print('student mIoU', mIoU_student)

    if rank == 0:
        logger.info('=========epoch[{}]=========,Val_Teacher mIoU = {}'.format(epoch, mIoU))
        logger.info('=========epoch[{}]=========,Val_Student mIoU = {}'.format(epoch, mIoU_student))
    #logger.info('=========epoch[{}]=========,IoU for novel classes = {}'.format(epoch, novel_IoU))
    torch.save(mIoU, 'eval_metric.pth.tar')
    return mIoU


if __name__ == '__main__':
    main()
    
