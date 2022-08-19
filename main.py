import argparse
import os
import time
import shutil
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from models import VideoModel
from utils.transforms import *
from utils.opts import parser
import utils.CosineAnnealingLR as CosineAnnealingLR
import utils.datasets_video as datasets_video
from utils.dataset import VideoDataset
from datetime import datetime
import os
import numpy
import pickle as pkl
from torch.cuda import amp


torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)



best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    if args.dataset == 'something-v1':
        num_class = 174
        args.rgb_prefix = ''
        rgb_read_format = "{:05d}.jpg"
    elif args.dataset == 'something-v2':
        num_class = 174
        args.rgb_prefix = ''
        rgb_read_format = "{:06d}.jpg"
    elif args.dataset == 'diving48':
        num_class = 48
        args.rgb_prefix = 'frame'
        rgb_read_format = "{:06d}.jpg"
    elif args.dataset == 'kinetics400':
        num_class = 400
        args.rgb_prefix = 'img_'
        rgb_read_format = "{:05d}.jpg"
    else:
        raise ValueError('Unknown dataset '+args.dataset)

    global model_dir

    model_dir = os.path.join('experiments', args.dataset, args.arch, datetime.now().strftime('%b%d_%H-%M-%S'))
    os.makedirs(model_dir)
    os.makedirs(os.path.join(model_dir, args.root_log))

    note_fl = open(model_dir + '/note.txt', 'a')
    note_fl.write('GSF:                   {}\n'
                  'Fusion:                {}\n'
                  'Channel ratio:         {}\n'
                  'Temp gates:            {}\n'
                  'Batch Size:            {}\n'
                  'Grad aggregation step: {}\n'
                  'Warmup:                {}\n'
                  'Epochs:                {}\n'
                  'Learning rate:         {}\n'
                  ''.format(args.gsf, args.gsf_ch_fusion, args.gsf_ch_ratio, args.gsf_temp_kern, args.batch_size,
                            args.iter_size, args.warmup, args.epochs, args.lr))
    note_fl.close()
    writer = SummaryWriter(model_dir)

    args.train_list, args.val_list, args.root_path, prefix = datasets_video.return_dataset(args.dataset, args.split)

    if 'something' in args.dataset:
        # label transformation for left/right categories
        target_transforms = {86: 87, 87: 86, 93: 94, 94: 93, 166: 167, 167: 166}
        print('Target transformation is enabled....')
    else:
        target_transforms = None

    args.store_name = '_'.join(
        [args.dataset, args.arch, 'segment%d' % args.num_segments])
    print('storing name: ' + args.store_name)

    model = VideoModel(num_class=num_class, num_segments=args.num_segments,
                       base_model=args.arch, consensus_type=args.consensus_type, dropout=args.dropout,
                       gsf=args.gsf, gsf_ch_ratio=args.gsf_ch_ratio,
                       target_transform=target_transforms)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    normalize = GroupNormalize(input_mean, input_std, div=(args.arch not in ['bninception', 'inceptionv3']))
    policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation()



    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = CosineAnnealingLR.WarmupCosineLR(optimizer=optimizer, milestones=[args.warmup, args.epochs],
                                                        warmup_iters=args.warmup, min_ratio=1e-7)
    scaler = amp.GradScaler(enabled=args.with_amp)

    model = model.cuda()
    model = torch.nn.DataParallel(model).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['model_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            for epoch in range(0, args.start_epoch):
                lr_scheduler.step()
            print(("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))
    
    train_transform = torchvision.transforms.Compose([
                        rand_augment_transform(config_str='rand-m9-mstd0.5', 
                                                hparams={'translate_const': 117, 'img_mean': (124, 116, 104)}
                                              ),
                        train_augmentation,
                        Stack(roll=(args.arch in ['bninception', 'inceptionv3'])),
                        ToTorchFormatTensor(),
                        normalize,
                                                    ])

    train_loader = torch.utils.data.DataLoader(
        VideoDataset(args.root_path, args.train_list, num_segments=args.num_segments,
                     image_tmpl=args.rgb_prefix+rgb_read_format,
                     transform=train_transform),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        VideoDataset(args.root_path, args.val_list, num_segments=args.num_segments,
                     image_tmpl=args.rgb_prefix+rgb_read_format,
                     random_shift=False,
                     transform=torchvision.transforms.Compose([
                               GroupScale(int(scale_size)),
                               GroupCenterCrop(crop_size),
                               Stack(roll=(args.arch in ['bninception', 'inceptionv3'])),
                               ToTorchFormatTensor(),
                               normalize,
                                                             ]), 
                    ),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        print('Standard CE loss')
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    log_training = open(os.path.join(model_dir, args.root_log, '%s.csv' % args.store_name), 'a')
    if args.evaluate:
        validate(val_loader, model, criterion, iter=0, epoch=args.start_epoch,
                 log=log_training, writer=writer)
        return

    for epoch in range(args.start_epoch, args.epochs):
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch + 1)
        train_prec1 = train(train_loader, model, criterion, optimizer, epoch, log_training, writer=writer, scaler=scaler)
        lr_scheduler.step()

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0:
            prec1 = validate(val_loader, model, criterion, (epoch + 1) * len(train_loader), log_training,
                                writer=writer, epoch=epoch)
            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'model_state_dict': model.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
                'best_prec1': best_prec1,
                'current_prec1': prec1,
                'lr': optimizer.param_groups[-1]['lr'],
            }, is_best, model_dir)
        else:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'model_state_dict': model.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
                'best_prec1': best_prec1,
                'current_prec1': train_prec1,
                'lr': optimizer.param_groups[-1]['lr'],
            }, False, model_dir)

def train(train_loader, model, criterion, optimizer, epoch, log, writer, scaler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    loss_summ = 0
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        
        # measure data loading time
        data_time.update(time.time() - end)
        
        target = target.cuda()
        input_var = input.cuda()
        
        # compute output
        #################amp########################################
        with amp.autocast(enabled=args.with_amp):
            output = model(input_var, args.with_amp)
            loss = criterion(output, target) / args.iter_size
            loss_summ += loss.data
            scaler.scale(loss).backward()
        
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        losses.update(loss_summ.data, input.size(0))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))

        if (i+1) % args.iter_size == 0:
            # scale down gradients when iter size is functioning
            if args.clip_gradient is not None:
                scaler.unscale_(optimizer)
                total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
                if total_norm > args.clip_gradient:
                    print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            loss_summ = 0

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr']))
            writer.add_scalar('train/batch_loss', losses.avg, epoch * len(train_loader) + i)
            writer.add_scalar('train/batch_top1Accuracy', top1.avg, epoch * len(train_loader) + i)
            print(output)
            log.write(output + '\n')
            log.flush()
    writer.add_scalar('train/loss', losses.avg, epoch + 1)
    writer.add_scalar('train/top1Accuracy', top1.avg, epoch + 1)
    writer.add_scalar('train/top5Accuracy', top5.avg, epoch + 1)
    return top1.avg


def validate(val_loader, model, criterion, iter, log, epoch, writer):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            
            target = target.cuda()
            input_var = input.cuda()

            # compute output
            with amp.autocast(enabled=args.with_amp):
                output = model(input_var, args.with_amp)
                loss = criterion(output, target)

            # measure accuracy and record loss
            losses.update(loss.data, input.size(0))
            prec1, prec5 = accuracy(output.data, target, topk=(1,5))
            top1.update(prec1, input.size(0))
            top5.update(prec5, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                            i, len(val_loader), batch_time=batch_time, loss=losses,
                            top1=top1, top5=top5))
                print(output)
                log.write(output + '\n')
                log.flush()

    output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
            .format(top1=top1, top5=top5, loss=losses))
    print(output)
    output_best = '\nBest Prec@1: %.3f'%(best_prec1)
    print(output_best)
    writer.add_scalar('test/loss', losses.avg, epoch + 1)
    writer.add_scalar('test/top1Accuracy', top1.avg, epoch + 1)
    writer.add_scalar('test/top5Accuracy', top5.avg, epoch + 1)
    log.write(output + ' ' + output_best + '\n')
    log.flush()

    return top1.avg

def save_checkpoint(state, is_best, model_dir):
    torch.save(state, '%s/%s_checkpoint.pth.tar' % (model_dir, args.store_name))
    if is_best:
        shutil.copyfile('%s/%s_checkpoint.pth.tar' % (model_dir, args.store_name),
                        '%s/%s_best.pth.tar' % (model_dir, args.store_name))

def save_checkpoint_epoch(state, epoch, model_dir):
    torch.save(state, '%s/%s_checkpoint_%d.pth.tar' % (model_dir, args.store_name, epoch))

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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
