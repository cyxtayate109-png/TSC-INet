import argparse
import csv
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import platform
import os
import glob
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import tsi.builder
from torch.utils.tensorboard import SummaryWriter
from dataset import get_pretraining_set
from moudle import *

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=551, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[450], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

parser.add_argument('--checkpoint-path', default='./checkpoints/pretrain/', type=str)
parser.add_argument('--pos-neg-path', default='./KDE', type=str)
parser.add_argument('--skeleton-representation', type=str,
                    help='input skeleton-representation  for self supervised training (joint or motion or bone)')
parser.add_argument('--pre-dataset', default='ntu60', type=str,
                    help='which dataset to use for self supervised training (ntu60 or ntu120)')
parser.add_argument('--protocol', default='cross_subject', type=str,
                    help='training protocol cross_view/cross_subject/cross_setup')

# specific configs:
parser.add_argument('--encoder-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--encoder-k', default=8192, type=int,
                    help='queue size; number of negative keys')
parser.add_argument('--encoder-m', default=0.999, type=float,
                    help='momentum of updating key encoder')
parser.add_argument('--encoder-t', default=0.2, type=float,
                    help='softmax temperature')

parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')

parser.add_argument('--gpu', default=0)


# beta sensitivity analysis
parser.add_argument('--beta-analyze', action='store_true', help='enable beta sensitivity analysis for 4 branches')
parser.add_argument('--beta-write-csv', action='store_true', help='write beta sensitivity stats to CSV')
parser.add_argument('--evaluate-only', action='store_true', help='only evaluate beta sensitivity without training')
parser.add_argument('--eval-batches', default=50, type=int, help='number of batches to evaluate in --evaluate-only mode')

# distributed arguments
parser.add_argument('--dist', action='store_true', help='enable DistributedDataParallel')
parser.add_argument('--dist-backend', default=None, type=str, help='ddp backend: nccl/gloo')
parser.add_argument('--dist-init', default='env://', type=str, help='ddp init method, default env:// (torchrun)')


def main():
    args = parser.parse_args()

    if platform.system() == 'Windows' and args.workers and args.workers > 0:
        print('[Info] Windows detected: forcing --workers 0 to avoid multiprocessing pickling issues.')
        args.workers = 0


    ks_env = os.environ.get('KS', '3')
    kt_env = os.environ.get('KT', '5')
    best_weight_name = f"ks_{ks_env}_kt_{kt_env}.pt"
    best_weight_path = os.path.join(args.checkpoint_path, best_weight_name)

    # init distributed (torchrun)
    is_distributed = bool(args.dist)
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = 1
    rank = 0
    if is_distributed:
        backend = args.dist_backend
        if backend is None:
            backend = 'gloo' if platform.system() == 'Windows' else 'nccl'
        dist.init_process_group(backend=backend, init_method=args.dist_init)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        print(f"[DDP] backend={backend}, world_size={world_size}, rank={rank}, local_rank={local_rank}")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # pretraining dataset and protocol
    from options import options_pretraining as options 
    if args.pre_dataset == 'ntu60' and args.protocol == 'cross_view':
        opts = options.opts_ntu_60_cross_view()
    elif args.pre_dataset == 'ntu60' and args.protocol == 'cross_subject':
        opts = options.opts_ntu_60_cross_subject()
    elif args.pre_dataset == 'ntu120' and args.protocol == 'cross_setup':
        opts = options.opts_ntu_120_cross_setup()
    elif args.pre_dataset == 'ntu120' and args.protocol == 'cross_subject':
        opts = options.opts_ntu_120_cross_subject()
    elif args.pre_dataset == 'pku_part1' and args.protocol == 'cross_subject':
        opts = options.opts_pku_part1_cross_subject()
    elif args.pre_dataset == 'pku_part2' and args.protocol == 'cross_subject':
        opts = options.opts_pku_part2_cross_subject()

    opts.train_feeder_args['input_representation'] = args.skeleton_representation

    # create model
    print("=> creating model")
    device = torch.device('cuda', local_rank) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    model = tsi.builder.TSI_CNet(opts.encoder_args, args.encoder_dim, args.encoder_k, args.encoder_m, args.encoder_t)
    model = model.to(device)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None,
                    output_device=local_rank if torch.cuda.is_available() else None,
                    find_unused_parameters=False)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    pos_scores = {k: [] for k in ['ti', 'si', 'it', 'is']}
    neg_scores = {k: [] for k in ['ti', 'si', 'it', 'is']}
    
    # beta sensitivity tracking
    if args.beta_analyze:
        beta_history = {'ti': [], 'si': [], 'it': [], 'is': []}

    # Data loading code
    train_dataset = get_pretraining_set(opts)

    train_sampler = None
    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    writer = None
    is_main = (not is_distributed) or (rank == 0)
    if is_main:
        writer = SummaryWriter(args.checkpoint_path)

    best_acc = -1.0
    
    # Evaluate-only mode for beta analysis
    if args.evaluate_only and args.beta_analyze:
        print("Running beta sensitivity evaluation on validation data...")
        beta_history = {'ti': [], 'si': [], 'it': [], 'is': []}
        
        # Run a few batches to collect beta statistics
        model.eval()
        with torch.no_grad():
            for i, (q_input, k_input) in enumerate(train_loader):
                if i >= args.eval_batches:  # Collect data from specified number of batches
                    break
                    
                q_input = q_input.float().to(device, non_blocking=True)
                k_input = k_input.float().to(device, non_blocking=True)
                
                output1, output2, output3, output4, target1, target2, target3, target4 = model(q_input, k_input)
                
                L_ti = criterion(output1, target1)
                L_si = criterion(output2, target2)
                L_it = criterion(output3, target3)
                L_is = criterion(output4, target4)
                
                beta_values = [L_ti.item(), L_si.item(), L_it.item(), L_is.item()]
                beta_names = ['ti', 'si', 'it', 'is']
                
                for name, val in zip(beta_names, beta_values):
                    beta_history[name].append(val)
                
                if i % 10 == 0:
                    print(f"[Beta Eval] Batch {i}: " + 
                          " ".join([f"{name}={val:.4f}" for name, val in zip(beta_names, beta_values)]))
        
        # Skip training loop, go directly to analysis
    else:
        for epoch in range(args.start_epoch, args.epochs):

            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            adjust_learning_rate(optimizer, epoch, args)
            if is_main and writer is not None:
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step=epoch)
            # train for one epoch
            # loss, acc1 = train(train_loader, model, criterion, optimizer, epoch, args)
            if args.beta_analyze:
                loss, acc1 = train(train_loader, model, criterion, optimizer,
                                   epoch, args, pos_scores, neg_scores, device, beta_history)
            else:
                loss, acc1 = train(train_loader, model, criterion, optimizer,
                                   epoch, args, pos_scores, neg_scores, device)
            if is_main and writer is not None:
                writer.add_scalar('train_loss', loss.avg, global_step=epoch)
                writer.add_scalar('acc', acc1.avg, global_step=epoch)

            # 以訓練期內的 top1 作為準確率指標
            try:
                current_acc = float(acc1.avg)
            except Exception:
                current_acc = acc1.avg.item() if hasattr(acc1.avg, 'item') else acc1.avg

            is_best = current_acc > best_acc
            if is_best and is_main:
                best_acc = current_acc

                os.makedirs(args.checkpoint_path, exist_ok=True)
                state_dict = model.module.state_dict() if is_distributed else model.state_dict()
                torch.save(state_dict, best_weight_path)

    try:

        if is_main:
            for fp in glob.glob(os.path.join(args.checkpoint_path, 'checkpoint_epoch_*.pth.tar')):
                try:
                    os.remove(fp)
                except Exception as e:
                    print(f"[Warn] 刪除檔案失敗: {fp} -> {e}")
            mb = os.path.join(args.checkpoint_path, 'model_best.pth.tar')
            if os.path.exists(mb):
                try:
                    os.remove(mb)
                except Exception as e:
                    print(f"[Warn] 刪除檔案失敗: {mb} -> {e}")
    except Exception as e:
        print(f"[Warn] 清理非最佳 checkpoint 失敗: {e}")

    if is_distributed:
        dist.destroy_process_group()

    # Beta sensitivity final analysis
    if args.beta_analyze and is_main:
        print("\n" + "="*80)

        print("="*80)
        
        import numpy as np
        
        # Check if we have beta data
        if not beta_history or not any(beta_history.values()):
            print("No beta sensitivity data collected. Make sure --beta-analyze was used during training.")
            print("="*80)
            return

        beta_stats = {}
        for branch in ['ti', 'si', 'it', 'is']:
            values = np.array(beta_history[branch])
            if len(values) == 0:
                print(f"Warning: No data for branch {branch}")
                continue
                
            beta_stats[branch] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'max': np.max(values),
                'min': np.min(values),
                'final_avg': np.mean(values[-50:]) if len(values) >= 50 else np.mean(values),  # last 50 iterations
                'count': len(values)
            }
        
        if not beta_stats:
            print("No valid beta statistics found.")
            print("="*80)
            return
            
        # Sort by mean sensitivity (higher = more important)
        sorted_branches = sorted(beta_stats.items(), key=lambda x: x[1]['mean'], reverse=True)
        
        print(f"\nRANKING BY AVERAGE SENSITIVITY (Loss Value):")
        print("-" * 60)
        for rank, (branch, stats) in enumerate(sorted_branches, 1):
            print(f"{rank}. {branch.upper()}: avg={stats['mean']:.4f} std={stats['std']:.4f} max={stats['max']:.4f} count={stats['count']}")
        
        print(f"\nBRANCH INTERPRETATIONS:")
        print("-" * 40)
        interpretations = {
            'ti': 'Temporal-Instance: Time domain vs Instance level contrast',
            'si': 'Spatial-Instance: Space domain vs Instance level contrast', 
            'it': 'Instance-Temporal: Instance level vs Time domain contrast',
            'is': 'Instance-Spatial: Instance level vs Space domain contrast'
        }
        for branch, desc in interpretations.items():
            sensitivity = beta_stats[branch]['mean']
            print(f"{branch.upper()}: {desc}")
            print(f"      Sensitivity: {sensitivity:.4f} ({'High' if sensitivity > 6.0 else 'Medium' if sensitivity > 4.0 else 'Low'})")
        
        print(f"\nRECOMMENDATIONS:")
        print("-" * 20)
        top_branch = sorted_branches[0][0]
        top_value = sorted_branches[0][1]['mean']
        bottom_branch = sorted_branches[-1][0]
        bottom_value = sorted_branches[-1][1]['mean']
        
        ratio = top_value / bottom_value
        print(f"• Most important branch: {top_branch.upper()} (avg loss: {top_value:.4f})")
        print(f"• Least important branch: {bottom_branch.upper()} (avg loss: {bottom_value:.4f})")
        print(f"• Sensitivity ratio: {ratio:.2f}x difference")
        
        if ratio > 1.2:
            print(f"• Suggestion: Increase β_{top_branch} weight in future experiments")
            print(f"• Suggestion: Consider reducing β_{bottom_branch} weight")
        else:
            print(f"• All branches show similar importance - balanced weighting recommended")
        
        # Save comprehensive report
        if args.beta_write_csv:
            os.makedirs('./beta_stats', exist_ok=True)
            report_path = './beta_stats/beta_final_report.csv'
            with open(report_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['branch', 'rank', 'mean_loss', 'std_loss', 'max_loss', 'min_loss', 'final_avg', 'count'])
                for rank, (branch, stats) in enumerate(sorted_branches, 1):
                    writer.writerow([branch, rank, stats['mean'], stats['std'], 
                                   stats['max'], stats['min'], stats['final_avg'], stats['count']])
            print(f"\n• Final report saved to: {report_path}")
        
        print("="*80)

    # Only save scores if we have training data (not in evaluate-only mode)
    if not args.evaluate_only:
        save_scores_only(pos_scores, neg_scores, args.pos_neg_path)

def train(train_loader, model, criterion, optimizer, epoch, args, pos_scores, neg_scores, device, beta_history=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, top1, ],
        prefix="Epoch: [{}] Lr_rate [{}]".format(epoch, optimizer.param_groups[0]['lr']))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (q_input, k_input) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # print(f"q_input shape: {q_input.shape}, k_input shape: {k_input.shape}")
        # q_input = motion_att_temp_mask(q_input, mask_frame=10)
        # k_input = central_spacial_mask(k_input, mask_joint=5)

        q_input = q_input.float().to(device, non_blocking=True)
        k_input = k_input.float().to(device, non_blocking=True)


        # compute output
        output1, output2, output3, output4, target1, target2, target3, target4 = model(q_input, k_input)

        batch_size = output2.size(0)

        # interactive level loss - compute individual branch losses for beta analysis
        L_ti = criterion(output1, target1)  # temporal-instance branch
        L_si = criterion(output2, target2)  # spatial-instance branch  
        L_it = criterion(output3, target3)  # instance-temporal branch
        L_is = criterion(output4, target4)  # instance-spatial branch
        
        # total loss (beta=1 for all branches)
        loss = L_ti + L_si + L_it + L_is
        
        # beta sensitivity analysis: ∂L/∂βi = L_i (since βi=1)
        if args.beta_analyze and beta_history is not None:
            beta_values = [L_ti.item(), L_si.item(), L_it.item(), L_is.item()]
            beta_names = ['ti', 'si', 'it', 'is']
            
            # record beta history for final analysis
            for name, val in zip(beta_names, beta_values):
                beta_history[name].append(val)
            
            if i % args.print_freq == 0:
                print(f"[Beta] Epoch {epoch} Iter {i}: " + 
                      " ".join([f"{name}={val:.4f}" for name, val in zip(beta_names, beta_values)]))
                
                # write to CSV if requested
                if args.beta_write_csv:
                    os.makedirs('./beta_stats', exist_ok=True)
                    csv_path = f"./beta_stats/beta_epoch{epoch}_iter{i}.csv"
                    with open(csv_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['branch', 'loss_value', 'sensitivity'])
                        for name, val in zip(beta_names, beta_values):
                            writer.writerow([name, val, val])  # sensitivity = loss value when beta=1

        losses.update(loss.item(), batch_size)

        # measure accuracy of model m1 and m2 individually
        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, _ = accuracy(output2, target2, topk=(1, 5))
        top1.update(acc1[0], batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        
        if epoch >= args.epochs - 2:      
            pos_scores['ti'].append(model.last_pti.cpu())
            pos_scores['si'].append(model.last_psi.cpu())
            pos_scores['it'].append(model.last_pit.cpu())
            pos_scores['is'].append(model.last_pis.cpu())

            neg_scores['ti'].append(model.last_nti.cpu())
            neg_scores['si'].append(model.last_nsi.cpu())
            neg_scores['it'].append(model.last_nit.cpu())
            neg_scores['is'].append(model.last_nis.cpu())

    return losses, top1

def validate(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix="Val Epoch: [{}]".format(epoch))
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (q_input, k_input) in enumerate(val_loader):
            q_input = q_input.float().cuda(non_blocking=True)
            k_input = k_input.float().cuda(non_blocking=True)
            output1, output2, output3, output4, target1, target2, target3, target4 = model(q_input, k_input)
            loss = criterion(output1, target1) + criterion(output2, target2) + criterion(output3, target3) + criterion(output4, target4)
            acc1, _ = accuracy(output2, target2, topk=(1, 5))
            losses.update(loss.item(), q_input.size(0))
            top1.update(acc1[0], q_input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                progress.display(i)
    return losses, top1

def save_scores_only(pos_scores, neg_scores, save_dir):

    os.makedirs(save_dir, exist_ok=True)
    for branch in ['ti', 'si', 'it', 'is']:
        torch.save(
            {
                'pos': torch.cat(pos_scores[branch]),
                'neg': torch.cat(neg_scores[branch])
            },
            os.path.join(save_dir, f'{branch}_sim_scores.pt')
        )
    print(f'[Info] similarity tensors saved to {save_dir}')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries), flush=True)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()