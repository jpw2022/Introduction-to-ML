import training as tr
import visualize as vis
import clitools as cli

import torch
import os

parser = cli.get_parser()
args = parser.parse_args()
if args.file:
    config_path = os.path.join('./config_files', args.file)
    with open(config_path, 'r') as f:
        options = f.read().split()
        args = parser.parse_args(options)

result = tr.train_by_args(args)
train_loss, train_acc, val_loss, val_acc = result

# plot and save the accuracy
if args.out:
    vis.plot_acc(train_acc, val_acc)
    vis.save_fig(args.out)
if args.save_val:
    torch.save({
                'val_acc': val_acc,
                'step_per_epoch': int(args.alpha * args.p ** args.k / args.batch_size),
                'alpha': args.alpha,
                }, "./ckpt/" + args.save_val + ".val")
