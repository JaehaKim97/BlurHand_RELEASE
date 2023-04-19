import math
import os
import os.path as osp
import torch
import torchvision.transforms as transforms

from data.multiple_dataset import MultipleDatasets
from glob import glob
from torch.nn.parallel.data_parallel import DataParallel
from torch.utils.data import DataLoader
from utils.options import parse_options, copy_opt_file
from utils.logger import ColorLogger, init_tb_logger
from utils.misc import mkdir_and_rename
from utils.timer import Timer
from models.blurhandnet import BlurHandNet


def main():
    # load opt and args from yaml
    opt, args = parse_options()
    
    # dynamic dataset import
    dataset_list = opt['dataset_list']
    for _, opt_data in dataset_list.items():
        dataset_name = opt_data['name']
        exec(f'from data.{dataset_name} import {dataset_name}')
    globals().update(locals())
    
    trainer = Trainer(opt, args)
    trainer._make_batch_generator()
    trainer._prepare_training()

    # training
    end_epoch = opt['train']['end_epoch']
    for epoch in range(trainer.start_epoch, (end_epoch+1)):
        
        trainer.set_lr(epoch)
        trainer.tot_timer.tic()
        trainer.read_timer.tic()
        for itr, (inputs, targets, meta_info) in enumerate(trainer.batch_generator):
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()

            # forward
            trainer.optimizer.zero_grad()
            loss = trainer.model(inputs, targets, meta_info, 'train')
            loss = {k:loss[k].mean() for k in loss}

            # tensorboard logging
            tot_iter = epoch * trainer.itr_per_epoch + itr
            for k,v in loss.items():
                trainer.tb_logger.add_scalar(k, v, tot_iter)
                
            # backward
            sum(loss[k] for k in loss).backward()
            trainer.optimizer.step()
            trainer.gpu_timer.toc()
            screen = [
                'Epoch %d/%d itr %d/%d:' % (epoch, end_epoch, itr, trainer.itr_per_epoch),
                'lr: %g' % (trainer.get_lr()),
                'speed: %.2f(%.2fs r%.2f)s/itr' % (
                    trainer.tot_timer.average_time, trainer.gpu_timer.average_time, trainer.read_timer.average_time),
                '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
                ]
            screen += ['%s: %.4f' % ('loss_' + k, v.detach()) for k,v in loss.items()]
            trainer.logger.info(' '.join(screen))

            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()
        
        trainer.save_state({'epoch': epoch, 'network': trainer.model.state_dict(),
                            'optimizer': trainer.optimizer.state_dict()}, epoch)


class Trainer():
    def __init__(self, opt, args):
        self.opt = opt
        self.exp_dir = osp.join('experiments', opt['name'])
        self.tb_dir = osp.join('tb_logger', opt['name'])
        self.cur_epoch = 0
        
        # timers
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()
        
        # directories
        if not opt.get('continue_train', False):
            mkdir_and_rename(osp.join(self.exp_dir))
            mkdir_and_rename(osp.join(self.tb_dir))
        
        # logger
        self.logger = ColorLogger(self.exp_dir, log_name='train_logs.txt')
        self.tb_logger = init_tb_logger(self.tb_dir)
        
        # copy the yml file to the experiment root
        copy_opt_file(args.opt, self.exp_dir)
        
    def _make_batch_generator(self):
        # data loader and construct batch generator
        trainset3d_loader = []
        trainset2d_loader = []
        
        dataset_list = self.opt['dataset_list']
        for _, opt_data in dataset_list.items():
            dataset_name = opt_data['name']
            self.logger.info(f"Creating dataset ... [{dataset_name}]")
            if opt_data.get('is_3d', False):
                trainset3d_loader.append(eval(dataset_name)(self.opt, opt_data, transforms.ToTensor(), "train"))
            else:
                trainset2d_loader.append(eval(dataset_name)(self.opt, opt_data, transforms.ToTensor(), "train"))
        
        # dataloader for validation
        valid_loader_num = 0
        if len(trainset3d_loader) > 0:
            trainset3d_loader = [MultipleDatasets(trainset3d_loader, make_same_len=False)]
            valid_loader_num += 1
        else:
            trainset3d_loader = []
        if len(trainset2d_loader) > 0:
            trainset2d_loader = [MultipleDatasets(trainset2d_loader, make_same_len=False)]
            valid_loader_num += 1
        else:
            trainset2d_loader = []
        if valid_loader_num > 1:
            trainset_loader = MultipleDatasets(trainset3d_loader + trainset2d_loader, make_same_len=True)
        else:
            trainset_loader = MultipleDatasets(trainset3d_loader + trainset2d_loader, make_same_len=False)

        num_gpus = self.opt['num_gpus']
        num_threads = self.opt['num_threads']
        train_batch_size = self.opt['train']['batch_size']
        self.itr_per_epoch = math.ceil(len(trainset_loader) / num_gpus / train_batch_size)
        self.batch_generator = DataLoader(dataset=trainset_loader, batch_size=(num_gpus*train_batch_size),
                                          shuffle=True, num_workers=num_threads, pin_memory=True, drop_last=True)

    def _prepare_training(self):
        # prepare network and optimizer
        self.logger.info("Creating network and optimizer ... [seed {}]".format(self.opt['manual_seed']))
        model = BlurHandNet(self.opt)
        model = DataParallel(model).cuda()
        optimizer = self.get_optimizer(model)
        
        # if continue training, load the most recent training state
        if self.opt.get('continue_train', False):
            start_epoch, model, optimizer = self.continue_train(model, optimizer)
        else:
            start_epoch = 1
        model.train()

        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer

    def save_state(self, state, epoch):
        os.makedirs(osp.join(self.exp_dir, 'training_states'), exist_ok=True)
        file_path = osp.join(self.exp_dir, 'training_states', 'epoch_{:02d}.pth.tar'.format(epoch))

        # do not save human model layer weights
        dump_key = []
        for k in state['network'].keys():
            if 'smpl_layer' in k or 'mano_layer' in k or 'flame_layer' in k:
                dump_key.append(k)
        for k in dump_key:
            state['network'].pop(k, None)

        torch.save(state, file_path)
        self.logger.info("Saving training states into {}".format(file_path))

    def continue_train(self, model, optimizer):
        states_list = glob(osp.join(self.exp_dir, 'training_states', '*.pth.tar'))
        
        # find the most recent training state
        cur_epoch = max([int(file_name[file_name.find('epoch_') + 6:file_name.find('.pth.tar')])
                         for file_name in states_list])
        ckpt_path = osp.join(self.exp_dir, 'training_states', 'epoch_' + '{:02d}'.format(cur_epoch) + '.pth.tar')
        ckpt = torch.load(ckpt_path) 
        
        start_epoch = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['network'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer'])
        
        self.logger.info('Load checkpoint from {}'.format(ckpt_path))
        
        return start_epoch, model, optimizer

    def get_optimizer(self, model):
        total_params = []
        for module in model.module.trainable_modules:
            total_params += list(module.parameters())
        optimizer = torch.optim.Adam(total_params, lr=self.opt['train']['optim']['lr'])
        
        return optimizer
    
    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']
            
        return cur_lr

    def set_lr(self, epoch):
        # manual lr schedueling
        lr = self.opt['train']['optim']['lr']
        lr_dec_epoch = self.opt['train']['optim']['lr_dec_epoch']
        lr_dec_factor = self.opt['train']['optim']['lr_dec_factor']
        
        for e in lr_dec_epoch:
            if epoch < e:
                break
        if epoch < lr_dec_epoch[-1]:
            idx = lr_dec_epoch.index(e)
            for g in self.optimizer.param_groups:
                g['lr'] = lr / (lr_dec_factor ** idx)
        else:
            for g in self.optimizer.param_groups:
                g['lr'] = lr / (lr_dec_factor ** len(lr_dec_epoch))


if __name__ == "__main__":
    main()
