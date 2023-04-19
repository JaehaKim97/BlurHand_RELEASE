import numpy as np
import os
import os.path as osp
import torch
import torchvision.transforms as transforms
import warnings

from models.blurhandnet import BlurHandNet
from torch.nn.parallel.data_parallel import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.logger import ColorLogger
from utils.options import parse_options, copy_opt_file

def main():
    # for visibility
    warnings.filterwarnings('ignore')
    
    # for reproducibility
    torch.backends.cudnn.deterministic = True
    
    # load opt and args from yaml
    opt, args = parse_options()
    
    # dynamic dataset import
    opt_data = opt['dataset']
    dataset_name = opt_data['name']
    exec(f'from data.{dataset_name} import {dataset_name}')
    globals().update(locals())

    tester = Tester(opt, args)
    tester._make_batch_generator()
    tester._prepare_testing()
    
    eval_result = {}
    cur_sample_idx = 0
    pbar = tqdm(tester.batch_generator)
    for inputs, targets, meta_info in pbar:
        # forward
        with torch.no_grad():
            out = tester.model(inputs, targets, meta_info, 'test')
        
        # save output
        out = {k: v.cpu().numpy() for k,v in out.items()}
        for k,v in out.items():
            batch_size = out[k].shape[0]
        out = [{k: v[b_idx] for k,v in out.items()} for b_idx in range(batch_size)]
        
        # evaluate
        cur_eval_result = tester._evaluate(out, cur_sample_idx)
        for k,v in cur_eval_result.items():
            if k in eval_result:
                if k != 'mpvpe':
                    for j in range(21): # Interhand
                        eval_result[k][j] += v[j]
                else:
                    eval_result[k] += v
            else: eval_result[k] = v
        cur_sample_idx += len(out)
        
        # pbar.set_description('MPJPE @ CURRENT {:.2f}'.format(np.mean(eval_result['mpvpe'])))
    
    evaluate_both_ends = opt['test'].get('evaluate_both_ends', False)
    tester._print_eval_result(eval_result, evaluate_both_ends)


class Tester():
    def __init__(self, opt, args):
        self.opt = opt
        self.exp_dir = osp.join('experiments', opt['name'])
        self.result_dir = osp.join('experiments', opt['name'], 'results')
        
        # directories
        os.makedirs(self.result_dir, exist_ok=True)

        # logger
        self.logger = ColorLogger(self.result_dir, log_name='test_log.txt')
        
        # copy the yml file to the experiment root
        copy_opt_file(args.opt, self.result_dir)

    def _make_batch_generator(self):
        # data loader and construct batch generator
        opt_data = self.opt['dataset']
        
        dataset_name = opt_data['name']
        self.logger.info(f"Creating dataset ... [{dataset_name}]")
        
        self.testset_loader = eval(dataset_name)(self.opt, opt_data, transforms.ToTensor(), "test")
        
        num_gpus = self.opt['num_gpus']
        num_threads = self.opt['num_threads']
        test_batch_size = self.opt['test']['batch_size']
        self.batch_generator = DataLoader(dataset=self.testset_loader, batch_size=(num_gpus*test_batch_size),
                                          shuffle=False, num_workers=num_threads, pin_memory=True)

    def _prepare_testing(self):
        # prepare network
        self.logger.info("Creating network ...")
        model = BlurHandNet(self.opt, weight_init=False)
        model = DataParallel(model).cuda()
        
        # load trained model
        file_path = osp.join(self.exp_dir, 'training_states', 'epoch_{:02d}.pth.tar'.format(self.opt['test']['epoch']))
        assert osp.exists(file_path), 'Cannot find training state at ' + file_path
        ckpt = torch.load(file_path)
        model.load_state_dict(ckpt['network'], strict=False)  # set strict=False due to MANO-related module
        model.eval()
        self.logger.info('Load checkpoint from {}'.format(file_path))

        self.model = model
    
    def _evaluate(self, outs, cur_sample_idx):
        eval_result = self.testset_loader.evaluate(outs, cur_sample_idx)
        
        return eval_result

    def _print_eval_result(self, eval_result, evaluate_both_ends):
        self.testset_loader.print_eval_result(self.logger, eval_result, evaluate_both_ends)


if __name__ == "__main__":
    main()