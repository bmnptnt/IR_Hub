from collections import OrderedDict
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam, AdamW

from models.select_network import define_G
from models.model_base import ModelBase
from models.loss import CharbonnierLoss, PerceptualLoss
from models.loss_ssim import SSIMLoss

from utils.utils_model import test_mode
from utils.utils_regularizers import regularizer_orth, regularizer_clip


class ModelPlain(ModelBase):
    """Train with pixel loss"""
    def __init__(self, opt):
        super(ModelPlain, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.opt_train = self.opt['train']    # training option
        self.netG = define_G(opt)
        self.netG = self.model_to_device(self.netG)
        if self.opt_train['E_decay'] > 0:
            self.netE = define_G(opt).to(self.device).eval()

    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.load()                           # load model
        self.netG.train()                     # set training mode,for BN
        self.define_loss()                    # define loss
        self.define_optimizer()               # define optimizer
        self.load_optimizers()                # load optimizer
        self.define_scheduler()               # define scheduler
        self.log_dict = OrderedDict()         # log

    # ----------------------------------------
    # load pre-trained G model
    # ----------------------------------------
    def load(self):
        load_path_G = self.opt['path']['pretrained_netG']
        if load_path_G is not None:
            print('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, strict=self.opt_train['G_param_strict'], param_key='params')
        load_path_E = self.opt['path']['pretrained_netE']
        if self.opt_train['E_decay'] > 0:
            if load_path_E is not None:
                print('Loading model for E [{:s}] ...'.format(load_path_E))
                self.load_network(load_path_E, self.netE, strict=self.opt_train['E_param_strict'], param_key='params_ema')
            else:
                print('Copying model for E ...')
                self.update_E(0)
            self.netE.eval()

    # ----------------------------------------
    # load optimizer
    # ----------------------------------------
    def load_optimizers(self):
        load_path_optimizerG = self.opt['path']['pretrained_optimizerG']
        if load_path_optimizerG is not None and self.opt_train['G_optimizer_reuse']:
            print('Loading optimizerG [{:s}] ...'.format(load_path_optimizerG))
            self.load_optimizer(load_path_optimizerG, self.G_optimizer)

    # ----------------------------------------
    # save model / optimizer(optional)
    # ----------------------------------------
    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
        if self.opt_train['E_decay'] > 0:
            self.save_network(self.save_dir, self.netE, 'E', iter_label)
        if self.opt_train['G_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.G_optimizer, 'optimizerG', iter_label)

    # ----------------------------------------
    # save best model
    # ----------------------------------------
    def save_best(self):
        self.save_network(self.save_best_dir, self.netG, 'G', 'best')
        if self.opt_train['E_decay'] > 0:
            self.save_network(self.save_best_dir, self.netE, 'E', 'best')
        if self.opt_train['G_optimizer_reuse']:
            self.save_optimizer(self.save_best_dir, self.G_optimizer, 'optimizerG', 'best')


    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):
        G_lossfn_type = self.opt_train['G_lossfn_type']
        if G_lossfn_type == 'l1':
            self.G_lossfn = nn.L1Loss().to(self.device)
        elif G_lossfn_type == 'l2':
            self.G_lossfn = nn.MSELoss().to(self.device)
        elif G_lossfn_type == 'l2sum':
            self.G_lossfn = nn.MSELoss(reduction='sum').to(self.device)
        elif G_lossfn_type == 'ssim':
            self.G_lossfn = SSIMLoss().to(self.device)
        elif G_lossfn_type == 'charbonnier':
            self.G_lossfn = CharbonnierLoss(self.opt_train['G_charbonnier_eps']).to(self.device)
        elif G_lossfn_type == 'huber':
            self.G_lossfn = nn.Huber()
        elif G_lossfn_type == 'freq_gen':
            self.G_lossfn = nn.L1Loss().to(self.device)
            self.F_lossfn = CharbonnierLoss(self.opt_train['G_charbonnier_eps']).to(self.device)
        elif G_lossfn_type == 'perceptual':
            self.G_lossfn == PerceptualLoss().to(self.device)
            self.G_lossfn.vgg = self.model_to_device(self.G_lossfn.vgg)
            self.G_lossfn.lossfn = self.G_lossfn.lossfn.to(self.device)
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_lossfn_type))
        self.G_lossfn_weight = self.opt_train['G_lossfn_weight']

    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        if self.opt_train['G_optimizer_type'] == 'adam':
            self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'],
                                    betas=self.opt_train['G_optimizer_betas'],
                                    weight_decay=self.opt_train['G_optimizer_wd'], capturable=True)
        elif self.opt_train['G_optimizer_type'] == 'adamw':
            self.G_optimizer = AdamW(G_optim_params, lr=self.opt_train['G_optimizer_lr'],
                                    betas=self.opt_train['G_optimizer_betas'],
                                    weight_decay=self.opt_train['G_optimizer_wd'], capturable=True)
        else:
            raise NotImplementedError

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        if self.opt_train['G_scheduler_type'] == 'MultiStepLR':
            self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                            self.opt_train['G_scheduler_milestones'],
                                                            self.opt_train['G_scheduler_gamma']
                                                            ))
        elif self.opt_train['G_scheduler_type'] == 'CosineAnnealingWarmRestarts':
            self.schedulers.append(lr_scheduler.CosineAnnealingWarmRestarts(self.G_optimizer,
                                                            self.opt_train['G_scheduler_periods'],
                                                            self.opt_train['G_scheduler_restart_weights'],
                                                            self.opt_train['G_scheduler_eta_min']
                                                            ))
        else:
            raise NotImplementedError

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data, need_H=True):
        self.L = data['L'].to(self.device)
        if need_H:
            self.H = data['H'].to(self.device)

    def feed_data_freq(self, data, need_H=True):
        self.L = data['L'].to(self.device)
        if need_H:
            self.H = data['H'].to(self.device)
            batch = (self.H).shape[0]
            fft_dim = (-2, -1)
            fft_H = torch.fft.rfftn(self.H, dim=fft_dim, norm='ortho')
            fft_H = torch.stack((fft_H.real, fft_H.imag), dim=-1)
            fft_H = fft_H.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
            fft_H = fft_H.view((batch, -1,) + fft_H.size()[3:])
            self.HF=fft_H

    def feed_data_patch(self, data, need_H=True,patch_num=4):
        self.L = data['L'].to(self.device)
        self.res_h=(self.L).shape[2]
        self.res_w = self.L.shape[3]
        self.L_patchs=[]

        for i in range (0,patch_num):
            for j in range(0, patch_num):
                self.L_patchs.append(self.L[:,:,i*(self.res_h//patch_num):(i+1)*(self.res_h//patch_num),j*(self.res_w//patch_num):(j+1)*(self.res_w//patch_num)])

        if need_H:
            self.H = data['H'].to(self.device)

    def feed_data_adaptive_patch(self, data, need_H=True):
        self.L = data['L'].to(self.device)
        self.res_h=(self.L).shape[2]
        self.res_w = self.L.shape[3]

        if self.res_h < 512 : self.patch_num_h=1
        elif self.res_h >= 512 and self.res_h < 1024: self.patch_num_h = 2
        elif self.res_h >= 1024 and self.res_h < 2048: self.patch_num_h = 4
        else : self.patch_num_h=8

        if self.res_w < 512 : self.patch_num_w=1
        elif self.res_w >= 512 and self.res_w < 1024: self.patch_num_w = 2
        elif self.res_w >= 1024 and self.res_w < 2048: self.patch_num_w = 4
        else : self.patch_num_w=8

        self.L_patchs=[]

        for i in range (0,self.patch_num_h):
            for j in range(0, self.patch_num_w):
                self.L_patchs.append(self.L[:,:,i*(self.res_h//self.patch_num_h):(i+1)*(self.res_h//self.patch_num_h),j*(self.res_w//self.patch_num_w):(j+1)*(self.res_w//self.patch_num_w)])

        if need_H:
            self.H = data['H'].to(self.device)

    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def netG_forward(self):
        self.E = self.netG(self.L)

    def netG_forward_vaunet(self):
        self.E, self.F, self.mu, self.log_var = self.netG(self.L)

    def netG_forward_adaptive_patch_vaunet(self):
        result_patchs=[]
        mu_sum = 0.0
        log_var_sum = 0.0
        for i, patch in enumerate(self.L_patchs):
            patch_out,_,mu_out,log_var_out=self.netG(patch)
            result_patchs.append(patch_out)
            mu_sum+=mu_out
            log_var_sum+=log_var_out

        merged_img=torch.tensor([]).to(self.device)
        for i in range(0,self.patch_num_h):
            merged_line=torch.tensor([]).to(self.device)
            for j in range(0,self.patch_num_w):
                merged_line=torch.cat((merged_line,result_patchs[self.patch_num_w*i+j]),axis=3)
            merged_img=torch.cat((merged_img,merged_line),axis=2)
        self.E = merged_img
        self.mu = mu_sum/len(result_patchs)
        self.log_var = log_var_sum/len(result_patchs)


    def netG_forward_test_patch(self,patch_num=4):
        result_patchs=[]
        for i, patch in enumerate(self.L_patchs):
            result=self.netG(patch)
            result_patchs.append(result)

        merged_img=torch.tensor([]).to(self.device)
        for i in range(0,patch_num):
            merged_line=torch.tensor([]).to(self.device)
            for j in range(0,patch_num):
                merged_line=torch.cat((merged_line,result_patchs[patch_num*i+j]),axis=3)
            merged_img=torch.cat((merged_img,merged_line),axis=2)
        self.E = merged_img


    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        self.G_optimizer.zero_grad()
        self.netG_forward()
        G_loss = self.G_lossfn_weight * self.G_lossfn(self.E, self.H)
        G_loss.backward()

        # ------------------------------------
        # clip_grad
        # ------------------------------------
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
        if G_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'], norm_type=2)

        self.G_optimizer.step()

        # ------------------------------------
        # regularizer
        # ------------------------------------
        G_regularizer_orthstep = self.opt_train['G_regularizer_orthstep'] if self.opt_train['G_regularizer_orthstep'] else 0
        if G_regularizer_orthstep > 0 and current_step % G_regularizer_orthstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_orth)
        G_regularizer_clipstep = self.opt_train['G_regularizer_clipstep'] if self.opt_train['G_regularizer_clipstep'] else 0
        if G_regularizer_clipstep > 0 and current_step % G_regularizer_clipstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_clip)

        # self.log_dict['G_loss'] = G_loss.item()/self.E.size()[0]  # if `reduction='sum'`
        self.log_dict['G_loss'] = G_loss.item()

        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters_vaunet(self, current_step):
        self.G_optimizer.zero_grad()
        self.netG_forward_vaunet()
        w_kl = 0.001
        w_freq = 0.565
        w_sr = 0.435


        loss_KL = torch.mean(-0.5 * torch.sum(1 + self.log_var - self.mu ** 2 - (self.log_var.exp()), dim=1), dim=0)
        # G_loss = self.G_lossfn_weight * (self.G_lossfn(self.E, self.H)* w_sr +self.F_lossfn(self.F,self.HF) * w_freq + loss_KL * w_kl )
        G_loss = self.G_lossfn_weight * (self.G_lossfn(self.E, self.H) + self.F_lossfn(self.F, self.HF) + loss_KL)
        G_loss.backward()

        # ------------------------------------
        # clip_grad
        # ------------------------------------
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
        if G_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'], norm_type=2)

        self.G_optimizer.step()

        # ------------------------------------
        # regularizer
        # ------------------------------------
        G_regularizer_orthstep = self.opt_train['G_regularizer_orthstep'] if self.opt_train['G_regularizer_orthstep'] else 0
        if G_regularizer_orthstep > 0 and current_step % G_regularizer_orthstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_orth)
        G_regularizer_clipstep = self.opt_train['G_regularizer_clipstep'] if self.opt_train['G_regularizer_clipstep'] else 0
        if G_regularizer_clipstep > 0 and current_step % G_regularizer_clipstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_clip)

        # self.log_dict['G_loss'] = G_loss.item()/self.E.size()[0]  # if `reduction='sum'`
        self.log_dict['G_loss'] = G_loss.item()

        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])

    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.netG_forward()
        self.netG.train()

    def test_patch(self,patch_num=4):
        self.netG.eval()
        with torch.no_grad():
            # self.netG_forward()
            self.netG_forward_test_patch(patch_num=patch_num)
        self.netG.train()
    def test_adaptive_patch(self):
        self.netG.eval()
        with torch.no_grad():
            # self.netG_forward()
            self.netG_forward_adaptive_patch_vaunet()
        self.netG.train()

    # ----------------------------------------
    # test / inference x8
    # ----------------------------------------
    def testx8(self):
        self.netG.eval()
        with torch.no_grad():
            self.E = test_mode(self.netG, self.L, mode=3, sf=self.opt['scale'], modulo=1)
        self.netG.train()

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # get L, E, H image
    # ----------------------------------------
    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach()[0].float().cpu()

        out_dict['E'] = self.E.detach()[0].float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach()[0].float().cpu()
        return out_dict

    # ----------------------------------------
    # get L, E, H batch images
    # ----------------------------------------
    def current_results(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach().float().cpu()
        out_dict['E'] = self.E.detach().float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach().float().cpu()
        return out_dict

    """
    # ----------------------------------------
    # Information of netG
    # ----------------------------------------
    """

    # ----------------------------------------
    # print network
    # ----------------------------------------
    def print_network(self):
        msg = self.describe_network(self.netG)
        print(msg)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.netG)
        print(msg)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.netG)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.netG)
        return msg
