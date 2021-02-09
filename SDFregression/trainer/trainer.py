import time

# import imageio
import numpy as np
import torch
# from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop
import datetime
import matplotlib.pyplot as plt
from scipy import ndimage


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, loss, metrics, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            if self.writer is not None:
                self.writer.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        torch.backends.cudnn.benchmark = False
        
        self.model.train()

        total_loss = 0
        # total_metrics = np.zeros(len(self.metrics))
        start_time = time.time()
        for batch_idx, sample_batched in enumerate(self.data_loader):
            img, lab = sample_batched['image'], sample_batched['label']

            img = img.to(self.device)
            lab = lab.to(self.device)

            img = img.permute(0, 4, 1, 2, 3)
            lab = lab.permute(0, 4, 1, 2, 3)

            self.optimizer.zero_grad()
            output = self.model(img)
            
            debug = False
            if debug:
                    #self.debug_training(img,lab,output,epoch,batch_idx,"training")
                    self.check_training_inputs(img,lab,output,epoch,batch_idx,"training")

#            Used to SDF regularization only:
#            target_np = output[0,:,:,:].cpu().detach().numpy()
#            df1 = ndimage.distance_transform_edt(target_np).reshape(target_np.shape)
#            df1 = 1-(df1/np.max(df1))
#            df2 = ndimage.distance_transform_edt(1-target_np).reshape(target_np.shape)
#            df2 = 1-(df2/np.max(df2))
#            inv_df = target_np*df1+(1-target_np)*df2
#            #regularizer_weight = np.stack((1-inv_df, inv_df),axis=0)
#            weight = torch.Tensor(inv_df).to(self.device)                       
#            loss = self.loss(output,lab,weight)
            
            loss = self.loss(output,lab)
            loss.backward()
            self.optimizer.step()

            # self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            if self.writer is not None:
                self.writer.writer.add_scalar('train/loss', loss.item())
            total_loss += loss.item()

            # TODO: Compute custom metrics
            # total_metrics += self._eval_metrics(output, target)

            time_per_test = (time.time() - start_time) / (batch_idx + 1)
            time_left = (self.len_epoch - batch_idx) * time_per_test

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} Time per batch: {:.5} Time left in epoch: {}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                    time_per_test,
                    str(datetime.timedelta(seconds=time_left))))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
            if batch_idx == self.len_epoch:
                break

        log = {
            'loss': total_loss / self.len_epoch,
            # 'metrics': (total_metrics / self.len_epoch).tolist()
        }
#        print('Debug saving checkpoint')  # TODO only for debug purposes
#        self._save_checkpoint(epoch, save_best=False)
        
        print('Doing validation')
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(val_log)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        n_validation = len(self.valid_data_loader)
        start_time = time.time()
        with torch.no_grad():
            for batch_idx, sample_batched in enumerate(self.valid_data_loader):
                img, lab = sample_batched['image'], sample_batched['label']
                # TODO: This transform should probably not be done here
#                data = data.permute(0, 3, 1, 2)  # from NHWC to NCHW

                img = img.to(self.device)
                lab = lab.to(self.device)
                
                img = img.permute(0, 4, 1, 2, 3)
                lab = lab.permute(0, 4, 1, 2, 3)                

                output = self.model(img)
                
                debug = False
                if debug:
                    #self.debug_training(img,lab,output,epoch,batch_idx,"validation")
                    self.check_training_inputs(img,lab,output,epoch,batch_idx,"validation")
                
#               Use for SDF regularization only:                
#                #loss = self.loss(output, torch.argmax(lab,dim=1))
#                target_np = output[0,:,:,:].cpu().detach().numpy()
#                df1 = ndimage.distance_transform_edt(target_np).reshape(target_np.shape)
#                df1 = 1-(df1/np.max(df1))
#                df2 = ndimage.distance_transform_edt(1-target_np).reshape(target_np.shape)
#                df2 = 1-(df2/np.max(df2))
#                inv_df = target_np*df1+(1-target_np)*df2
#                #regularizer_weight = np.stack((1-inv_df, inv_df),axis=0)
#                weight = torch.Tensor(inv_df).to(self.device)            
#                loss = self.loss(output,lab,weight)
                
                loss = self.loss(output,lab)

                # self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                # if self.writer is not None:
                #    self.writer.writer.add_scalar('validation/loss', loss.item())
                total_val_loss += loss.item()

                time_per_test = (time.time() - start_time) / (batch_idx + 1)
                time_left = (n_validation - batch_idx) * time_per_test

                if batch_idx % self.log_step == 0:
                    self.logger.debug(
                        'Validation: {}/{} Loss: {:.6f} Time per batch: {:.5} Time left in validation: {}'.format(
                            batch_idx,
                            n_validation,
                            loss,
                            time_per_test,
                            str(datetime.timedelta(seconds=time_left))))

                total_val_metrics += self._eval_metrics(output, lab)  # TODO: Add custom metrics
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #    self.writer.add_histogram(name, p, bins='auto')

        if self.writer is not None:
            avg_val_loss = total_val_loss / len(self.valid_data_loader)
            self.writer.writer.add_scalar('validation/loss', avg_val_loss, epoch)

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }
        
    def debug_training(self,img,lab,output,epoch,batch_idx,debug_string):
        # TODO: Debugging. Pls remove!
        #print("OUTPUT SIZE: ",output.size())
        out_np = output.cpu().detach().numpy()
        #print("OUTPUT: ", np.min(out_np), np.max(out_np))
        #print(out_np[0,0,32,32,:])
        #print("LABEL SIZE: ",lab.size())
        lab_np = lab.cpu().detach().numpy()
        #print(np.min(lab_np), np.max(lab_np))
        #print(np.unique(lab_np))
        #print(lab_np[0,0,32,32,:])
        img_np = img.cpu().detach().numpy()
        
        #print("======= DEBUGGING =======: ", batch_idx)
        #debug_path = "C:/Users/kajul/Documents/LAA/Segmentation/PyTorch3DUnet/3D-Unet/debug/"+debug_string+"/"
        debug_path = "H:/SDF-regression/ImageSegmentation/roi-pytorch/debug/" + debug_string  + "/"
        #debug_path = "E:/roi-pytorch/debug/"+debug_string+"/"
#                np.save(debug_path+"Output_"+str(batch_idx), out_np)
#                np.save(debug_path+"Label_"+str(batch_idx), lab_np)
#                np.save(debug_path+"Image_"+str(batch_idx), img_np)
#                
        #pred = torch.argmax(output, dim=1)
        pred = torch.nn.functional.softmax(output,dim=1)
        pred_np = pred.cpu().detach().numpy()
        
        plt.subplot(131)
        plt.imshow(img_np[0,0,32,:,:])
        plt.subplot(132)
        plt.imshow(pred_np[0,0,32,:,:])
        plt.subplot(133)
        plt.imshow(lab_np[0,0,32,:,:])
        plt.savefig(debug_path+str(epoch)+"_"+str(batch_idx)+".png")
        
    def check_training_inputs(self,img,lab,output,epoch,batch_idx,debug_string):
        out_np = output.cpu().detach().numpy()
        lab_np = lab.cpu().detach().numpy()
        img_np = img.cpu().detach().numpy()
        
        debug_path = "H:/SDF-regression/ImageSegmentation/roi-pytorch/debug/" + debug_string  + "/"
        print("Debugging images saved at: ", debug_path)
       
        plt.subplot(231)
        plt.imshow(img_np[0,0,32,:,:])
        plt.subplot(234)
        plt.imshow(lab_np[0,0,32,:,:])
        
        plt.subplot(232)
        plt.imshow(img_np[0,0,:,32,:])
        plt.subplot(235)
        plt.imshow(lab_np[0,0,:,32,:])
        
        plt.subplot(233)
        plt.imshow(img_np[0,0,:,:,32])
        plt.subplot(236)
        plt.imshow(lab_np[0,0,:,:,32])

        plt.savefig(debug_path+str(epoch)+"_"+str(batch_idx)+".png")

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
