import torch
import torch.optim as optim
import torch.utils.data
import os
import shutil
import pdb
import logging
import numpy as np
from muat.util import *
import json
import zipfile
import glob
import pandas as pd
from copy import deepcopy

logger = logging.getLogger(__name__)

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 4
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.001 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = True

    show_loss_interval = 10
    gfm_embedding = False # whether to use GFM embedding or not
    use_CLS_token = False
    
    # checkpoint settings
    save_ckpt_path = None
    string_logs = None
    num_workers = 32 # for DataLoader
    ckpt_name = 'model'
    args = None
    
    # Weight averaging settings
    use_ema = True  # Use Exponential Moving Average
    ema_decay = 0.95  # EMA decay rate
    use_swa = False  # Use Stochastic Weight Averaging
    swa_start_epoch = 20  # Start SWA after this epoch
    swa_lr = 0.05  # SWA learning rate factor (multiplied by initial LR)
    swa_freq = 1  # Update SWA weights every N batches


    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class ExponentialMovingAverage:
    """
    Maintains exponential moving averages of model parameters.
    """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow_params = {}
        self.backup_params = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = param.data.clone()
    
    def update(self):
        """Update the moving averages of model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                self.shadow_params[name].mul_(self.decay).add_(
                    param.data, alpha=1 - self.decay
                )
    
    def apply_shadow(self):
        """Apply the moving averaged parameters to the model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                self.backup_params[name] = param.data.clone()
                param.data.copy_(self.shadow_params[name])
    
    def restore(self):
        """Restore the original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup_params:
                param.data.copy_(self.backup_params[name])
        self.backup_params = {}
    
    def state_dict(self):
        """Return the state dict of shadow parameters."""
        return {name: param.clone() for name, param in self.shadow_params.items()}
    
    def load_state_dict(self, state_dict):
        """Load shadow parameters from state dict."""
        self.shadow_params = state_dict

class StochasticWeightAveraging:
    """
    Implements Stochastic Weight Averaging (SWA).
    """
    def __init__(self, model):
        self.model = model
        self.swa_model = None
        self.swa_n = 0
        
    def update(self):
        """Update SWA model with current model weights."""
        if self.swa_model is None:
            self.swa_model = deepcopy(self.model)
            self.swa_n = 1
        else:
            self.swa_n += 1
            for param_swa, param_model in zip(
                self.swa_model.parameters(), self.model.parameters()
            ):
                param_swa.data.mul_(self.swa_n - 1).add_(param_model.data).div_(self.swa_n)
    
    def get_swa_model(self):
        """Return the SWA model."""
        return self.swa_model
    
    def state_dict(self):
        """Return the state dict of SWA model."""
        if self.swa_model is not None:
            return self.swa_model.state_dict()
        return None
    
    def load_state_dict(self, state_dict):
        """Load SWA model from state dict."""
        if state_dict is not None:
            if self.swa_model is None:
                self.swa_model = deepcopy(self.model)
            self.swa_model.load_state_dict(state_dict)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.global_acc = 0
        self.pd_logits = []
        
        # Initialize weight averaging
        self.ema = None
        self.swa = None
        if self.config.use_ema:
            self.ema = ExponentialMovingAverage(model, decay=self.config.ema_decay)
        if self.config.use_swa:
            self.swa = StochasticWeightAveraging(model)

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        
        self.complete_save_dir = self.config.save_ckpt_dir
        # print configuration of trainer
        print(f"Batch size: {self.config.batch_size}")
        print(f"Use GFM embedding: {self.config.gfm_embedding}")
        print(f"Max epochs: {self.config.max_epochs}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Weight decay: {self.config.weight_decay}")
        print(f"Checkpoint path: {self.config.save_ckpt_path}")
        print(f"Checkpoint name: {self.config.ckpt_name}")
        print(f"Training on: {self.device}")
        print(f"Use EMA: {self.config.use_ema} (decay: {self.config.ema_decay})")
        print(f"Use SWA: {self.config.use_swa} (start epoch: {self.config.swa_start_epoch})")
        print(f"Use CLS token: {self.config.use_CLS_token}")

    def batch_train(self):
        model = self.model
        model = model.to(self.device)

        if self.config.save_ckpt_dir is not None:
            os.makedirs(self.config.save_ckpt_dir, exist_ok=True) 

        model = torch.nn.DataParallel(model).to(self.device)
        
        # Re-initialize EMA and SWA after DataParallel
        if self.config.use_ema:
            self.ema = ExponentialMovingAverage(model, decay=self.config.ema_decay)
        if self.config.use_swa:
            self.swa = StochasticWeightAveraging(model)
            
        # optimizer = optim.SGD(model.parameters(), lr=self.config.learning_rate, momentum=0.9,weight_decay=self.config.weight_decay)
        # use RADAM optimizer
        optimizer = optim.RAdam(
            model.parameters(),
            lr=self.config.learning_rate,
            # betas=self.config.betas,
            weight_decay=self.config.weight_decay
        )

        # cosine decay learning rate scheduler
        if self.config.lr_decay:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.config.max_epochs, 
                eta_min=self.config.learning_rate * 0.1
            )
        else:
            scheduler = None
            
        # SWA scheduler if enabled
        swa_scheduler = None
        if self.config.use_swa:
            swa_scheduler = optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=self.config.learning_rate * self.config.swa_lr,
                max_lr=self.config.learning_rate,
                step_size_up=100,
                mode='triangular',
                cycle_momentum=False
            )

        trainloader = torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            num_workers=self.config.num_workers, 
            pin_memory=True
        )
        valloader = torch.utils.data.DataLoader(
            self.test_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=self.config.num_workers, 
            pin_memory=True
        )

        self.global_acc = 0
        self.save_checkpoint_v3(self.config.save_ckpt_dir)
        # make log file for validation accuracy
        log_file = os.path.join(self.config.save_ckpt_dir, 'validation_accuracy.log')
        fp_log =  open(log_file, 'w')
        fp_log.write("Epoch\tValidation Accuracy\n")

        for e in range(self.config.max_epochs):
            running_loss = 0
            model.train(True)

            train_corr = []

            for batch_idx, (data, target, sample_path) in enumerate(trainloader):
                
                string_data = target
                numeric_data = data
                numeric_data = numeric_data.to(self.device)
                class_keys = [x for x in string_data.values() if not isinstance(x, list)]
                class_values = []
                for x in string_data.keys():
                    values = string_data[x]
                    if not isinstance(values, list):
                        class_values.append(values)
                #pdb.set_trace()
                if len(class_values)>1:
                    target = torch.stack(class_values, dim=1)
                elif len(class_values)==1:
                    target = class_values[0].unsqueeze(dim=1)
                target = target.to(self.device)
                
                # forward the model
                with torch.set_grad_enabled(True):

                    optimizer.zero_grad()
                    #pdb.set_trace()
                    # logits, loss = model(numeric_data, target)
                    vis = False 
                    if batch_idx % 20 == 19:
                        vis = True
                    logits, _ = model(numeric_data, vis=vis)
                    
                    loss1 = F.cross_entropy(logits["first_logits"], target[:, 0].squeeze())
                    loss2 = F.cross_entropy(logits["second_logits"], target[:, 1].squeeze())
                    loss = loss1 + loss2

                    if isinstance(logits, dict):
                        logit_keys = ["first_logits", "second_logits"]
                        train_corr_inside = []

                        for nk, lk in enumerate(logit_keys):
                            logit = logits[lk]
                            _, pred = torch.max(logit, dim=1)  # Get the index of the max log-probability
                            # Ensure target is on the same device as pred
                            target_on_device = target[:, nk].to(pred.device)  # Move target to the same device as pred
                            train_corr_inside.append((pred == target_on_device).sum().item())

                        if len(train_corr) == 0:
                            train_corr = np.zeros(len(logit_keys))
                        train_corr += np.asarray(train_corr_inside)
                    else:
                        pass
                    
                    loss.backward()
                    #And optimizes its weights here
                    optimizer.step()
                    
                    # Update EMA after each batch
                    if self.config.use_ema and self.ema is not None:
                        self.ema.update()
                    
                    # Update SWA if enabled and after start epoch
                    if self.config.use_swa and e >= self.config.swa_start_epoch:
                        if batch_idx % self.config.swa_freq == 0:
                            self.swa.update()
                            if swa_scheduler is not None:
                                swa_scheduler.step()
                    
                    running_loss += loss.item()
                    train_acc = train_corr / len(self.train_dataset)

                    if batch_idx % self.config.show_loss_interval == 0:
                        show_text = "Epoch {} - Batch ({}/{}) - Mini-batch Training loss: {:.4f}".format(e+1,batch_idx , len(trainloader) , running_loss/(batch_idx+1))
                        for x in range(len(logit_keys)):
                            show_text = show_text + ' - Training Acc {}: {:.2f}'.format(x+1,train_acc[x])
                        print(show_text)
                        
            show_text = "Epoch {} - Full-batch Training loss: {:.4f}".format(e+1, running_loss/(batch_idx+1))
            for x in range(len(logit_keys)):
                show_text = show_text + ' - Training Acc {}: {:.2f}'.format(x+1,train_acc[x])
            
            if self.config.lr_decay and (not self.config.use_swa or e < self.config.swa_start_epoch):
                scheduler.step()  # Step the scheduler after each epoch

            print(show_text)

            #validation
            test_loss = 0
            test_correct = []
            
            model.train(False)
            
            # Apply EMA weights for validation if enabled
            if self.config.use_ema and self.ema is not None:
                print("Applying EMA weights for validation...")
                self.ema.apply_shadow()
                
            for batch_idx_val, (data, target, sample_path) in enumerate(valloader):

                string_data = target
                numeric_data = data
                numeric_data = numeric_data.to(self.device)
                class_keys = [x for x in string_data.values() if not isinstance(x, list)]
                class_values = []
                for x in string_data.keys():
                    values = string_data[x]
                    if not isinstance(values, list):
                        class_values.append(values)
                #pdb.set_trace()
                if len(class_values)>1:
                    target = torch.stack(class_values, dim=1)
                elif len(class_values)==1:
                    target = class_values[0].unsqueeze(dim=1)
                target = target.to(self.device)

                # forward the model
                with torch.set_grad_enabled(False):
                    # logits, loss = model(numeric_data, target)    
                    logits, _ = model(numeric_data)    
                    loss1 = F.cross_entropy(logits["first_logits"], target[:, 0].squeeze())
                    loss2 = F.cross_entropy(logits["second_logits"], target[:, 1].squeeze())
                    loss = loss1 + loss2
                    test_loss += loss.item()

                    if isinstance(logits, dict):
                        logit_keys = [x for x in logits.keys() if 'logits' in x]
                        test_correct_inside = []
                        for nk, lk in enumerate(logit_keys):
                            logit = logits[lk]
                            _, predicted = torch.max(logit.data, 1)                            

                            logits_cpu = logit.detach().cpu().numpy()
                            logit_filename = 'val_{}.tsv'.format(lk)
                            if batch_idx_val == 0:
                                f = open(self.complete_save_dir + logit_filename, 'w+')
                                target_handler = self.config.target_handler[nk]
                                header_class = target_handler.classes_
                                write_header = "\t".join(header_class)
                                f.write(write_header)
                                f.write('\ttarget_name\tsample')
                                f.close()
                                
                            f = open(self.complete_save_dir + logit_filename, 'a+')
                            for i_b in range(len(sample_path)):
                                f.write('\n')
                                logits_cpu_flat = logits_cpu[i_b].flatten()
                                logits_cpu_list = logits_cpu_flat.tolist()
                                write_logits = [f'{i:.8f}' for i in logits_cpu_list]
                                target_handler = self.config.target_handler[nk]
                                target_name = target_handler.inverse_transform([target[:, nk].detach().cpu().numpy().tolist()[i_b]])[0]
                                write_logits.append(str(target_name))
                                write_logits.append(sample_path[i_b])
                                write_header = "\t".join(write_logits)
                                f.write(write_header)
                            f.close()

                            # Ensure target is on the same device as predicted
                            target_on_device = target[:,nk].to(predicted.device)  # Move target to the same device as predicted
                            test_correct_inside.append(predicted.eq(target_on_device.view_as(predicted)).sum().item())
                        if len(test_correct) == 0:
                            test_correct = np.zeros(len(logit_keys))
                        test_correct += np.asarray(test_correct_inside)
                    else:
                        pass   

            # Restore original weights after validation if EMA was applied
            if self.config.use_ema and self.ema is not None:
                self.ema.restore()
                
            test_loss /= (batch_idx_val+1)
            test_acc = test_correct[0] / len(self.test_dataset) #accuracy based on first target
            print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, test_correct[0], len(self.test_dataset), 100. * test_acc))
            
            fp_log.write(f"{e+1}\t{test_acc:.4f}\n")
            fp_log.flush()
            #pdb.set_trace()
            modelname_substring = f"_epoch_{e+1}_acc_{test_acc:.3f}"
            self.save_checkpoint_v3(self.config.save_ckpt_dir, modelname_substring=modelname_substring)

            if test_acc >= self.global_acc:
                self.global_acc = test_acc
                # print(self.global_acc)
                for nk,lk in enumerate(logit_keys):
                    logit_filename = 'val_{}.tsv'.format(lk)
                    shutil.copyfile(self.complete_save_dir + logit_filename, self.complete_save_dir + 'best_' + logit_filename)
                    os.remove(self.complete_save_dir + logit_filename)
                
                zip_name = self.config.ckpt_name + modelname_substring + '.pthx'
                ckpt_path = os.path.join(self.config.save_ckpt_dir, zip_name)
                shutil.copyfile(ckpt_path, self.config.save_ckpt_dir + 'best_ckpt.pthx')

    def unziping_from_package_installation(self):
        pkg_ckpt = resource_filename('muat', 'pkg_ckpt')
        pkg_ckpt = ensure_dirpath(pkg_ckpt)

        all_zip = glob.glob(pkg_ckpt+'*.zip')
        if len(all_zip)>0:
            for checkpoint_file in all_zip:
                with zipfile.ZipFile(checkpoint_file, 'r') as zip_ref:
                    zip_ref.extractall(path=pkg_ckpt)
                os.remove(checkpoint_file) 

    def make_json_serializable(self,obj):
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")  # List of row dicts
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, set):
            return list(obj)
        elif hasattr(obj, '__class__') and obj.__class__.__name__ == 'LabelEncoderFromCSV':
            return {
                "class_to_idx": obj.class_to_idx,
                "idx_to_class": obj.idx_to_class,
                "classes_": obj.classes_
            }
        else:
            return obj

    def save_model_config_to_json(self,config, filepath: str):
        def recursive_serialize(obj):
            if isinstance(obj, dict):
                return {k: recursive_serialize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_serialize(v) for v in obj]
            else:
                return self.make_json_serializable(obj)

        serialisable_dict = {
            k: recursive_serialize(v)
            for k, v in config.__dict__.items()
        }
        # Set save_ckpt_dir to empty string in the serialized dict
        if 'save_ckpt_dir' in serialisable_dict:
            serialisable_dict['save_ckpt_dir'] = ''
            
        with open(filepath, "w") as f:
            json.dump(serialisable_dict,f)

    def save_dict_to_json(self,data, filepath: str):
        """Helper function to save dictionary data to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(data, f)

    def save_dataframe_to_json(self,df, filepath: str):
        """Helper function to save pandas DataFrame to JSON file."""
        data = df.to_dict(orient="records")
        self.save_dict_to_json(data, filepath)

    def save_checkpoint_v3(self, save_dir: str = None, use_averaged_weights: bool = True, modelname_substring = ""):
        """
        Save the current model state and configuration in v3 format.
        This breaks down the checkpoint into separate files for better organization.
        
        Args:
            save_dir (str, optional): Directory to save the checkpoint. If None, uses config.save_ckpt_path
            use_averaged_weights (bool): Whether to use averaged weights (EMA/SWA) if available
        """
        if save_dir is None:
            save_dir = self.config.save_ckpt_path
        if save_dir is None:
            raise ValueError("No save directory specified. Either provide save_dir or set config.save_ckpt_path")
            
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Determine which weights to save
        
        weight_averaging_info = {
            'use_ema': self.config.use_ema,
            'use_swa': self.config.use_swa,
            'averaged_weights_used': False
        }
        
        if self.config.use_ema and self.ema is not None:
            self.ema.apply_shadow()

        if use_averaged_weights:
            if self.config.use_ema and self.ema is not None:
                # Apply EMA weights temporarily
                
                print("Using EMA weights for saving...")
                weights_to_save = self.model.state_dict()
                
                weight_averaging_info['averaged_weights_used'] = True
                weight_averaging_info['averaging_method'] = 'ema'
                weight_averaging_info['ema_decay'] = self.config.ema_decay
            elif self.config.use_swa and self.swa is not None and self.swa.swa_model is not None:
                # Use SWA weights
                weights_to_save = self.swa.state_dict()
                weight_averaging_info['averaged_weights_used'] = True
                weight_averaging_info['averaging_method'] = 'swa'
                weight_averaging_info['swa_start_epoch'] = self.config.swa_start_epoch
        else:
            print("Using non-EMA weights for saving...")
            weights_to_save = self.model.state_dict()

        # Prepare checkpoint data
        checkpoint = {
            'weight': weights_to_save,
            'target_handler': self.config.target_handler,
            'model_config': self.model.config,
            'trainer_config': self.config,
            'dataloader_config': self.train_dataset.config,
            'model_name': self.model.__class__.__name__,
            'motif_dict': self.model.config.dict_motif,
            'pos_dict': self.model.config.dict_pos,
            'ges_dict': self.model.config.dict_ges,
            'weight_averaging_info': weight_averaging_info
        }
        
        # Save EMA and SWA states separately if they exist
        if self.ema is not None:
            checkpoint['ema_state'] = self.ema.state_dict()
        if self.swa is not None and self.swa.swa_model is not None:
            checkpoint['swa_state'] = self.swa.state_dict()
        
        # Save weights
        weights_path = os.path.join(save_dir, 'weight.pth')
        torch.save(checkpoint['weight'], weights_path)
        
        # Save EMA/SWA states if present
        if 'ema_state' in checkpoint:
            ema_path = os.path.join(save_dir, 'ema_state.pth')
            torch.save(checkpoint['ema_state'], ema_path)
        if 'swa_state' in checkpoint:
            swa_path = os.path.join(save_dir, 'swa_state.pth')
            torch.save(checkpoint['swa_state'], swa_path)

        # Save target handlers
        for idx, handler in enumerate(checkpoint['target_handler']):
            filepath = os.path.join(save_dir, f'target_handler_{idx+1}.json')
            self.save_dict_to_json({
                "class_to_idx": handler.class_to_idx,
                "idx_to_class": handler.idx_to_class,
                "classes_": handler.classes_
            }, filepath)

        # Save configs
        configs = {
            'model_config': checkpoint['model_config'],
            'trainer_config': checkpoint['trainer_config'],
            'dataloader_config': checkpoint['dataloader_config']
        }
        
        for name, config in configs.items():
            filepath = os.path.join(save_dir, f'{name}.json')
            self.save_model_config_to_json(config, filepath)
        
        # Save model name
        self.save_dict_to_json(checkpoint['model_name'], os.path.join(save_dir, 'model_name.json'))
        
        # Save weight averaging info
        self.save_dict_to_json(checkpoint['weight_averaging_info'], 
                             os.path.join(save_dir, 'weight_averaging_info.json'))

        # Save dictionaries
        dicts = {
            'motif_dict': checkpoint['motif_dict'],
            'pos_dict': checkpoint['pos_dict'],
            'ges_dict': checkpoint['ges_dict']
        }
        
        for name, df in dicts.items():
            filepath = os.path.join(save_dir, f'{name}.json')
            self.save_dataframe_to_json(df, filepath)

        # Create zip file
        zip_name = self.config.ckpt_name + modelname_substring + '.pthx'
        zip_path = os.path.join(save_dir, zip_name)
        
        # Get all .json and .pth files
        files_to_zip = []
        for ext in ['.json', '.pth']:
            files_to_zip.extend(glob.glob(os.path.join(save_dir, f'*{ext}')))
        
        # Create zip file
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file in files_to_zip:
                zipf.write(file, os.path.basename(file))
        
        # Clean up individual files
        for file in files_to_zip:
            os.remove(file)  
        logger.info(f"Checkpoint saved to {zip_path} (averaged weights: {weight_averaging_info['averaged_weights_used']})")

        if self.config.use_ema and self.ema is not None:
            # Restore original weights after saving
            self.ema.restore()

        return zip_path