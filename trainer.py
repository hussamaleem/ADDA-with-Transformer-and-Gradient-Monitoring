import torch
import config


class Trainer():
    def __init__(self,
                 src_encoder,
                 trg_encoder,
                 discr,
                 device,
                 discr_optimizer,
                 trg_optimizer,
                 discr_criterion,
                 trg_criterion,
                 rul_predictor,
                 gm,
                 gm_method,
                 lr_sched_trg,
                 lr_sched_discr,
                 use_gm
                 ):

        self.src_encoder = src_encoder
        self.trg_encoder = trg_encoder
        self.discr = discr
        self.device = device
        self.discr_optimizer = discr_optimizer
        self.trg_optimizer = trg_optimizer
        self.discr_criterion = discr_criterion
        self.trg_criterion = trg_criterion
        self.rul_predictor = rul_predictor
        self.gm = gm
        self.gm_method=gm_method
        self.lr_sched_trg = lr_sched_trg
        self.lr_sched_discr = lr_sched_discr
        self.use_gm = gm

        
    def training_loop(self, 
                      src_loader,
                      trg_loader,
                      batch_size,
                      epoch,
                      inference):
        
        discr_train_loss = 0
        trg_encoder_loss = 0
        total_iterations = 0
        latent_space_feat = []
        latent_space_label = []
        lr_list = []

        
        self.discr.train()
        self.trg_encoder.train()

        for i,(src_loader, trg_loader) in enumerate(zip(src_loader, trg_loader)):

            self.discr_optimizer.zero_grad()
            self.trg_optimizer.zero_grad()
            
            src_data, _ = src_loader
            trg_data, _ = trg_loader
            min_len = min([src_data.shape[0], trg_data.shape[0]])
            src_data = src_data[:min_len]
            trg_data = trg_data[:min_len]
            
            src_data = src_data.to(self.device)
            trg_data = trg_data.to(self.device)
            
            src_feat = self.src_encoder(src_data)
            trg_feat = self.trg_encoder(trg_data)
            if ('Adaptive' in self.gm_method) and (epoch+1 == config.start_epoch):
                self.gm.get_mmd_loss(src_feat.detach(), trg_feat.detach(), start_epoch=True)
            concat_feat = torch.cat((src_feat, trg_feat), dim = 0)
            
            src_labels = torch.ones((src_feat.size(0))).long().to(self.device)
            trg_labels = torch.zeros((trg_feat.size(0))).long().to(self.device)
            inverted_labels = torch.ones((trg_feat.size(0))).long().to(self.device)
            concat_labels = torch.cat((src_labels,trg_labels), dim = 0)

            if all(config.latent_space_conditions):
                latent_space_feat.extend([concat_feat.detach().cpu()])
                latent_space_label.extend([concat_labels.detach().cpu()])
                
                    
            discr_pred = self.discr(concat_feat)
            discr_loss = self.discr_criterion(discr_pred,concat_labels)
            discr_loss.backward()
            

            self.discr_optimizer.step()
            
            self.discr_optimizer.zero_grad()
            self.trg_optimizer.zero_grad()
            
            trg_feat = self.trg_encoder(trg_data)
            trg_pred = self.discr(trg_feat)
            trg_loss = self.discr_criterion(trg_pred, inverted_labels)
            trg_loss.backward()
            
            if self.use_gm:
                self.gm.method_selector(gm_method=self.gm_method, 
                                        epoch=None if 'Momentum_GM' in self.gm_method else epoch,
                                        src_feat=src_feat.detach() if 'Adaptive' in self.gm_method else None,
                                        trg_feat=trg_feat.detach() if 'Adaptive' in self.gm_method else None,
                                        batch_num=i if 'Adaptive' in self.gm_method else None)
            
            self.trg_optimizer.step()
            
            current_lr = self.lr_sched_trg.get_last_lr()
            lr_list.append(float(current_lr[0])) 
            if not inference:
                self.lr_sched_trg.step()
                self.lr_sched_discr.step()

            discr_train_loss += discr_loss.item()
            trg_encoder_loss += trg_loss.item()
            total_iterations += 1

        total_discr_loss = discr_train_loss/total_iterations
        total_trg_loss = trg_encoder_loss/total_iterations

        return total_discr_loss, total_trg_loss, latent_space_feat, latent_space_label, lr_list

    def testing_loop(self, test_data):
        
        self.trg_encoder.eval()
        self.rul_predictor.eval()
        with torch.no_grad():
            
            test_loss = 0
            total_iterations = 0
            for i, (data, label) in enumerate(test_data):
                
                data, label = data.to(self.device), label.to(self.device)
                enc_pred = self.trg_encoder(data)
                pred = self.rul_predictor(enc_pred)
                loss = torch.sqrt(self.trg_criterion(pred, label))
                pred = pred.detach().cpu()
                label = label.detach().cpu()
                test_loss+=loss.detach().item()
                if i == 0:  
                    
                    predictions = pred
                    targets = label
                else:
                    
                    predictions = torch.cat((predictions, pred), dim = 0)
                    targets = torch.cat((targets, label), dim = 0)
                total_iterations += 1
            total_test_loss = test_loss/total_iterations
            
            
            return predictions, targets, total_test_loss


