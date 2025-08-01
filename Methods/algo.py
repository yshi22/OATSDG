from .base import *
class LabelDifference(nn.Module):
    def __init__(self, distance_type='l1'):
        super(LabelDifference, self).__init__()
        self.distance_type = distance_type

    def forward(self, labels):
        if self.distance_type == 'l1':
            return torch.abs(labels.unsqueeze(1) - labels.unsqueeze(0))
        elif self.distance_type == 'csm':
            labels = F.normalize(labels.view(labels.shape[0],-1), dim=1)
            return torch.matmul(labels, labels.T)
        elif self.distance_type == 'l1_mean':
            batch_size, feature_size = labels.shape[0], labels.shape[-1]
            labels = labels.view([batch_size, feature_size])
            k = 1
            center_index = feature_size // 2
            start_index = max(center_index - k, 0)
            end_index = min(center_index + k + 1, feature_size)
            labels = labels[:, start_index:end_index].mean(dim=1).view(-1)
            return torch.abs(labels.unsqueeze(1) - labels.unsqueeze(0))
        else:
            raise ValueError(self.distance_type)
    



class FeatureSimilarity(nn.Module):
    def __init__(self, similarity_type='l2'):
        super(FeatureSimilarity, self).__init__()
        self.similarity_type = similarity_type

    def forward(self, features):
        if self.similarity_type == 'l2':
            return - (features[:, None, :] - features[None, :, :]).norm(2, dim=-1)
        elif self.similarity_type == 'csm':
            features = F.normalize(features, dim=1)
            return torch.matmul(features, features.T)
        else:
            raise ValueError(self.similarity_type)


class ConOALoss(nn.Module):
    # Modified from Rank-N-Contrast
    # https://github.com/kaiwenzha/Rank-N-Contrast
    def __init__(self, zy_temperature, label_diff, zy_sim):
        super(ConOALoss, self).__init__()
        self.zy_t = zy_temperature
        self.label_diff_fn = LabelDifference(label_diff)
        self.zy_sim_fn = FeatureSimilarity(zy_sim)

    def forward(self, zy, labels):
        _device = zy.device
        label_diffs = self.label_diff_fn(labels)
        zy_logits = self.zy_sim_fn(zy).div(self.zy_t)
        zy_logits_max, _ = torch.max(zy_logits, dim=1, keepdim=True)
        zy_logits -= zy_logits_max.detach()
        zy_exp_logits = zy_logits.exp()
        n = zy.shape[0]
        zy_logits = zy_logits.masked_select((1 - torch.eye(n).to(_device)).bool()).view(n, n - 1)
        zy_exp_logits = zy_exp_logits.masked_select((1 - torch.eye(n).to(_device)).bool()).view(n, n - 1)
        label_diffs = label_diffs.masked_select((1 - torch.eye(n).to(_device)).bool()).view(n, n - 1)
        loss = 0.
        for k in range(n - 1):
            pos_logits = zy_logits[:, k]
            pos_label_diffs = label_diffs[:, k]
            neg_mask = (label_diffs >= pos_label_diffs.view(-1, 1)).float()
            pos_log_probs = pos_logits - torch.log((neg_mask * zy_exp_logits).sum(dim=-1))
            loss += - (pos_log_probs / (n * (n - 1))).sum()
        return loss
    
class ConDDLoss(nn.Module):
    # Modified from Supervised Contrastive Learning
    # https://github.com/XG293/SupConLoss
    def __init__(self, zyzd_temperature, zyzd_sim, min_diff = 0):
        super(ConDDLoss, self).__init__()
        self.zyzd_t = zyzd_temperature
        self.zyzd_sim_fn = FeatureSimilarity(zyzd_sim)
        self.min_diff = min_diff
        self.use_supConLoss = True
        
    def forward(self, zy, zd, domains, labels):
        _device = zy.device
        zyzd = torch.cat([zy, zd], dim=0)
        zyzd_logits = self.zyzd_sim_fn(zyzd).div(self.zyzd_t)
        zyzd_logits_max, _ = torch.max(zyzd_logits, dim=1, keepdim=True)
        zyzd_logits -= zyzd_logits_max.detach()
        zyzd_exp_logits = zyzd_logits.exp()
        n = zy.shape[0]
        domains = domains.view(-1, 1)
        labels = labels.view(-1, 1)
        label_mask = ((labels - labels.T).abs() > self.min_diff).float()
        domain_mask = torch.eq(domains, domains.T).float().to(_device)
        neg_domain_mask = 1 - domain_mask
        domain_mask[n:, n:] = domain_mask[n:, n:] * label_mask
        logits_mask = torch.ones((2*n, 2*n))
        logits_mask.fill_diagonal_(0)
        logits_mask = logits_mask.float().to(_device)
        domain_mask = domain_mask * logits_mask
        neg_domain_mask = neg_domain_mask * logits_mask
        zyzd_exp_logits = zyzd_exp_logits * logits_mask
        log_prob = zyzd_logits - torch.log(
            zyzd_exp_logits.sum(1, keepdim=True) if self.use_supConLoss else \
            (zyzd_exp_logits * neg_domain_mask).sum(1, keepdim=True)
        )
        mask_pos_pairs = domain_mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = -(domain_mask * log_prob).sum(1) / mask_pos_pairs
        return mean_log_prob_pos.mean()

class ConMILoss(nn.Module):
    def __init__(self, t = 0.07):
        super(ConMILoss, self).__init__()
        self.sim_fn = FeatureSimilarity('csm')
        self.t = t
    def forward(self, zyzd):
        _device = zyzd.device
        zyzd_logits = self.sim_fn(zyzd).div(self.t)
        zyzd_logits_max, _ = torch.max(zyzd_logits, dim=1, keepdim=True)
        zyzd_logits -= zyzd_logits_max.detach()
        zyzd_exp_logits = zyzd_logits.exp()
        size = len(zyzd) // 2
        label = torch.tensor([1] * size + [0] * size).view(-1, 1)
        domain_mask = torch.eq(label, label.T).float().to(_device)
        logits_mask = torch.ones((2*size, 2*size))
        logits_mask.fill_diagonal_(0)
        logits_mask = logits_mask.float().to(_device)
        domain_mask = domain_mask * logits_mask
        log_prob = zyzd_logits - torch.log(zyzd_exp_logits.sum(1, keepdim=True))
        mask_pos_pairs = domain_mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = -(domain_mask * log_prob).sum(1) / mask_pos_pairs
        return mean_log_prob_pos.mean()

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.device = args.device
        self.w_pred = args['w_pred']
        self.w_oa = args['w_oa']
        self.w_dd = args['w_dd']
        self.w_mi = args['w_mi']
        self.encoder,self.gy_encoder,self.gd_encoder,self.head,self.fc,self.mi_head = model_arch[args['dataset']]

        self.LOA = ConOALoss(args['zy_temperature'],args['label_diff'], args['zy_sim'])
        self.LDD = ConDDLoss(args['zyzd_temperature'], args['zyzd_sim'], args['min_diff'])
        self.LMI = ConMILoss(args['zyzd_temperature'])

    def freeze_layers(self):
        for param in self.gd_encoder.parameters():
            param.requires_grad = False
        for param in self.head.parameters():
            param.requires_grad = False
        for param in self.mi_head.parameters():
            param.requires_grad = False

    def forward(self, x):
        if self.args.use_batch_norm:
            x, min_vals, max_vals = batch_minmax_norm(x)

        x = self.encoder(x)
        gy_enc = self.gy_encoder(x).view(x.size(0), -1)
        gd_enc = self.gd_encoder(x).view(x.size(0), -1)

        if self.args.use_batch_norm:
            encode = torch.cat([gy_enc, min_vals, max_vals], dim=1)
        else:
            encode = gy_enc

        out = self.fc(encode)
        return out.view(-1), (gy_enc, gd_enc)


    def loss_function(self, x, y, d):
        out, (gy_enc, gd_enc) = self.forward(x)
        prediction_loss = F.mse_loss(out, y)

        zy_enc_label_val = self.args.n_domains
        zy_enc_labels = torch.full((gy_enc.size(0),), zy_enc_label_val, dtype=torch.long, device=gy_enc.device)
        con_label = torch.cat([zy_enc_labels, d], dim=0)

        _gy_enc = self.head(gy_enc)
        _gd_enc = self.head(gd_enc)
        loa = self.LOA(_gy_enc, y)
        ldd = self.LDD(_gy_enc, _gd_enc, con_label, y)

        zy = gy_enc.detach()
        zd = gd_enc.detach()
        zy_grl = GradReverse.apply(gy_enc)
        zyzd = torch.concat([zy_grl, zd], dim = 1)
        zyzd = self.mi_head(zyzd)
        perm_index = torch.randperm(zd.size(0))
        zy_zd = torch.concat([zy, zd[perm_index]], dim = 1)
        zy_zd = self.mi_head(zy_zd)

        lmi = self.LMI(torch.concat([zyzd, zy_zd], dim = 0))

        return  self.w_pred * prediction_loss + self.w_oa * loa + self.w_dd * ldd + self.w_mi * lmi, \
                             (self.w_pred * prediction_loss, self.w_oa * loa + self.w_dd * ldd + self.w_mi * lmi)

    def classifier(self, x, y):
        with torch.no_grad():
            if self.args.use_batch_norm:
                x, min_vals, max_vals = batch_minmax_norm(x)
            x = self.encoder(x)
            gy_enc = self.gy_encoder(x).view(x.size(0), -1)

            encode = gy_enc
            if self.args.use_batch_norm:
                encode = torch.cat([gy_enc, min_vals, max_vals], dim=1)

            pred = self.fc(encode).view(-1)
            if self.args.unnorm:
                pred = torch.relu(pred * self.args.std + self.args.mean)
                y = y * self.args.std + self.args.mean
        
        return pred, y
    
    def classifier_tsne(self, x, y):
        with torch.no_grad():
            if self.args.use_batch_norm:
                x, min_vals, max_vals = batch_minmax_norm(x)
            x = self.encoder(x)
            gy_enc = self.di_encoder(x).view(x.size(0), -1)
            gd_enc = self.ds_encoder(x).view(x.size(0), -1)

            encode = gy_enc
            if self.args.use_batch_norm:
                encode = torch.cat([gy_enc, min_vals, max_vals], dim=1)

            out = self.fc(encode)
            gy_enc = self.head(gy_enc)
            gd_enc = self.head(gd_enc)
        return out.view(-1), y, (gy_enc, gd_enc)


def res_plot_tsne(source_loaders, target_loader, model, classifier_tsne, idx2name, rep, first_batchs = None):
    model.eval()
    zy_reps = []
    true_y = []
    pred_y = []
    label_T_vs_per_S = []
    label_T_vs_all_S = []
    zy_zd_reps = []
    zy_zd_label = []

    with torch.no_grad():
        for idx, loader in enumerate(source_loaders):
            sample_batch = 0
            for (xs, ys, ds) in loader:
                xs, ys, ds = xs, ys, ds
                sample_batch += 1
                pred, y, (zy,zd) = classifier_tsne(xs, ys)
                zy_reps += zy.detach().cpu().tolist()
                label_T_vs_per_S += ds.detach().cpu().tolist()
                label_T_vs_all_S += [0] * len(xs)
                true_y += y.detach().cpu().tolist()
                pred_y += pred.detach().cpu().tolist()

                zy_zd_reps += zy.detach().cpu().tolist() + zd.detach().cpu().tolist()
                zy_zd_label += [0] * len(xs) + (ds + 3).detach().cpu().tolist()
                if first_batchs is not None and sample_batch > first_batchs:
                    break

        for idx, loader in enumerate(target_loader):
            sample_batch = 0
            for (xs, ys, ds) in loader:
                xs, ys, ds = xs, ys, ds
                sample_batch += 1
                pred, y, (zy,zd) = classifier_tsne(xs, ys)
                zy_reps += zy.detach().cpu().tolist()
                label_T_vs_per_S += [len(idx2name)-1] * len(xs)
                label_T_vs_all_S += [1] * len(xs)
                true_y += y.detach().cpu().tolist()
                pred_y += pred.detach().cpu().tolist()
                if first_batchs is not None and sample_batch > first_batchs:
                    break
    zy_reps = np.array(zy_reps)
    zy_zd_reps = np.array(zy_zd_reps)
    label_T_vs_per_S = np.array(label_T_vs_per_S)
    label_T_vs_all_S = np.array(label_T_vs_all_S)
    true_y = np.array(true_y)
    pred_y = np.array(pred_y)
    zy_zd_label = np.array(zy_zd_label)
    loc = t_sne(zy_reps)
    zy_zd_loc = t_sne(zy_zd_reps)

    t_sne_plot(loc, label_T_vs_all_S, ['source', 'target'],  f't_SNE_TvsAllS_{rep}.pdf', cmap = "rainbow")
    t_sne_plot_y(loc, true_y,  f't_SNE_true_y_{rep}.pdf', colarbar = False)
    t_sne_plot_y(zy_zd_loc, zy_zd_label,  f't_SNE_zizs_{rep}.pdf', colarbar = False)

def train_one_epoch(args, source_loaders, DEVICE, model, optimizer):
    model.train()
    train_loss = 0
    pred_loss = 0
    con_loss = 0 
    data_size = 0
    for source_loader in source_loaders:
        for x, y, d in source_loader:
            optimizer.zero_grad()

            loss_origin, (prediction_loss, oats_loss) = model.loss_function(x, y, d)
            loss_origin.backward()
            optimizer.step()

            train_loss += loss_origin * len(d)
            pred_loss += prediction_loss * len(d)
            con_loss += oats_loss * len(d)
            data_size += len(d)
    train_loss /= data_size
    pred_loss /= data_size
    con_loss /= data_size
    return train_loss, (prediction_loss, con_loss)


def get_accuracy(source_loaders, DEVICE, model, classifier_fn, batch_size, args):
    model.eval()
    y_preds, y_tures = [], []
    metrics = {}
    with torch.no_grad():
        for source_loader in source_loaders:
            for (xs, ys, ds) in source_loader:
                pred_y, ture_y  = classifier_fn(xs, ys)
                pred_y = pred_y.detach().cpu().view(-1) 
                ture_y = ture_y.detach().cpu().view(-1) 
                y_preds += pred_y.tolist()
                y_tures += ys.tolist()

        rmse = mean_squared_error(y_preds, y_tures) ** 0.5
        mae = mean_absolute_error(y_preds, y_tures)
        metrics['mse'] = rmse
        metrics['mae'] = mae

        return metrics


def train(model, DEVICE, optimizer, source_loaders, target_loader, args):
    metrics_recorder = Metrics_tracker(args.metric_list, 'mae', True)
    for e in range(args.n_epoch):
        avg_epoch_loss, (prediction_loss, cont_loss) = train_one_epoch(args, source_loaders, DEVICE, model, optimizer)
        if args.freeze_epoch is not None and e == args.freeze_epoch: model.freeze_layers()
        adjust_learning_rate(args.lr, args.lr_decay, e, args.n_epoch, optimizer)
        train_metrics = get_accuracy(source_loaders, DEVICE, model, model.classifier, args.batch_size, args)
        test_metrics = get_accuracy([target_loader], DEVICE, model, model.classifier, args.batch_size, args)
        metrics_recorder.update(train_metrics, test_metrics)
        best_eval_iter = metrics_recorder.get_best_eval_iter('mae')
        best_eval_mse = metrics_recorder.get_test_metric('mse', best_eval_iter)
        best_eval_mae = metrics_recorder.get_test_metric('mae', best_eval_iter)
        tqdm.tqdm.write('Epoch:[{}], avg loss: {:.4f}, y loss: {:.4f}, contrast loss: {:.4f}, test mse:{:.4f} mae:{:.4f}' \
                        .format(e + 1, avg_epoch_loss, prediction_loss, cont_loss, best_eval_mse, best_eval_mae))
    return best_eval_mae, best_eval_mse
