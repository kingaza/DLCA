import numpy as np
import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc1   = nn.Conv3d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv3d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class PostRes(nn.Module):
    def __init__(self, n_in, n_out, group_norm=True, stride = 1):
        super(PostRes, self).__init__()
        self.conv1 = nn.Conv3d(n_in, n_out, kernel_size = 3, stride = stride, padding = 1)
        self.norm1 = nn.GroupNorm(n_out//16, n_out) if group_norm else nn.BatchNorm3d(n_out)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv3d(n_out, n_out, kernel_size = 3, padding = 1)
        self.norm2 = nn.GroupNorm(n_out//16, n_out) if group_norm else nn.BatchNorm3d(n_out)

        self.ca = ChannelAttention(n_out)
        self.sa = SpatialAttention()

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv3d(n_in, n_out, kernel_size = 1, stride = stride),
                nn.GroupNorm(n_out//16, n_out) if group_norm else nn.BatchNorm3d(n_out))
        else:
            self.shortcut = None


    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        # print("cacaca")

        out = self.ca(out) * out
        out = self.sa(out) * out

        out += residual
        out = self.relu(out)
        return out


def hard_mining(neg_output, neg_labels, num_hard):
    _, idcs = torch.topk(neg_output, min(num_hard, len(neg_output)))
    neg_output = torch.index_select(neg_output, 0, idcs)
    neg_labels = torch.index_select(neg_labels, 0, idcs)
    return neg_output, neg_labels

class FocalLoss(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCELoss(reduction='none')
        #self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, probs, label):
        '''
        logits and label have same shape, and label data type is long
        args:
            logits: tensor of shape (N, ...)
            label: tensor of shape(N, ...)
        '''
        # compute loss
        probs = probs.float() # use fp32 if logits is fp16
        with torch.no_grad():
            alpha = torch.empty_like(probs).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha

        pt = torch.where(label==1, probs, 1-probs)
        ce_loss = self.crit(probs, label.float())
        loss = (alpha * torch.pow(1-pt, self.gamma) * ce_loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss

# Note: the names of pos_true, pos_false, neg_true, neg_false are not defined correctly
class Loss(nn.Module):

    def __init__(self, num_hard = 0, margin_range=[0.3, 0.7]):
        super(Loss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = FocalLoss() # nn.BCELoss()
        self.regress_loss = nn.SmoothL1Loss()
        self.num_hard = num_hard
        self.margin_range = margin_range

    def forward(self, output, labels, train = True):
        # print('output shape', output.shape)
        # print('labels shape', labels.shape)

        batch_size = labels.size(0)
        output = output.view(-1, 5)
        labels = labels.view(-1, 5)

        pos_idcs = labels[:, 0] > 0.5
        pos_idcs = pos_idcs.unsqueeze(1).expand(pos_idcs.size(0), 5)
        pos_output = output[pos_idcs].view(-1, 5)
        pos_labels = labels[pos_idcs].view(-1, 5)

        neg_idcs = labels[:, 0] < -0.5
        neg_output = output[:, 0][neg_idcs]
        neg_labels = labels[:, 0][neg_idcs]

        if self.num_hard > 0 and train:
            neg_output, neg_labels = hard_mining(neg_output, neg_labels, self.num_hard * batch_size)
        neg_prob = self.sigmoid(neg_output)

        pos_preds = []
        if len(pos_output)>0:
            pos_prob = self.sigmoid(pos_output[:, 0])
            pz, ph, pw, pd = pos_output[:, 1], pos_output[:, 2], pos_output[:, 3], pos_output[:, 4]
            lz, lh, lw, ld = pos_labels[:, 1], pos_labels[:, 2], pos_labels[:, 3], pos_labels[:, 4]
            regress_losses = [
                self.regress_loss(pz, lz),
                self.regress_loss(ph, lh),
                self.regress_loss(pw, lw),
                self.regress_loss(pd, ld)]
            regress_losses_data = [l.item() for l in regress_losses]
            classify_loss = 0.5 * self.classify_loss(pos_prob, pos_labels[:, 0]) + \
                            0.5 * self.classify_loss(neg_prob, neg_labels + 1)
                            
            pos_probs = pos_prob.detach().cpu().numpy()   
            pos_preds = list(pos_probs)

            pos_true = (pos_probs >= 0.5).sum()
            pos_margin = ((pos_probs>self.margin_range[0]) & (pos_probs<0.5)).sum()
            pos_total = len(pos_prob)

            # print('Output', pos_output[:, 1:4].cpu().detach().numpy(), 'prob', pos_prob.cpu().detach().numpy())
            # print('label', pos_labels[:, 1:4].cpu().numpy())

            # data_shape = labels.shape
            # idcs = pos_idcs.detach().cpu().numpy()[:,0]
            # idcs = np.arange(len(idcs))[idcs]
            # print('positive index', idcs)
            # print([np.unravel_index(i, data_shape[:-1]) for i in idcs])

            # print('Positive prob.', pos_prob.detach().cpu().numpy())
            # false_pos_labels = pos_labels[pos_prob.detach()<0.5, 1:4].detach().cpu().numpy()
            # false_pos_output = pos_output[pos_prob.detach()<0.5, 1:4].detach().cpu().numpy()
            # print('false Positive of labels position', len(false_pos_labels), false_pos_labels)        
            # print('false Positive of output position', len(false_pos_output), false_pos_output)           

        else:
            regress_losses = [0,0,0,0]
            classify_loss =  0.5 * self.classify_loss(neg_prob, neg_labels + 1)
            pos_true = 0
            pos_margin = 0
            pos_total = 0
            regress_losses_data = [0,0,0,0]
        classify_loss_data = classify_loss.item()

        loss = classify_loss
        for regress_loss in regress_losses:
            loss += regress_loss
        
        neg_probs = neg_prob.detach().cpu().numpy()
        neg_false = (neg_probs < 0.5).sum()
        neg_margin = ((neg_probs>0.5) & (neg_probs<self.margin_range[1])).sum()
        neg_total = len(neg_prob)

        num_total = len(labels[:, 0])

        # np.set_printoptions(precision=2, suppress=True)
        # print('positive labels', pos_labels.detach().cpu().numpy())
        # print('positive output', pos_output.detach().cpu().numpy())
        # print('negative labels', neg_labels.detach().cpu().numpy())  
        # print('negative output', neg_output.detach().cpu().numpy())     
        # print('positive:', 0 if len(pos_output)==0 else pos_true.detach().cpu().numpy(), '/', pos_total)        
        # print('negative:', 0 if len(neg_output)==0 else neg_true.detach().cpu().numpy(), '/', neg_total)    

        # len=6
        total_losses = [loss, classify_loss_data] + regress_losses_data
        # len=7+preds
        loss_info = [pos_true, pos_total, neg_false, neg_total, num_total, pos_margin, neg_margin] + pos_preds

        return total_losses + loss_info


class GetPBB(object):
    def __init__(self, config):
        self.stride = config['stride']
        self.anchors = np.asarray(config['anchors'])

    def __call__(self, output, thresh=-3, ismask=False):
        stride = self.stride
        anchors = self.anchors
        output = np.copy(output)
        offset = (float(stride) - 1) / 2
        output_size = output.shape
        oz = np.arange(offset, offset + stride * (output_size[0] - 1) + 1, stride)
        oh = np.arange(offset, offset + stride * (output_size[1] - 1) + 1, stride)
        ow = np.arange(offset, offset + stride * (output_size[2] - 1) + 1, stride)

        output[:, :, :, :, 1] = oz.reshape((-1, 1, 1, 1)) + output[:, :, :, :, 1] * anchors.reshape((1, 1, 1, -1))
        output[:, :, :, :, 2] = oh.reshape((1, -1, 1, 1)) + output[:, :, :, :, 2] * anchors.reshape((1, 1, 1, -1))
        output[:, :, :, :, 3] = ow.reshape((1, 1, -1, 1)) + output[:, :, :, :, 3] * anchors.reshape((1, 1, 1, -1))
        output[:, :, :, :, 4] = np.exp(output[:, :, :, :, 4]) * anchors.reshape((1, 1, 1, -1))
        mask = output[..., 0] > thresh
        xx,yy,zz,aa = np.where(mask)

        output = output[xx,yy,zz,aa]
        if ismask:
            return output,[xx,yy,zz,aa]
        else:
            return output