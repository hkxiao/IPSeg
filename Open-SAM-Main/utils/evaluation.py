r""" Evaluate mask prediction """
import torch


class Evaluator:
    r""" Computes intersection and union between prediction and ground-truth """
    @classmethod
    def initialize(cls):
        cls.ignore_index = 255

    @classmethod
    def classify_prediction(cls, pred_mask, gt_mask, query_ignore_idx=None):
        print(pred_mask.shape, gt_mask.shape)
        print(torch.unique(pred_mask), torch.unique(gt_mask))
        # Apply ignore_index in PASCAL-5i masks (following evaluation scheme in PFE-Net (TPAMI 2020))
        if query_ignore_idx is not None:
            assert torch.logical_and(query_ignore_idx, gt_mask).sum() == 0
            query_ignore_idx *= cls.ignore_index
            gt_mask = gt_mask + query_ignore_idx
            pred_mask[gt_mask == cls.ignore_index] = cls.ignore_index

        # compute intersection and union of each episode in a batch
        area_inter, area_pred, area_gt = [],  [], []
        print(pred_mask.shape, gt_mask.shape)
        for _pred_mask, _gt_mask in zip(pred_mask, gt_mask):
            print(_pred_mask.shape, _gt_mask.shape)
            _inter = _pred_mask[_pred_mask == _gt_mask]
            #print(_inter.shape)
            if _inter.size(0) == 0:  # as torch.histc returns error if it gets empty tensor (pytorch 1.5.1)
                _area_inter = torch.tensor([0, 0], device=_pred_mask.device)
            else:
                _area_inter = torch.histc(_inter, bins=2, min=0, max=1)
            #print(_area_inter)
            area_inter.append(_area_inter)
            area_pred.append(torch.histc(_pred_mask, bins=2, min=0, max=1))
            area_gt.append(torch.histc(_gt_mask.to(torch.uint8), bins=2, min=0, max=1))
        
        # print(area_inter[0].shape)
        # raise NameError
        area_inter = torch.stack(area_inter).t()
        area_pred = torch.stack(area_pred).t()
        area_gt = torch.stack(area_gt).t()
        
        print(area_pred, area_gt, area_inter)
        raise NameError
        area_union = area_pred + area_gt - area_inter

        return area_inter, area_union
