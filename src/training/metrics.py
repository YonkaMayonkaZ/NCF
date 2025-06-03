import torch
import numpy as np

def metrics(model, test_loader, top_k):
    HR, NDCG = [], []
    
    for user, item, _ in test_loader:
        user = user.cuda() if torch.cuda.is_available() else user
        item = item.cuda() if torch.cuda.is_available() else item
        
        with torch.no_grad():
            predictions = model(user, item)
            _, indices = torch.topk(predictions, top_k)
            recommends = torch.take(item, indices).cpu().numpy()
        
        gt_item = item[0].item()
        HR.append(int(gt_item in recommends))
        
        ndcg = 0.0
        if gt_item in recommends:
            index = np.where(recommends == gt_item)[0][0]
            ndcg = 1.0 / np.log2(index + 2)
        NDCG.append(ndcg)
    
    return HR, NDCG
