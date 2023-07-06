import time

import numpy as np

def accuracy(ground_embeddings, 
             aerial_embeddings, 
             ground_labels, 
             top_k=[1,5,10]):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    ts = time.time()
    num_ground_embeddings = ground_embeddings.shape[0]
    num_aerial_embeddings = aerial_embeddings.shape[0]
    top_k.append(num_aerial_embeddings//100)
    results = np.zeros([len(top_k)])
    ground_embeddings_norm = np.sqrt(np.sum(ground_embeddings**2, axis=1, keepdims=True))
    aerial_embeddings_norm = np.sqrt(np.sum(aerial_embeddings**2, axis=1, keepdims=True))
    similarity = np.matmul(ground_embeddings/ground_embeddings_norm, 
                           (aerial_embeddings/aerial_embeddings_norm).transpose())

    for grd_embed_idx in range(num_ground_embeddings):
        ranking = np.sum((similarity[grd_embed_idx,:]>similarity[grd_embed_idx, ground_labels[grd_embed_idx]])*1.)
        for rank_idx, k in enumerate(top_k):
            if ranking < k:
                results[rank_idx] += 1.
    
    results = (results/num_ground_embeddings) * 100.
    print('Percentage-top1:{}, top5:{}, top10:{}, top1%:{}, time:{}'.\
          format(results[0], 
                 results[1], 
                 results[2], 
                 results[-1], 
                 time.time() - ts))
    
    return results[:2]