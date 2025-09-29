import itertools
import numpy as np
import math
import random
import torch
import torch.nn.functional as F
from PIL import Image

def distance(p1, p2):
	p1, p2 = np.array(p1), np.array(p2)
	return np.sum((p1 - p2) ** 2) ** 0.5

def sample_bins(img_pixel_cnt, bin_cnt=16):
	bin_range = 256 // bin_cnt # the pixel range each bin contain

	tmp = {}
	for x in itertools.product(range(bin_cnt),repeat=3):
		tmp[x] = {'val': np.array([0,0,0]), 'cnt': 0}
	for pixel, cnt in img_pixel_cnt.items():
		idx = tuple([c // bin_range for c in pixel])
		tmp[idx]['val'] += np.array(pixel) * cnt
		tmp[idx]['cnt'] += cnt

	res = {}
	for bin_item in tmp.values():
		if bin_item['cnt'] != 0:
			res[tuple((bin_item['val'] / bin_item['cnt']))] = bin_item['cnt']

	return res

def init_means(bins, k=5):

	def attenuation(color,last_mean):
		return 1 - math.exp(((distance(color, last_mean) / 80) ** 2) * -1)

	res = []
	bins = {k: v for k, v in sorted(bins.items(), key=lambda item: item[1], reverse=True)}
	# for color, cnt in bins.items():
	for _ in range(k):
		for color,cnt in bins.items():
			if color not in res: 
				res.append(color)
				break
		bins = {k: v * attenuation(k,res[-1]) for k, v in bins.items()}
		bins = {k: v for k, v in sorted(bins.items(), key=lambda item: item[1], reverse=True)}

	return res

def k_means(bins, k=5, init_mean=True, max_iter=100, black=True):
	if init_mean is False: means = random.sample(list(bins),k)
	else: means = init_means(bins, k)
	if black: means.append([0, 128, 128])
	means = np.array(means)
	mean_cnt = means.shape[0]

	#cluster_cnt = [0 for i in range(mean_cnt)]
	cluster_cnt = np.zeros(mean_cnt)
	for _ in range(max_iter):
		cluster_sum = [np.array([0,0,0],dtype=float) for i in range(mean_cnt)]
		cluster_cnt = np.zeros(mean_cnt)
		for color, cnt in bins.items():
			color = np.array(color)	
			dists = [distance(color,mean) for mean in means]
			cluster_th = dists.index(min(dists))
			cluster_sum[cluster_th] += color * cnt
			cluster_cnt[cluster_th] += cnt

		new_means = [cluster_sum[i] / cluster_cnt[i] if cluster_cnt[i] > 0 else [0,0,0] for i in range(k)]
		if black: new_means.append([0,128,128])
		new_means = np.array(new_means)
		if (new_means == means).all(): break
		else: means = new_means

	arg_th = np.argsort(means[:k], axis=0)[:,0][::-1]

	return means[arg_th], cluster_cnt[arg_th]
    
def image_cluster(image, k=5):
    ratio = image.size(3) / image.size(2)
    image_r = F.interpolate(image, size=(512, int(512 * ratio)), mode='bilinear')
    img = Image.fromarray((image_r.squeeze().permute(1,2,0)*255).type(torch.uint8).cpu().numpy())
    colors = img.getcolors(img.size[0] * img.size[1])
    bins = {}
    for count, pixel in colors:
        bins[pixel] = count
    bins = sample_bins(bins)
    means, means_weight = k_means(bins, k=k, init_mean=True)
    
    return means

class PaletteGenerate(object):
    def __init__(self, gpu_id=0, k=5):
        self.gpu_id = gpu_id
        self.k = k

    def means_to_palette(self, means):
        means = means.repeat(1,1,64,64)
        palette = torch.cat(torch.split(means,1,0), dim=3)
        return palette

    def forward(self, x):
        means = image_cluster(x, self.k) 
        means = torch.from_numpy(means).unsqueeze(2).unsqueeze(3).cuda(self.gpu_id).type(torch.float32)/255.
        palette_img = self.means_to_palette(means)

        distance = []
        for i in range(self.k):
            distance.append(torch.sqrt(torch.sum((x - means[i:i+1,...])**2, dim=1, keepdim=True)))
        distance = torch.cat(distance, dim=1)
        mask_distance, mask_label = torch.min(distance, dim=1, keepdim=True)
        mask = torch.zeros((1,self.k,x.size(2),x.size(3)), dtype=torch.float32).to(x.device)
        for i in range(self.k):
            try:
                max_dist = torch.max(mask_distance[mask_label==i])
                mask[:,i:i+1,...] = torch.exp(-torch.sum(torch.pow(x - means[i:i+1,...],2),dim=1)/(2*torch.pow(max_dist/2.,2)))
            except:
                pass
        return mask, palette_img