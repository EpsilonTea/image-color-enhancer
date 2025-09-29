import torch
import numpy as np

def apply_lut_to_image(image, lut, lut_size):
    temp_image = image.clone()
    temp_image[[0, 2]] = temp_image[[2, 0]]

    # Ensure the temp_image is in the range [0, 1]
    temp_image = temp_image.clamp(0, 1)

    c, h, w = temp_image.shape

    flat_image = temp_image.view(-1, 3)

    lut_indices = flat_image * (lut_size - 1)
    lut_indices = lut_indices.reshape(3, h*w)

    xi = lut_indices[0].floor().long()
    yi = lut_indices[1].floor().long()
    zi = lut_indices[2].floor().long()

    dx = (lut_indices[0] - xi.float()).unsqueeze(1)
    dy = (lut_indices[1] - yi.float()).unsqueeze(1)
    dz = (lut_indices[2] - zi.float()).unsqueeze(1)

    xi = xi.clamp(0, lut_size - 2)
    yi = yi.clamp(0, lut_size - 2)
    zi = zi.clamp(0, lut_size - 2)

    c000 = lut[xi, yi, zi]
    c001 = lut[xi, yi, zi + 1]
    c010 = lut[xi, yi + 1, zi]
    c011 = lut[xi, yi + 1, zi + 1]
    c100 = lut[xi + 1, yi, zi]
    c101 = lut[xi + 1, yi, zi + 1]
    c110 = lut[xi + 1, yi + 1, zi]
    c111 = lut[xi + 1, yi + 1, zi + 1]
    
    c00 = (1 - dx) * c000 + dx * c100
    c01 = (1 - dx) * c001 + dx * c101
    c10 = (1 - dx) * c010 + dx * c110
    c11 = (1 - dx) * c011 + dx * c111

    c0 = (1 - dy) * c00 + dy * c10
    c1 = (1 - dy) * c01 + dy * c11

    c = (1 - dz) * c0 + dz * c1
    
    return(c.reshape(h, w, 3).permute(2, 0, 1))


def apply_weighted_luts_to_image(image, luts, weights, lut_size, masks):
    lut_transformed_images = torch.stack(
        [apply_lut_to_image(image, lut, lut_size) for lut in luts]
    )

    lut_transformed_images = lut_transformed_images.unsqueeze(0).expand(len(masks), -1, -1, -1, -1)
    
    weights = weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    
    masks = masks.unsqueeze(1).unsqueeze(1).expand(-1, len(luts), 3, -1, -1)

    final_image = (lut_transformed_images * masks * weights).sum(dim=0).sum(dim=0)

    return final_image.clamp(0, 1)


def apply_weighted_luts_to_batch(batch, luts, weights, lut_size, batch_masks):
    transformed_batch = []
    for i, image in enumerate(batch):
        transformed_image = apply_weighted_luts_to_image(image, luts, weights[i], lut_size, batch_masks[i])
        transformed_batch.append(transformed_image)

    return torch.stack(transformed_batch)

def read_lut(file_path):
    """
    Reads a .cube LUT file and returns the LUT as a numpy array.
    Assumes the LUT file has a format where the second line contains the LUT_3D_SIZE value.
    """
    lut_size = 0
    lut = []

    with open(file_path, 'r') as file:
        line = file.readline().strip()
        if line.startswith('LUT_3D_SIZE'):
            lut_size = int(line.split()[-1])
        else:
            raise ValueError("LUT file does not contain 'LUT_3D_SIZE' on the first line")

        # Now read the rest of the lines, which contain the LUT color values
        for line in file:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip comments
                lut.append(list(map(float, line.split())))

    lut = np.array(lut).reshape((lut_size, lut_size, lut_size, 3))
    return lut, lut_size


def get_luts():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    luts = []
    lut_path = "./luts"

    lut_files = [f'{lut_path}/identity.cube', f'{lut_path}/cool.cube', f'{lut_path}/colorful.cube', f'{lut_path}/warm.cube', f'{lut_path}/saturation.cube', f'{lut_path}/sad_warm.cube', f'{lut_path}/gray_to_blue.cube', f'{lut_path}/gamma.cube']


    for lut_file in lut_files:
        lut, lut_size = read_lut(lut_file)
        luts.append(lut)

    luts = [torch.from_numpy(lut).to(device) for lut in luts]
    luts = torch.stack(luts, dim=0)

    return luts