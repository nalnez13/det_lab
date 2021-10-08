import torch
import numpy as np
import cv2


def collater(data):
    """Data Loader에서 생성된 데이터를 동일한 shape으로 정렬해서 Batch로 전달

    Args:
        data ([dict]): albumentation Transformed 객체
        'image': list of Torch Tensor len == batch_size, item shape = ch, h, w
        'bboxes': list of list([x1, y1, w, h, cid])

    Returns:
        [dict]: 정렬된 batch data.
        'img': list of image tensor
        'annot': 동일 shape으로 정렬된 tensor [x1,y1,x2,y2] format
    """
    imgs = [s['image'] for s in data]
    bboxes = [torch.tensor(s['bboxes'])for s in data]
    batch_size = len(imgs)

    max_num_annots = max(annots.shape[0] for annots in bboxes)

    if max_num_annots > 0:
        padded_annots = torch.ones((batch_size, max_num_annots, 5)) * -1
        for idx, annot in enumerate(bboxes):
            if annot.shape[0] > 0:
                # To x1, y1, x2, y2
                annot[:, 2] += annot[:, 0]
                annot[:, 3] += annot[:, 1]
                padded_annots[idx, :annot.shape[0], :] = annot
    else:
        padded_annots = torch.ones((batch_size, 1, 5)) * -1

    return {'img': torch.stack(imgs), 'annot': padded_annots}


def visualize(images, bboxes, batch_idx=0):
    """batch data를 opencv로 visualize

    Args:
        images ([list]): list of img tensor
        bboxes ([tensor]): tensor data of annotations
                        shape == [batch, max_annots, 5(x1,y1,x2,y2,cid)]
                        max_annots 은 batch sample 중 가장 많은 bbox 갯수.
                        다른 sample 은 -1 로 패딩된 데이터가 저장됨.
        batch_idx (int, optional): [description]. Defaults to 0.
    """
    img = images[batch_idx].numpy()
    img = (np.transpose(img, (1, 2, 0))*255.).astype(np.uint8).copy()

    for b in bboxes[batch_idx]:
        x1, y1, x2, y2, cid = b.numpy()
        if cid > -1:
            img = cv2.rectangle(img, (int(x1), int(y1)),
                                (int(x2), int(y2)), (0, 255, 0))
    cv2.imshow('img', img)
    cv2.waitKey(0)
