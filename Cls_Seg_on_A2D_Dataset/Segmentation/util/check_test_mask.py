import numpy as np
import argparse
import pickle

def check_masks(masks):
    """
    check the validity of the predicted mask
    :param masks:
    """
    assert len(masks) == 1044, 'the number of predicted masks in testing set should be 1053 instead of {}'.format(len(masks))
    for mask in masks:
        assert mask.dtype == np.uint8, 'the mask data type should be np.uint8!'
        assert 0 <= mask.min() and mask.max() <= 43, 'the range of the mask should be [0, 43]'
    print('Passed the validity test!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--masks', type=str, default='output/test_pred.pkl',
                        help='pickle file path for predicated masks')
    args = parser.parse_args()

    with open(args.masks, 'rb') as f:
        masks = pickle.load(f)
    check_masks(masks)
