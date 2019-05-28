# Actor-Action Segmentation 


## Task description 
In this task, I achieve fully-supervised image semantic segmentation.

The task equals to do pixel-level classification -- to predict the class label for each pixel of an image.

There are 44 categories, including background. Each category is a pair of (actor, action).

The following is the requirements for this task. By default, the working directory is the project root directory

## Method
I use FCN, SegNet and UNet on this task.
FCN has better pixel accuracy. SegNet and UNet has better edge segmentation.
Finally, I ensemble FCN and SegNet(SegNet's performance is close to UNet, and SegNet is lighter network) to get final result.

Here is the 3 papers: 
[FCN](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf).
[UNet](https://arxiv.org/pdf/1505.04597.pdf).
[SegNet](https://arxiv.org/pdf/1511.00561.pdf).
#### Accuracy and time analysis 
| epoch | miou%(val/test) | class accuracy%(val/test) | train time(h) |
|-------|-----------------|---------------------------|---------------|
| 33    | 21.33/-         | 32.16/-                   |3.0            |
| 47    | 23.37/22.96     | 32.77/32.41               |4.2            |

#### Qualitative Example
Segmentation Visualization Result at epoch 33 and 47. The first row shows ground truth, the second row shows our results.

Epoch 33
![Epoch18](img/iter000000020000.jpg "Segmentation Visualization Result at Epoch 18. First row is gt, second row is result")

Epoch 47
![Epoch18](img/iter000000064000.jpg "Segmentation Visualization Result at Epoch 60. First row is gt, second row is result")

## Dataloader 

Test the data loader by

```bash
python loader/a2d_dataset.py
```


## Evaluation
For evaluation of the validation and test set, you need to evaluate them with our provided evaluation code `a2d_eval.py`.
The ground truth label for the validation set is provided, however, test label are not. Hence, 
you need to take cautions with the following steps for evaluation:


1. For testing, you need to store the predicted masks with the image loading order into a python list
   , e.g. [mask_1, mask_2, ..., mask_n], where each mask_i has the same height and width with the input image_i, e.g. the input image_i with shape (h, w, 3) and the mask_i with shape (h, w). The value of mask_i is interger ranging [0, 43] ,indicating 44 classes, with value type **np.uint8**.

2. After you stored the masks in a list, you need to dump the list into pickle file.

    Please use the following command to store the list  
    ```python
    import pickle
    
    # A series of operations to get the mask list as mask_list
    
    with open(save_file_name, 'wb') as f:
        pickle.dump(mask_list, f)
        
    # where save_file_name is the name of the file you want to dump the list to. Usually, you will name the save_file_name with extention .pkl
    
    ```
    The saved val masks should be around 205M.

3. You need to evaluate the validation set using the provided evaluation script `eval.py`. The way to use evaluate your segmentation result is 

    ```bash
    python model/eval.py --gt_label $YOUR_GT_MASK_FILE --pred_label $YOUR_PRED_MASK_FILE
    ```
