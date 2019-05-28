In this project, I build deep learning models for classification and segmentation on A2D dataset, using Google Cloud. 
A2D dataset contains 3782 videos with at least 99 instances per valid actor-action tuple and videos are labeled with both pixel-level actors and actions for sampled frames. It considers jointly various types of actors undergoing various actions.
Details about A2D dataset:http://web.eecs.umich.edu/~jjcorso/r/a2d/



## Preparation

Before start working on a specific task, please do the following preparation on your Google Cloud.

- **Clone the repository**

  Please use the following command to clone this repository:

  ```bash
  git clone --recursive https://github.com/hzx829/Cls_Seg_on_A2D_Dataset.git
  ```

  If there is any updates of the repository, please use the following commands to update:

  ```bash
  git submodule update --remote --merge
  git pull --recurse-submodules
  ```

  cd to the cloned repo:

  ```bash
  cd Cls_Seg_on_A2D_Dataset

  ```

- **Environment Configuration**

  1. Download and install miniconda from https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

  2. Install the virtual enviroment and its dependencies by:

     ```bash
     conda env create -f env.yml
     ```

  3. Activate the virtual environment by (please remember to activate virtual environment everytime you login on google cloud):

     ```bash
     conda activate pytorch_0_4_1
     ```

  4. Then, install Pytorch 0.4.1 and torchvision

     ```bash
     conda install pytorch=0.4.1 cuda92 -c pytorch
     conda install torchvision
     ```

  5. Install the ffmpeg via

     ```bash
     sudo apt install ffmpeg
     ```

- **Download A2D dataset**

  Please make sure you are at the `Cls_Seg_on_A2D_Dataset` directory.

  1. Download the A2D dataset

     ```bash
     curl http://www.cs.rochester.edu/~cxu22/t/249S19/A2D.tar.gz --output A2D.tar.gz
     ```

  2. Decompress the tar ball and remove tar ball.

     ```bash
     tar xvzf A2D.tar.gz
     rm A2D.tar.gz
     ```

  3. Extract frames from videos

     (Tip: Since it takes a long time to extract frames from video, you can execute the command in  `screen` or `tmux`, in case the disconnection happens.)

     ```bash
     python extract_frames.py
     ```


