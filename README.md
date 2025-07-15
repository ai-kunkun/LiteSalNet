# 📢 LiteSalNet
A PyTorch implementation of LiteSalNet for Remote Sensing Salient Object Detection.

[TGRS has accepted this work!!!](https://ieeexplore.ieee.org/document/10945380)

We updated the LiteSalNet result graph！

We updated the LiteSalNet main model code！

The training code can be found at [SeaNet](https://github.com/MathLee/SeaNet). You only need to modify the model output. Dr.Li has done a very standard job in this regard.

<p align="center">
  <br>
  <a href="https://ieeexplore.ieee.org/document/10945380">
    <img src="https://img.shields.io/badge/Paper-IEEE-green?style=for-the-badge&logo=adobeacrobatreader&logoColor=white&labelColor=66cc00&color=94DD15" alt="Paper PDF">
  </a>
  <a href="https://ai-kunkun.github.io/Niagara_page/">
    <img src="https://img.shields.io/badge/Project-Page-orange?style=for-the-badge&logo=Google%20chrome&logoColor=white&labelColor=D35400" alt="Project Page">
  </a>
  <a href="https://github.com/ai-kunkun/LiteSalNet">
    <img src="https://img.shields.io/badge/Code-Github-blue?style=for-the-badge&logo=github&logoColor=white&labelColor=181717" alt="Code Github">
  </a>
  <br>
</p>

  
# 🦉 Network Architecture
![LiteSalNet Architecture](https://github.com/ai-kunkun/LiteSalNet/blob/main/image/LiteSalNet.png)

# 📝 Requirements
- Python 3.7
- PyTorch 1.9.0

# 🎉 Saliency maps
![LiteSalNet Architecture](https://github.com/ai-kunkun/LiteSalNet/blob/main/image/table.png)

# 🏃‍♂️ Data
Download this dataset and put it into datasets.

[LiteSalNet_data](https://pan.baidu.com/s/1JXwvfIvSVv0lXrDaNwxXuQ?pwd=AZXD) (code: AZXD) 
# 🚀 Training
Run train_LiteSalNet.py.

# 🧩 Pre-trained model and testing
Download the following pre-trained model and put them in ./models/LiteSalNet/, then run test_LiteSalNet.py. 

# 🎆 Result
Download
Download the LiteSalnet model [result graph](https://pan.baidu.com/s/1w-jO8Y9HuY72X94NsZceww?pwd=AZXD) (code:AZXD)

# 🛠️ Evaluation Tool
You can use the [evaluation tool (MATLAB version)](https://github.com/MathLee/MatlabEvaluationTools) to evaluate the above saliency maps.

# 📖 Citation
        @ARTICLE{10945380,
                author = {Ai, Zhenxin and Luo, Huilan and Wang, Jianqin},
                title = {A Lightweight Multistream Framework for Salient Object Detection in Optical Remote Sensing},
                journal = {IEEE Transactions on Geoscience and Remote Sensing},
                volume = {63},
                year = {2025},
                doi = {10.1109/TGRS.2025.3555647},
                }
                
