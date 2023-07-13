# **CST-Net: Lesion Segmentation via Hybrid Convolution and Stack Transformer Network**

## Requirements
1. pytorch==1.10.0
2. Res2Net required weights file (already uploaded to Baidu.com), link below:
   Link: https://pan.baidu.com/s/1Hpazf2A8UkT1nffdMALU2Q?pwd=pyku 
   Extract code: pyku



## 🚨 IMPORTANT: Utilize Masks for Precise Edge Detection 🚨

get_edges.py 

This file generates (2 * 2), (3 * 3), and (5 * 5) edges respectively using mask images.
When running one of them, you need to comment out the rest. After modifying the directory, you can run it directly to generate the corresponding edges images



## Dataset

To apply the model to a custom dataset, the data tree should be constructed as:
``` 
    ├── dataset
          ├── images
                ├── image_1.png
                ├── image_2.png
                .......
                ├── image_n.png
          ├── masks
                ├── image_1.png
                ├── image_2.png
                .......
                ├── image_n.png
          ├── edges
                ├── image_1.png
                ├── image_2.png
                .......
                ├── image_n.png
```




## 💡 Train
 just  run trian.py 

Before training ,make sure you change the parameter following: train_path, train_save, epoch, batchsize, trainsize(it is better not to change it)，and backbone.



## 💡 Test
Test the image using the .pth weights from the training.

Before running the test.py, make sure to change the parameters data_path, pth_path,save_path and testize (it os better not to change it).





## 🌟 Acknowledgement
The codes are modified from the Inf-Net (https://github.com/DengPingFan/Inf-Net)

If you encounter any issues, please feel free to reach out to me via email at [666zy666@163.com](mailto:666zy666@163.com).
