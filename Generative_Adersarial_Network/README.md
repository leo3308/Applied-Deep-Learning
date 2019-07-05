# Generative Adversarial Network

## Task Description
In this task , I will generate the picture as similar as the cartoon picture dataset

The dataset link : https://google.github.io/cartoonset/download.html

* The real image :

![](https://github.com/leo3308/Applied-Deep-Learning/blob/master/Generative_Adersarial_Network/img/real.png)

## how to train model

```
bash ./train.sh /path/to/the/data/root/dir 
```
after you run this code
it will automatically generate the picture in my report

## how to generate the specified label images

```
bash ./cgan.sh /path/to/your/label.txt /path/to/output/dir
```

## Experiment

* AC_GAN with training steps 50000

![](https://github.com/leo3308/Applied-Deep-Learning/blob/master/Generative_Adersarial_Network/img/AC_50000.png)

* W_GAN_GP with training steps 80000 and batch size 32

![](https://github.com/leo3308/Applied-Deep-Learning/blob/master/Generative_Adersarial_Network/img/WGANGP_batch32_80000.png)

* W_GAN_GP with training steps 38000 and batch size 128

![](https://github.com/leo3308/Applied-Deep-Learning/blob/master/Generative_Adersarial_Network/img/WGANGP_batch128_38000.png)
