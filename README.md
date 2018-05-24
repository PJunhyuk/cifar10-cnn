# cifar10-cnn
Based on [pytorch/tutorials - cifar10_tutorial.py](https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py).  

## Results
###### f25505e322408afb8b11805890cbce44e23c33b7

conv1(3, 64, 5) -> pool -> conv2(64, 128, 5) -> pool -> conv3(128, 128, 3) -> fc

| Epoch | Train loss | Test loss | Accuracy |
|:-:|:-:|:-:|:-:|
| 1  | 1.462  | 1.209  | 56%  |
| 2  | 1.032  | 0.930  | 67%  |
| 3  | 0.830  | 0.884  | 69%  |
| 4  | 0.699  | 0.856  | 72%  |
| 5  | 0.593  | 0.810  | 72%  |
| 6  | 0.508  | 0.818  | 74%  |
| 7  | 0.433  | 0.869  | 72%  |
| 8  | 0.368  | 1.057  | 71%  |
| 9  | 0.319  | 0.971  | 72%  |
| 10 | 0.281  | 1.160  | 70%  |
| 11 | 0.262  | 1.175  | 72%  |
| 12 | 0.234  | 1.221  | 72%  |
| 13 | 0.217  | 1.421  | 72%  |
| 14 | 0.227  | 1.493  | 70%  |
| 15 | 0.224  | 1.486  | 72%  |

14x sec for train 1 epoch.

###### 

## Reference
[PyTorch 튜토리얼 4 - 분류기(classifier) 학습](http://bob3rdnewbie.tistory.com/317)
