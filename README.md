



There has already a lot of works demonstrating the increasing accuracy and robustness from additional training samples.

Through this simple trial, I wonder how accuracy and robustness vary across different number of trained samples.
Here, I apply the classical PGD attack with 20 iterations for robustness evaluation.

## Usage


    python STD.py cifar cifar10 --nums=100
    python AT.py cifar cifar10 --nums=100 -wd=0.0005 -lp=default --epochs=110

## Results



| NUMs  | Method | TA(%)  | RA(%)  |
| :---: | :----: | :----: | :----: |
|  100  |  STD   | 24.940 | 1.020  |
|  100  |   AT   | 26.720 | 7.380  |
| 1000  |  STD   | 60.980 | 0.030  |
| 1000  |   AT   | 50.110 | 12.850 |
| 5000  |  STD   | 80.050 | 0.020  |
| 5000  |   AT   | 64.690 | 22.510 |
| 10000 |  STD   | 84.230 | 0.010  |
| 10000 |   AT   | 70.100 | 33.030 |

