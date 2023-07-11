# ESPCN_paddle
This repository is implementation of the ["Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network"](https://arxiv.org/abs/1609.05158).

参考：https://github.com/yjn870/ESPCN-pytorch


## Requirements

- paddlepaddle 2.4.0
- paddleseg    2.8.0
- Numpy 1.15.4
- Pillow 5.4.1
- h5py 2.8.0
- tqdm 4.30.0


## Train

The 91-image, Set5 dataset converted to HDF5 can be downloaded from the links below.

| Dataset  | Scale | Type  | Link                                                         |
| -------- | ----- | ----- | ------------------------------------------------------------ |
| 91-image | 3     | Train | [Download](https://www.dropbox.com/s/4mv1v4qfjo17zg3/91-image_x3.h5?dl=0) |
| Set5     | 3     | Eval  | [Download](https://www.dropbox.com/s/9qlb94in1iqh6nf/Set5_x3.h5?dl=0) |

Otherwise, you can use `prepare.py` to create custom dataset.

```bash
python train.py --train-file "BLAH_BLAH/91-image_x3.h5" \
                --eval-file "BLAH_BLAH/Set5_x3.h5" \
                --outputs-dir "BLAH_BLAH/outputs" \
                --scale 3 \
                --lr 1e-3 \
                --batch-size 16 \
                --num-epochs 50 \
                --num-workers 0 \
                --seed 123                
```

## Test

Pre-trained weights can be found in BLAH_BLAH/outputs.

The results are stored in the same path as the query image.

```bash
python test.py --weights-file "BLAH_BLAH/outputs/x3/best.pdiparams" \
               --image-file "data/baboon.bmp" \
               --scale 3
```

## Results

PSNR was calculated on the Y channel.

### Set5

| Eval. Mat | Scale | Paper |
| --------- | ----- | ----- |
| PSNR      | 3     | 33.06 |


<table>
    <tr>
        <td><center>Original</center></td>
        <td><center>BICUBIC x3</center></td>
        <td><center>ESPCN x3 (23.84 dB)</center></td>
    </tr>
    <tr>
    	<td>
    		<center><img src="./data/baboon.bmp""></center>
    	</td>
    	<td>
    		<center><img src="./data/baboon_bicubic_x3.bmp"></center>
    	</td>
    	<td>
    		<center><img src="./data/baboon_espcn_x3.bmp"></center>
    	</td>
    </tr>
    <tr>
        <td><center>Original</center></td>
        <td><center>BICUBIC x3</center></td>
        <td><center>ESPCN x3 (25.32 dB)</center></td>
    </tr>
    <tr>
    	<td>
    		<center><img src="./data/comic.bmp""></center>
    	</td>
    	<td>
    		<center><img src="./data/comic_bicubic_x3.bmp"></center>
    	</td>
    	<td>
    		<center><img src="./data/comic_espcn_x3.bmp"></center>
    	</td>
    </tr>  
</table>




