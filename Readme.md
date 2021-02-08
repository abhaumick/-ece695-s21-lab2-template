# ECE 695 CUDA Programming Part 1

## Professor Tim Rogers <br> TA: Abhishek Bhaumick

## Due on (TODO: Confirm)
# Introduction


The first assignment exploited data parallelism along a single dimension - with a direct one-to-one or many-to-one correlation between each thread and the data it processes.
In the second assignment we will be operating on 2-D data and look at ways to adapt some common operations for the GPU.

-----------------------------------------------------------
<br>

# PART A: Image Filters 
## Median Filtering

The [median filter](https://en.wikipedia.org/wiki/Median_filter) is a non-linear digital filtering technique, often used to remove noise from an image or signal [[1]](#1). For small to moderate levels of Gaussian noise, the median filter is demonstrably better than Gaussian blur at removing noise whilst preserving edges for a given, fixed window size. [[2]](#2).

While a number of efficient implementations of median filtering exists [[3]](#3), we will be using the relatively compute-intensive "simple" implementation, that precisely computes the median. Furthermore, to simplify the implementation, we will be using a cental symmetric rectangular / square window such that the pixel at the centre will be replaced by the median of all the pixels in

A simple description of the algorithm can be found [here](http://fourier.eng.hmc.edu/e161/lectures/smooth_sharpen/node2.html).


## Setting Up a Python Virtual Environment

Use the following commands to setup and use a python virtual environment. This step is essential to install python packages on scholar without modifying the built-in Python copy.

```bash
# Create a new python virtual environment in local directory `./python`
$ python3 -m venv ./python

# Activate your python virtual environment
# **NOTE** This step must be repeated every time to select your venv as default python distribution
$ source ./python/bin/activate

# Verify python path
$ which python3
/home/ ... /lab2/python/bin/python3

# Install required python packages
pip3 install numpy pillow

# Run script 
$ python3 makeImage.py 
No GUI element found - disabling GUI
usage: makeImage.py [-h] [-f FILE] [-l LEVEL] [-r RATIO]

Prepare Image for CUDA Programming Lab 2

optional arguments:
  -h, --help  show this help message and exit
  -f FILE     input image file path
  -l LEVEL    added noise intensity
  -r RATIO    Salt vs Pepper Ratio
usage: makeImage.py [-h] [-f FILE] [-l LEVEL] [-r RATIO]
makeImage.py: error: No GUI - must specify image filepath using -f 

```

## Using the `.bytes` file format 

Modern images are stored in compressed formats like `JPEG` which have their own storage formats and compression/decompression schemes. A `custom RAW Pixel` format (`.bytes`) has been defined for the purpose of this class to enable you to quickly convert any image into the RGB pixels and use it with the code developed as part of this assignment. The required Python scripts for interconversion and viewing the original and processed images have been provided.

- `makeBytesImage.py` can be used to convert any given image (tested with `.jpg` and `.tiff` formats) into a `.bytes` file.
- `viewBytesImage.py` can be used in a GUI-enabled environment to view the contents of a `.bytes` image. 


> **NOTE** : Viewing the images through any of this scripts will require you to run the scripts on a GUI enabled workspace (ThinClient or your local machines). 
> 
> - The scripts still work on console-only workspaces but will have the GIUs disabled.
> - In non-GUI environments, `makeBytesImage.py` will take console arguments and directly store the output into a `.bytes` file.



```C++
// .bytes metadata
typedef struct ImageDim_t
{
	uint32_t height;
	uint32_t width;
	uint32_t channels;
	uint32_t pixelSize;
} ImageDim;
```

The bytes format packs any image in a raw 8-bit pixel format comprised of a 16 byte metadata followed by RGB inteleaved data in `little-endian` format. 
- For the purpose of this assignment, _all images will have 3 channels and 8-bit (1-Byte) pixels_.
- The first 16 bytes of any .bytes file will have the metadata required to interpret the contents of the rest of the file (can be used to calculate dimensions and sizes)
- The remaining bytes in tne file will have RGB pixels interleaved.
- Pixels are stored in Row-Major format.

Look at the contents of `loadImageBytes()` in `cpuLib.cpp` to get an idea of storage format.

|  MetaData  | Px (0,0) | Px (0,1) |  ...  | Px (0,N-1) | Px (1,0) |  ...  | Px (N-1,N-1) |
|------------| -------- | ---------|-------|------------|----------|-------|--------------|
|            | R G B    | R G B    | R G B | R G B      | R G B    | R G B | R G B        |
|  16 bytes  | 3 bytes  | 3 bytes  |       | 3 bytes    | 3 bytes  |       | 3 bytes      |

## Median Filtering

<br>

# PART B: Pooling 


## CPU Implementation

A basic **incomplete** implementaton of a Max-Pool layer is provided inside `cpuLib.h`. It lays down the API and (after completion) can be used for checking your GPU implementation later.

1. Complete the CPU code for max pooling
1. Add provisions for padding and strides
	- implement padding by using conditional statements rather than by actually padding the data.

## GPU Implementation

#### Question

1. Parallelize accross batches ?

## References

<a id="1">[1]</a> 
[Wikipedia - Median Filtering](https://en.wikipedia.org/wiki/Median_filter)

<a id="2">[2]</a> 
Arias-Castro, Ery; Donoho, David L. Does median filtering truly preserve edges better than linear filtering?. Ann. Statist. 37 (2009), no. 3, 1172--1206. doi:10.1214/08-AOS604. https://projecteuclid.org/euclid.aos/1239369019

<a id="3">[3]</a> 
T. Huang, G. Yang and G. Tang, "A fast two-dimensional median filtering algorithm," in IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 27, no. 1, pp. 13-18, February 1979, doi: 10.1109/TASSP.1979.1163188. https://ieeexplore.ieee.org/document/1163188

<a id="4">[4]</a> 
Arce, G.R. (2004). Weighted Median Filters. In Nonlinear Signal Processing, G.R. Arce (Ed.). https://doi-org.ezproxy.lib.purdue.edu/10.1002/0471691852.ch6

#### Queries

- Use of shared memory ?



Let me break the tie and schedule the office hours at 
`4:00 PM to 5:00 PM on Thursday, 04 Feb`

The format will be as before - [Zoom](https://purdue-edu.zoom.us/j/93584887789?pwd=Qlg4VHpCNGdpNy9KQ292U2lrQ0xBZz09) with [Queup](https://www.queuplive.com/room/SMFBUYOI) for ordering

> abhaumic@purdue.edu is inviting you to a scheduled Zoom meeting. 
> 
> Join Zoom Meeting \
> https://purdue-edu.zoom.us/j/93584887789?pwd=Qlg4VHpCNGdpNy9KQ292U2lrQ0xBZz09\
> Meeting ID: 935 8488 7789 \
> Passcode: 823460 \
> One tap mobile \
> +16465588656,,93584887789#,,,,*823460# US (New York) \
> +13017158592,,93584887789#,,,,*823460# US (Washington DC) \
> Dial by your location \
>         +1 646 558 8656 US (New York) \
>         +1 301 715 8592 US (Washington DC) \
>         +1 312 626 6799 US (Chicago) \
>         +1 669 900 6833 US (San Jose) \
>         +1 253 215 8782 US (Tacoma) \
>         +1 346 248 7799 US (Houston) \
> Meeting ID: 935 8488 7789 \
> Passcode: 823460 \
> Find your local number: https://purdue-edu.zoom.us/u/aj1jX3lps 

Queup @ 
https://www.queuplive.com/room/SMFBUYOI

