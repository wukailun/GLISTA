# GLISTA
source code of GLISTA

# Sparse Coding with Gated Learned ISTA
This repository contains an implementation of the simulation experiments in Sparse Coding with Gated Learned ISTA. The code base is based from
the LISTA-CPSS repo (https://github.com/xchen-tamu/linear-lista-cpss).

# Dependencies
* [tensorflow](https://www.tensorflow.org/)

# Usage
The code supports GLISTA, LISTA, LAMP, LISTA-SS, and LISTA-CP-SS. To run our experiment, use "main.py".
For GLISTA with a combination gain gate functions with coupled parameters under SNR = 10dB, run:

```bash
python main.py --net GLISTA_cp \ 
			   --SNR 10 \
			   --gpu 0 \
			   -M 250 \
			   -N 500 \
			   -gain \
			   -fixval False \
			   -u \
			   -a 5.0 \
			   -uf combine
For GLISTA with an inverse overshoot gate functions under SNR = 40dB, run:

```bash
python main.py --net GLISTA \ 
			   --SNR 40 \
			   --gpu 0 \
			   -M 250 \
			   -N 500 \
			   -o \
			   -fixval False \
			   -a 1.0 \
			   -uf inv \
```
