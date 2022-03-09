# Pineda

Pineda recurrent backpropagation algorithm

## Install

Requires morphologica (https://github.com/ABRG-Models/morphologica)

build in the usual way: mkdir build cd build cmake .. make cd ..

## Overview

This project is essentially an implementation of the recurrent backpropagation algorithm for tranining neural networks, in a supervised manner to map a given set of inputs to a given set of outputs. The algorithm has the advantage over the commonly used backpropagation algorithm that it can be used to train networks with any topolgy (backprop is restricted to feed-forward networks only). Networks with recurrent connectivity can display attractor dynamics, hence it is important to allow the dynamics of recurrent networks to settle before connection weights are adjusted, and to detect when the dynamics have failed to settle. The resulting algorithm was introduced by Pineda in the following journal article:

Pineda, FJ. (1987) Generalization of back-propagation to recurrent neural networks. Physical Review Letters, 59, 2229.
