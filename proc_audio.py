import CoreAudioML.miscfuncs as miscfuncs
import CoreAudioML.training as training
import CoreAudioML.dataset as dataset
import CoreAudioML.networks as networks
import argparse
from scipy.io.wavfile import write
import torch
import torch.nn as nn
import time

import json

# Example
# python3 proc_audio.py -l RNN-aidadsp-13b2 -i Data/test/aidadsp-13b2-input.wav -t Data/test/aidadsp-13b2-target.wav -o ./proc.wav

def parse_args():
    parser = argparse.ArgumentParser(
        description='''This script takes an input .wav file, loads it and processes it with a neural network model of a
                    device, i.e guitar amp/pedal, and saves the output as a .wav file. Optionally it can calculate the ESR
                    over a provided target .wav file, so that you can see how good the prediction was.''')
    parser.add_argument('--load_model', '-l', help="Json model file at the end of the training", default='model.json')
    parser.add_argument('--data_location', '-dl', default='./Data', help='Location of the "Data" directory')
    parser.add_argument('--input_file', '-i', default='', help='Location of the input file')
    parser.add_argument('--target_file', '-t', default='', help='Location of the target file')
    parser.add_argument('--output_file', '-o', default='', help='Location of the output file')
    parser.add_argument('--start', '-s', type=int, default=-1, help='Start point expressed in samples')
    parser.add_argument('--end', '-e', type=int, default=-1, help='End point expressed in samples')
    return parser.parse_args()


def proc_audio(args):
    print("Using %s file" % args.load_model)

    # Load network model from config file
    network_data = miscfuncs.json_load(args.load_model)
    network = networks.load_model(network_data)

    data = dataset.DataSet(data_dir='', extensions='')
    data.create_subset('input')
    data.load_file(args.input_file, set_names='input')

    if args.target_file:
        data.create_subset('target')
        data.load_file(args.target_file, set_names='target')

        lossESR = training.ESRLoss()
        lossDC = training.DCLoss()

    if args.start < 0:
        args.start = 0
    if args.end < 0:
        args.end = int(list(data.subsets['input'].data['data'][0].size())[0])

    print("Using start = %s, end = %s" % (str(args.start), str(args.end)))

    with torch.no_grad():
        output = network(data.subsets['input'].data['data'][0][args.start:args.end])

    if args.target_file:
        test_loss_ESR = lossESR(output, data.subsets['target'].data['data'][0][args.start:args.end])
        test_loss_ESR_p = lossESR(output, data.subsets['target'].data['data'][0][args.start:args.end], pooling=True)
        test_loss_DC = lossDC(output, data.subsets['target'].data['data'][0][args.start:args.end])
        write(args.output_file.rsplit('.', 1)[0]+'_ESR.wav', data.subsets['input'].fs, test_loss_ESR_p.cpu().numpy()[:, 0, 0])
        write(args.output_file.rsplit('.', 1)[0]+'_target.wav', data.subsets['input'].fs, data.subsets['target'].data['data'][0][args.start:args.end].cpu().numpy()[:, 0, 0])
        print("test_loss_ESR = %.6f test_loss_DC = %.6f" % (test_loss_ESR.item(), test_loss_DC.item()))

    # Output is in this format tuple(tensor, tuple(tensor, tensor))
    write(args.output_file, data.subsets['input'].fs, output.cpu().numpy()[:, 0, 0])


def main():
    args = parse_args()
    print(args)
    proc_audio(args)

if __name__ == '__main__':
    main()